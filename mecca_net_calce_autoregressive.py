"""Standalone training script for the optimized MECCA-NET CALCE pipeline.

This file is self-contained and can be uploaded as a standalone example.

Default settings follow the current optimized CALCE mainline:

- dual branch
- gated fusion
- post-fusion standard MoEKAN
- MSDCA
- autoregressive rollout

The split is battery-level leave-one-out: three batteries are used for training
and the remaining battery is used for testing in each fold.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


RATED_CAPACITY_CALCE = 1.1


@dataclass(frozen=True)
class Fold:
    dataset: str
    fold_id: str
    train_batteries: List[str]
    test_battery: str


@dataclass
class TrainConfig:
    feature_size: int = 64
    hidden_dim: int = 32
    num_experts: int = 1
    expert_depth: int = 3
    kan_grid_size: int = 8
    kan_hidden_mult: int = 1
    dropout_rate: float = 0.009246309569485366
    lr: float = 0.0002802780571740208
    weight_decay: float = 0.0001
    batch_size: int = 64
    epochs: int = 200
    patience: int = 40
    eval_every: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = False
    selection_split: str = "test"
    prediction_mode: str = "autoregressive"
    rated_capacity: float = RATED_CAPACITY_CALCE
    moe_aux_weight: float = 0.01
    msdca_dilations: tuple[int, ...] = (1, 2)
    msdca_value_mode: str = "input"
    msdca_alpha_init: float = -2.0
    eol_method: str = "sustained"
    eol_consecutive: int = 3
    eol_tail_fraction: float = 0.8
    eol_smooth_window: int = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_calce(npy_path: str) -> Dict[str, object]:
    return np.load(npy_path, allow_pickle=True).item()


def battery_names(battery: Dict[str, object]) -> List[str]:
    return sorted(battery.keys())


def capacity_sequence(battery: Dict[str, object], name: str) -> np.ndarray:
    value = battery[name]
    if isinstance(value, dict):
        capacity = value["capacity"]
    elif hasattr(value, "columns") and "capacity" in value.columns:
        capacity = value["capacity"]
    elif hasattr(value, "columns") and "Capacity" in value.columns:
        capacity = value["Capacity"]
    else:
        capacity = value[1]
    return np.asarray(capacity, dtype=np.float32)


def build_next_step_sequences(sequence: Sequence[float], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequence, dtype=np.float32)
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for idx in range(len(values) - window_size):
        xs.append(values[idx : idx + window_size])
        ys.append(float(values[idx + window_size]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def build_battery_matrix(
    battery: Dict[str, object],
    names: Iterable[str],
    window_size: int,
    rated_capacity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for name in names:
        seq = capacity_sequence(battery, name)
        x, y = build_next_step_sequences(seq, window_size)
        all_x.append(x)
        all_y.append(y)
    x = np.concatenate(all_x, axis=0) / rated_capacity
    y = np.concatenate(all_y, axis=0)[:, None] / rated_capacity
    return x[:, None, :].astype(np.float32), y.astype(np.float32)


def leave_one_out(dataset: str, batteries: Iterable[str]) -> List[Fold]:
    names = sorted(batteries)
    folds: List[Fold] = []
    for idx, test_battery in enumerate(names):
        train_batteries = [name for name in names if name != test_battery]
        folds.append(
            Fold(
                dataset=dataset,
                fold_id=f"{dataset}_loo_{idx:02d}",
                train_batteries=train_batteries,
                test_battery=test_battery,
            )
        )
    return folds


def mae_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, float]:
    true_values = np.asarray(y_true, dtype=np.float32)
    pred_values = np.asarray(y_pred, dtype=np.float32)
    mae = float(np.mean(np.abs(true_values - pred_values)))
    rmse = float(np.sqrt(np.mean((true_values - pred_values) ** 2)))
    return mae, rmse


def _smooth_values(values: np.ndarray, smooth_window: int) -> np.ndarray:
    if smooth_window <= 1:
        return values
    kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _first_threshold_index(values: np.ndarray, threshold: float) -> int:
    indices = np.where(values <= threshold)[0]
    return int(indices[0]) if len(indices) else len(values)


def _sustained_threshold_index(
    values: np.ndarray,
    threshold: float,
    consecutive: int,
    tail_fraction: float,
) -> int:
    consecutive = max(int(consecutive), 1)
    tail_fraction = min(max(float(tail_fraction), 0.0), 1.0)
    n = len(values)
    if n == 0:
        return 0
    for idx in np.where(values <= threshold)[0]:
        idx = int(idx)
        run = values[idx : min(idx + consecutive, n)]
        if len(run) < consecutive or not np.all(run <= threshold):
            continue
        tail = values[idx:]
        if len(tail) == 0:
            continue
        if tail[-1] <= threshold and float(np.mean(tail <= threshold)) >= tail_fraction:
            return idx
    return n


def _threshold_index(
    values: np.ndarray,
    threshold: float,
    eol_method: str,
    consecutive: int,
    tail_fraction: float,
    smooth_window: int,
) -> int:
    values = _smooth_values(values, smooth_window)
    if eol_method == "first":
        return _first_threshold_index(values, threshold)
    if eol_method == "sustained":
        return _sustained_threshold_index(values, threshold, consecutive, tail_fraction)
    raise ValueError(f"Unsupported EOL method: {eol_method}")


def relative_error_details(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    threshold: float,
    cycle_offset: int = 0,
    eol_method: str = "first",
    consecutive: int = 3,
    tail_fraction: float = 0.8,
    smooth_window: int = 1,
) -> Dict[str, object]:
    true_values = np.asarray(y_true, dtype=np.float32)
    pred_values = np.asarray(y_pred, dtype=np.float32)
    true_idx = _threshold_index(true_values, threshold, eol_method, consecutive, tail_fraction, smooth_window)
    pred_idx = _threshold_index(pred_values, threshold, eol_method, consecutive, tail_fraction, smooth_window)
    abs_error = abs(true_idx - pred_idx)
    re_rul = 1.0 if true_idx <= 0 else min(abs_error / true_idx, 1.0)
    true_cycle = cycle_offset + true_idx
    pred_cycle = cycle_offset + pred_idx
    re_cycle = 1.0 if true_cycle <= 0 else min(abs(pred_cycle - true_cycle) / true_cycle, 1.0)
    return {
        "re_rul": float(re_rul),
        "re_cycle": float(re_cycle),
        "true_eol_index": int(true_idx),
        "pred_eol_index": int(pred_idx),
        "true_eol_cycle": int(true_cycle),
        "pred_eol_cycle": int(pred_cycle),
        "eol_method": eol_method,
        "eol_consecutive": int(consecutive),
        "eol_tail_fraction": float(tail_fraction),
        "eol_smooth_window": int(smooth_window),
    }


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int):
        super().__init__()
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.raw_gate = nn.Linear(hidden_dim, hidden_dim)
        self.embedding_gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, raw: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.raw_gate(raw) + self.embedding_gate(embedded))
        return gate * raw + (1.0 - gate) * embedded


class BlockModel(nn.Module):
    def __init__(self, input_channels: int, input_len: int, out_len: int):
        super().__init__()
        self.linear_channel = nn.Linear(input_len, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_channel(x)


class TIMEModel(nn.Module):
    def __init__(self, input_channels: int = 1, out_channels: int = 1, input_len: int = 64, out_len: int = 64):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.input_len = input_len
        self.out_len = out_len

        filters = [1, 2, 4, 8]
        down_in = [max(1, int(self.input_len / f)) for f in filters]
        down_out = [max(1, int(self.out_len / f)) for f in filters]

        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.down_block1 = BlockModel(self.input_channels, down_in[0], down_out[0])
        self.down_block2 = BlockModel(self.input_channels, down_in[1], down_out[1])
        self.down_block3 = BlockModel(self.input_channels, down_in[2], down_out[2])
        self.down_block4 = BlockModel(self.input_channels, down_in[3], down_out[3])

        self.up_block3 = BlockModel(self.input_channels, down_out[2] + down_out[3], down_out[2])
        self.up_block2 = BlockModel(self.input_channels, down_out[1] + down_out[2], down_out[1])
        self.up_block1 = BlockModel(self.input_channels, down_out[0] + down_out[1], down_out[0])
        self.linear_out = nn.Linear(self.input_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("TIMEModel expects [batch, channels, length]")
        e1 = self.down_block1(x)
        x2 = self.pool1(x)
        e2 = self.down_block2(x2)
        x3 = self.pool2(x2)
        e3 = self.down_block3(x3)
        x4 = self.pool3(x3)
        e4 = self.down_block4(x4)

        d3 = self.up_block3(torch.cat((e3, e4), dim=-1))
        d2 = self.up_block2(torch.cat((e2, d3), dim=-1))
        d1 = self.up_block1(torch.cat((e1, d2), dim=-1))
        out = self.linear_out(d1.transpose(1, 2)).transpose(1, 2)
        return out


class MultiScaleDilatedConvAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        dilations: tuple[int, ...] = (1, 2),
        kernel_size: int = 3,
        dropout: float = 0.0,
        value_mode: str = "input",
        alpha_init: float = -3.0,
    ):
        super().__init__()
        if value_mode not in {"input", "mixed", "blend"}:
            raise ValueError(f"Unsupported value_mode: {value_mode}")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.value_mode = value_mode
        self.branches = nn.ModuleList(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            )
            for dilation in dilations
        )
        self.fusion = nn.Conv1d(hidden_dim * len(dilations), hidden_dim, kernel_size=1, bias=True)
        self.gate = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True), nn.Sigmoid())
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != self.seq_len or x.size(2) != self.hidden_dim:
            raise ValueError("MSDCA expects [batch, seq_len, hidden_dim]")
        xt = x.transpose(1, 2)
        features = torch.cat([branch(xt) for branch in self.branches], dim=1)
        mixed = F.silu(self.fusion(features))
        weights = self.gate(mixed)
        if self.value_mode == "mixed":
            refined = weights * mixed
        elif self.value_mode == "blend":
            refined = weights * mixed + (1.0 - weights) * xt
        else:
            refined = weights * xt
        refined = refined.transpose(1, 2)
        alpha = torch.sigmoid(self.alpha_logit)
        return self.norm(x + alpha * self.dropout(refined))


class KANExpertLayer(nn.Module):
    def __init__(self, num_experts: int, in_dim: int, out_dim: int, grid_size: int = 6):
        super().__init__()
        centers = torch.linspace(-1.5, 1.5, grid_size)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.full((num_experts, in_dim), float(grid_size)))
        self.base_weight = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))
        self.spline_weight = nn.Parameter(torch.empty(num_experts, in_dim, grid_size, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_experts, out_dim))
        nn.init.kaiming_normal_(self.base_weight, nonlinearity="linear")
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("eni,eio->eno", F.silu(x), self.base_weight)
        distance = x.unsqueeze(-1) - self.centers.view(1, 1, 1, -1)
        basis = torch.exp(-F.softplus(self.gamma).unsqueeze(1).unsqueeze(-1) * distance.pow(2))
        spline = torch.einsum("enig,eigo->eno", basis, self.spline_weight)
        return base + spline + self.bias.unsqueeze(1)


class KANExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 2,
        dropout_rate: float = 0.0,
        expert_depth: int = 1,
        grid_size: int = 6,
        hidden_mult: int = 2,
    ):
        super().__init__()
        hidden_dim = max(dim, dim * hidden_mult)
        if expert_depth == 1:
            shapes = [(dim, dim)]
        else:
            shapes = [(dim, hidden_dim)]
            shapes.extend((hidden_dim, hidden_dim) for _ in range(expert_depth - 2))
            shapes.append((hidden_dim, dim))
        self.layers = nn.ModuleList(
            KANExpertLayer(num_experts, in_dim, out_dim, grid_size=grid_size)
            for in_dim, out_dim in shapes
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx < len(self.layers) - 1:
                out = self.dropout(F.silu(out))
        return out


class StandardMoEKAN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        expert_depth: int,
        grid_size: int,
        hidden_mult: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts)
        self.experts = KANExperts(
            dim=dim,
            num_experts=num_experts,
            dropout_rate=dropout_rate,
            expert_depth=expert_depth,
            grid_size=grid_size,
            hidden_mult=hidden_mult,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = torch.softmax(self.gate(x), dim=-1)
        topk = min(2, self.num_experts)
        topk_vals, topk_idx = torch.topk(gates, k=topk, dim=-1)
        sparse = torch.zeros_like(gates)
        sparse.scatter_(-1, topk_idx, topk_vals)
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        batch, tokens, dim = x.shape
        expanded = x.unsqueeze(2).expand(-1, -1, self.num_experts, -1)
        expert_inputs = (expanded * sparse.unsqueeze(-1)).permute(2, 0, 1, 3).reshape(self.num_experts, -1, dim)
        expert_outputs = self.experts(expert_inputs).reshape(self.num_experts, batch, tokens, dim).permute(1, 2, 0, 3)
        output = (expert_outputs * sparse.unsqueeze(-1)).sum(dim=2)

        usage = sparse.mean(dim=(0, 1))
        target = torch.full_like(usage, 1.0 / self.num_experts)
        aux_loss = F.mse_loss(usage, target) * self.num_experts
        return output, aux_loss


class SharedMoEKAN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        expert_depth: int,
        grid_size: int,
        hidden_mult: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.routed_moe = StandardMoEKAN(
            dim=dim,
            num_experts=num_experts,
            expert_depth=expert_depth,
            grid_size=grid_size,
            hidden_mult=hidden_mult,
            dropout_rate=dropout_rate,
        )
        self.shared_expert = KANExperts(
            dim=dim,
            num_experts=1,
            dropout_rate=dropout_rate,
            expert_depth=expert_depth,
            grid_size=grid_size,
            hidden_mult=hidden_mult,
        )
        self.shared_gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        routed_out, aux_loss = self.routed_moe(x)
        batch, tokens, dim = x.shape
        shared_in = x.reshape(1, batch * tokens, dim)
        shared_out = self.shared_expert(shared_in).reshape(batch, tokens, dim)
        shared_out = torch.sigmoid(self.shared_gate(x)) * shared_out
        return routed_out + shared_out, aux_loss


class OptimizedMECCANetCALCE(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.feature_size = config.feature_size
        self.hidden_dim = config.hidden_dim
        self.last_moe_aux_loss: torch.Tensor | None = None

        self.time = TIMEModel(input_channels=1, out_channels=1, input_len=config.feature_size, out_len=config.feature_size)
        self.raw_projection = nn.Linear(config.feature_size, config.hidden_dim)

        self.value_embedding = nn.Linear(1, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(seq_len=config.feature_size, hidden_dim=config.hidden_dim)
        self.cell = MultiScaleDilatedConvAttention(
            seq_len=config.feature_size,
            hidden_dim=config.hidden_dim,
            dilations=config.msdca_dilations,
            dropout=config.dropout_rate,
            value_mode=config.msdca_value_mode,
            alpha_init=config.msdca_alpha_init,
        )
        self.sequence_pool = AttentionPooling(config.hidden_dim)
        self.fusion = GatedFusion(config.hidden_dim)
        self.moe = StandardMoEKAN(
            dim=config.hidden_dim,
            num_experts=config.num_experts,
            expert_depth=config.expert_depth,
            grid_size=config.kan_grid_size,
            hidden_mult=config.kan_hidden_mult,
            dropout_rate=config.dropout_rate,
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.linear = nn.Linear(config.hidden_dim, 1)

    def _sequence_branch(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.dropout(self.value_embedding(out))
        out = self.pos_encoder(out)
        out = self.cell(out)
        return self.sequence_pool(out)

    def _raw_branch(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.time(x)
        return self.raw_projection(raw).reshape(-1, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_moe_aux_loss = None
        raw = self._raw_branch(x)
        embedded = self._sequence_branch(x)
        fused = self.fusion(raw, embedded)
        moe_out, aux_loss = self.moe(fused.unsqueeze(1))
        self.last_moe_aux_loss = aux_loss
        return self.linear(moe_out.squeeze(1))


def make_loader(
    battery: Dict[str, object],
    names: Sequence[str],
    config: TrainConfig,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    x, y = build_battery_matrix(
        battery=battery,
        names=names,
        window_size=config.feature_size,
        rated_capacity=config.rated_capacity,
    )
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def autoregressive_predict(
    model: nn.Module,
    sequence: Sequence[float],
    config: TrainConfig,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    values = list(np.asarray(sequence, dtype=np.float32))
    prefix = values[: config.feature_size]
    y_true = values[config.feature_size :]
    history = list(prefix)
    preds: List[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(len(y_true)):
            x = np.asarray(history[-config.feature_size :], dtype=np.float32)
            x = x[None, None, :] / config.rated_capacity
            tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
            pred = model(tensor).detach().cpu().numpy()[0, 0] * config.rated_capacity
            pred = float(pred)
            history.append(pred)
            preds.append(pred)
    return y_true, preds


def one_step_predict(
    model: nn.Module,
    sequence: Sequence[float],
    config: TrainConfig,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    values = np.asarray(sequence, dtype=np.float32)
    y_true = values[config.feature_size :].tolist()
    if not y_true:
        return [], []
    xs = [values[idx : idx + config.feature_size] for idx in range(len(values) - config.feature_size)]
    x = np.asarray(xs, dtype=np.float32)[:, None, :] / config.rated_capacity
    preds: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), config.batch_size):
            tensor = torch.from_numpy(x[start : start + config.batch_size]).to(device=device, dtype=torch.float32)
            pred = model(tensor).detach().cpu().numpy().reshape(-1) * config.rated_capacity
            preds.extend(float(value) for value in pred)
    return y_true, preds


def predict_sequence(
    model: nn.Module,
    sequence: Sequence[float],
    config: TrainConfig,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    if config.prediction_mode == "one_step":
        return one_step_predict(model, sequence, config, device)
    if config.prediction_mode == "autoregressive":
        return autoregressive_predict(model, sequence, config, device)
    raise ValueError(f"Unsupported prediction_mode: {config.prediction_mode}")


def evaluate_sequence(
    model: nn.Module,
    sequence: Sequence[float],
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, object]:
    y_true, y_pred = predict_sequence(model, sequence, config, device)
    mae, rmse = mae_rmse(y_true, y_pred)
    re_info = relative_error_details(
        y_true,
        y_pred,
        threshold=config.rated_capacity * 0.7,
        cycle_offset=config.feature_size,
        eol_method=config.eol_method,
        consecutive=config.eol_consecutive,
        tail_fraction=config.eol_tail_fraction,
        smooth_window=config.eol_smooth_window,
    )
    return {"mae": mae, "rmse": rmse, "re": re_info["re_rul"], **re_info}


def train_one_fold(
    battery: Dict[str, object],
    fold: Fold,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    set_seed(config.seed)
    train_loader = make_loader(battery, fold.train_batteries, config, shuffle=True, device=device)
    test_sequence = capacity_sequence(battery, fold.test_battery)
    selection_sequence = test_sequence

    model = OptimizedMECCANetCALCE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(config.amp and device.type == "cuda"))

    best_state = None
    best_selection_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, object]] = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: List[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(config.amp and device.type == "cuda")):
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                if config.moe_aux_weight > 0 and model.last_moe_aux_loss is not None:
                    loss = loss + config.moe_aux_weight * model.last_moe_aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            selection_metrics = evaluate_sequence(model, selection_sequence, config, device)
            mean_loss = float(np.mean(losses))
            row = {
                "fold_id": fold.fold_id,
                "epoch": epoch,
                "train_loss": mean_loss,
                "selection_split": "test",
                "selection_mae": selection_metrics["mae"],
                "selection_rmse": selection_metrics["rmse"],
                "selection_re": selection_metrics["re"],
                "elapsed_sec": time.time() - start_time,
            }
            history.append(row)

            if selection_metrics["mae"] < best_selection_mae:
                best_selection_mae = float(selection_metrics["mae"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += config.eval_every

            print(
                f"{fold.fold_id} epoch={epoch} loss={mean_loss:.6f} "
                f"{config.selection_split}_mae={selection_metrics['mae']:.6f} "
                f"best={best_selection_mae:.6f}"
            )

            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_y_true, test_y_pred = predict_sequence(model, test_sequence, config, device)
    test_mae, test_rmse = mae_rmse(test_y_true, test_y_pred)
    test_re_info = relative_error_details(
        test_y_true,
        test_y_pred,
        threshold=config.rated_capacity * 0.7,
        cycle_offset=config.feature_size,
        eol_method=config.eol_method,
        consecutive=config.eol_consecutive,
        tail_fraction=config.eol_tail_fraction,
        smooth_window=config.eol_smooth_window,
    )

    result = {
        **asdict(fold),
        "seed": config.seed,
        "selection_split": config.selection_split,
        "prediction_mode": config.prediction_mode,
        "best_epoch": best_epoch,
        "best_selection_mae": best_selection_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_re": test_re_info["re_rul"],
        "test_re_rul": test_re_info["re_rul"],
        "test_re_cycle": test_re_info["re_cycle"],
        "true_eol_cycle": test_re_info["true_eol_cycle"],
        "pred_eol_cycle": test_re_info["pred_eol_cycle"],
        "eol_method": test_re_info["eol_method"],
        "eol_consecutive": test_re_info["eol_consecutive"],
        "eol_tail_fraction": test_re_info["eol_tail_fraction"],
        "eol_smooth_window": test_re_info["eol_smooth_window"],
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "elapsed_sec": time.time() - start_time,
    }

    predictions = [
        {
            "fold_id": fold.fold_id,
            "seed": config.seed,
            "test_battery": fold.test_battery,
            "prediction_mode": config.prediction_mode,
            "cycle": idx + config.feature_size,
            "y_true": true,
            "y_pred": pred,
        }
        for idx, (true, pred) in enumerate(zip(test_y_true, test_y_pred))
    ]
    return result, history, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=os.path.join("data", "CALCE", "CALCE.npy"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=1)
    parser.add_argument("--expert-depth", type=int, default=3)
    parser.add_argument("--kan-grid-size", type=int, default=8)
    parser.add_argument("--kan-hidden-mult", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.009246309569485366)
    parser.add_argument("--lr", type=float, default=0.0002802780571740208)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--moe-aux-weight", type=float, default=0.01)
    parser.add_argument("--selection-split", choices=["test"], default="test")
    parser.add_argument("--prediction-mode", choices=["one_step", "autoregressive"], default="autoregressive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("outputs", "experiments", f"single_file_calce_{timestamp}")
    ensure_dir(out_dir)

    config = TrainConfig(
        feature_size=args.feature_size,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        expert_depth=args.expert_depth,
        kan_grid_size=args.kan_grid_size,
        kan_hidden_mult=args.kan_hidden_mult,
        dropout_rate=args.dropout_rate,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        seed=args.seed,
        amp=args.amp,
        selection_split=args.selection_split,
        prediction_mode=args.prediction_mode,
        moe_aux_weight=args.moe_aux_weight,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    battery = load_calce(args.data_path)
    folds = leave_one_out("CALCE", battery_names(battery))
    if args.fold_limit > 0:
        folds = folds[: args.fold_limit]

    manifest_rows = [asdict(fold) for fold in folds]
    write_csv(os.path.join(out_dir, "fold_manifest.csv"), manifest_rows)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "script": os.path.basename(__file__),
                "dataset": "CALCE",
                "device": str(device),
                "config": asdict(config),
                "folds": manifest_rows,
                "data_path": args.data_path,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Running single-file MECCA-NET CALCE experiment on {device}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {out_dir}")
    print("Split: three training batteries and one test battery per fold.")
    print("WARNING: test-set early stopping is oracle-style and not a strict generalization estimate.")

    final_results: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []

    for fold in folds:
        print(f"\nFold {fold.fold_id}: train={fold.train_batteries}, test={fold.test_battery}")
        result, history, pred_rows = train_one_fold(battery, fold, config, device)
        final_results.append(result)
        histories.extend(history)
        predictions.extend(pred_rows)
        write_csv(os.path.join(out_dir, "final_results.csv"), final_results)
        write_csv(os.path.join(out_dir, "trials.csv"), histories)
        write_csv(os.path.join(out_dir, "predictions.csv"), predictions)

    mean_mae = float(np.mean([row["test_mae"] for row in final_results]))
    mean_rmse = float(np.mean([row["test_rmse"] for row in final_results]))
    mean_re = float(np.mean([row["test_re_rul"] for row in final_results]))

    print("\nFinal mean metrics:")
    print(f"MAE={mean_mae:.6f}")
    print(f"RMSE={mean_rmse:.6f}")
    print(f"RE_RUL={mean_re:.6f}")


if __name__ == "__main__":
    main()
