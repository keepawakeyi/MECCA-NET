"""Standalone iTransformer / ModernTCN training script for NASA RUL.

This script matches the current release-style setup:

- NASA dataset
- battery-level leave-one-out
- three training batteries and one test battery per fold
- autoregressive rollout by default
- test-set early stopping for exploratory comparison

Run examples:

python baseline_nasa_autoregressive.py --model itransformer --data-path data/NASA/NASA.npy
python baseline_nasa_autoregressive.py --model moderntcn --data-path data/NASA/NASA.npy
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


RATED_CAPACITY_NASA = 2.0


@dataclass(frozen=True)
class Fold:
    dataset: str
    fold_id: str
    train_batteries: List[str]
    test_battery: str


@dataclass
class TrainConfig:
    model: str = "itransformer"
    feature_size: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    nhead: int = 4
    dropout_rate: float = 0.05
    lr: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 200
    patience: int = 40
    eval_every: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = False
    prediction_mode: str = "autoregressive"
    rated_capacity: float = RATED_CAPACITY_NASA
    eol_method: str = "sustained"
    eol_consecutive: int = 3
    eol_tail_fraction: float = 0.8
    eol_smooth_window: int = 1
    tcn_kernel_size: int = 5


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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_nasa(npy_path: str) -> Dict[str, list]:
    return np.load(npy_path, allow_pickle=True).item()


def battery_names(battery: Dict[str, list]) -> List[str]:
    preferred = ["B0005", "B0006", "B0007", "B0018"]
    return [name for name in preferred if name in battery]


def capacity_sequence(battery: Dict[str, list], name: str) -> np.ndarray:
    value = battery[name]
    if isinstance(value, dict) and "capacity" in value:
        capacity = value["capacity"]
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
    battery: Dict[str, list],
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
    eol_method: str = "sustained",
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
        "true_eol_cycle": int(true_cycle),
        "pred_eol_cycle": int(pred_cycle),
        "eol_method": eol_method,
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


class ITransformerRUL(nn.Module):
    """Compact Transformer-style baseline for univariate RUL windows."""

    def __init__(self, feature_size: int, hidden_dim: int, num_layers: int, nhead: int, dropout_rate: float):
        super().__init__()
        self.value_embedding = nn.Linear(1, hidden_dim)
        self.pos_encoder = PositionalEncoding(feature_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.value_embedding(out)
        out = self.pos_encoder(out)
        out = self.encoder(out)
        out = self.norm(out)
        weights = torch.softmax(self.score(out), dim=1)
        pooled = torch.sum(weights * out, dim=1)
        return self.head(self.dropout(pooled))


class ModernTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
            bias=False,
        )
        self.pointwise1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.pointwise2 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise(x)
        out = self.norm(out)
        out = F.gelu(self.pointwise1(out))
        out = self.dropout(out)
        out = self.pointwise2(out)
        return residual + out


class ModernTCNRUL(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, num_layers: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.input_projection = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            ModernTCNBlock(hidden_dim, kernel_size=kernel_size, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )
        self.feature_size = feature_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_projection(x)
        for block in self.blocks:
            out = block(out)
        pooled = self.pool(out).squeeze(-1)
        return self.head(pooled)


def build_model(config: TrainConfig) -> nn.Module:
    if config.model == "itransformer":
        return ITransformerRUL(
            feature_size=config.feature_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            nhead=config.nhead,
            dropout_rate=config.dropout_rate,
        )
    if config.model == "moderntcn":
        return ModernTCNRUL(
            feature_size=config.feature_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            kernel_size=config.tcn_kernel_size,
            dropout_rate=config.dropout_rate,
        )
    raise ValueError(f"Unsupported model: {config.model}")


def make_loader(
    battery: Dict[str, list],
    names: Sequence[str],
    config: TrainConfig,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    x, y = build_battery_matrix(battery, names, config.feature_size, config.rated_capacity)
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
    y_true = values[config.feature_size :]
    history = list(values[: config.feature_size])
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
    return autoregressive_predict(model, sequence, config, device)


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
    battery: Dict[str, list],
    fold: Fold,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    set_seed(config.seed)
    train_loader = make_loader(battery, fold.train_batteries, config, shuffle=True, device=device)
    test_sequence = capacity_sequence(battery, fold.test_battery)

    model = build_model(config).to(device)
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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            selection_metrics = evaluate_sequence(model, test_sequence, config, device)
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
                f"test_mae={selection_metrics['mae']:.6f} best={best_selection_mae:.6f}"
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
        "model": config.model,
        "seed": config.seed,
        "prediction_mode": config.prediction_mode,
        "best_epoch": best_epoch,
        "best_selection_mae": best_selection_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_re_rul": test_re_info["re_rul"],
        "test_re_cycle": test_re_info["re_cycle"],
        "true_eol_cycle": test_re_info["true_eol_cycle"],
        "pred_eol_cycle": test_re_info["pred_eol_cycle"],
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "elapsed_sec": time.time() - start_time,
    }
    predictions = [
        {
            "fold_id": fold.fold_id,
            "model": config.model,
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
    parser.add_argument("--model", choices=["itransformer", "moderntcn"], required=True)
    parser.add_argument("--data-path", default=os.path.join("data", "NASA", "NASA.npy"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--feature-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prediction-mode", choices=["one_step", "autoregressive"], default="autoregressive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("outputs", "experiments", f"{args.model}_nasa_3train_{timestamp}")
    ensure_dir(out_dir)

    config = TrainConfig(
        model=args.model,
        feature_size=args.feature_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        tcn_kernel_size=args.tcn_kernel_size,
        dropout_rate=args.dropout_rate,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        seed=args.seed,
        amp=args.amp,
        prediction_mode=args.prediction_mode,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    battery = load_nasa(args.data_path)
    folds = leave_one_out("NASA", battery_names(battery))
    if args.fold_limit > 0:
        folds = folds[: args.fold_limit]

    manifest_rows = [asdict(fold) for fold in folds]
    write_csv(os.path.join(out_dir, "fold_manifest.csv"), manifest_rows)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "script": os.path.basename(__file__),
                "dataset": "NASA",
                "device": str(device),
                "config": asdict(config),
                "folds": manifest_rows,
                "data_path": args.data_path,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Running {args.model} NASA experiment on {device}")
    print("Split: three training batteries and one test battery per fold.")
    print("WARNING: test-set early stopping is oracle-style and not a strict generalization estimate.")
    print(f"Output directory: {out_dir}")

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
