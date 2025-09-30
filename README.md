# MECCA-NET
This is the code for our paper MECCA-Net: A Remaining Useful Life Prediction Model for Lithium-ion Batteries Based on a Multi-Layer Perceptron Expert Network and Temporal Feature Fusion.
We will open-source the model code following the paper’s acceptance.
### System Requirements
- Python 3.7+
- CUDA 11.8 (for GPU acceleration)

### Core Dependencies

#### Deep Learning Framework
- **torch** (2.4.1+cu118) - PyTorch deep learning framework
- **torchvision** (0.19.1+cu118) - Computer vision toolkit
- **torchaudio** (2.4.1+cu118) - Audio processing toolkit


#### Citation
If you find this work useful in your research, please consider citing:
@article{YI2025238371,
    title = {A lithium-ion battery remaining useful life prediction model based on multilayer perceptron expert networks and temporal feature composition},
    journal = {Journal of Power Sources},
    volume = {659},
    pages = {238371},
    year = {2025},
    issn = {0378-7753},
    doi = {https://doi.org/10.1016/j.jpowsour.2025.238371},
    url = {https://www.sciencedirect.com/science/article/pii/S0378775325022074},
    author = {Xuan Yi and Jianmao Xiao and Gang Lei and Xin Hu and Zhiyong Feng},
    keywords = {Remaining Useful Life Prediction (RUL prediction), Lithium-ion battery, Multilayer perceptron (MLP) mixture of experts (moE), Temporal pattern composer},
    abstract = {Unscheduled downtime caused by lithium-ion battery failures in electric vehicles and energy storage systems poses a significant challenge for accurately predicting remaining useful life (RUL). Existing methods, however, typically depend on high-quality and comprehensive performance data, limiting their applicability in complex real-world scenarios. To overcome this limitation, we propose MECCA-Net, a novel neural network framework whose core component is a self-designed Temporal Pattern Composer (TPC) that adaptively captures multi-level and cross-scale temporal degradation patterns from limited discharge capacity data. MECCA-Net further integrates multi-layer denoising autoencoders, multi-head self-attention mechanisms, and a mixture-of-experts structure to enhance its generalization capability and robustness. The experimental results demonstrate that MECCA-Net reduces the Relative Error (RE) by approximately 40% on several authoritative lithium-ion battery lifespan datasets compared to the latest state-of-the-art models. Furthermore, this approach exhibits superior prediction accuracy and stability performance over mainstream time-series modeling techniques, showcasing its efficiency and practical value in lithium-ion battery health management and predictive maintenance. The source code and datasets are available at https://github.com/keepawakeyi/MECCA-NET.}
}
