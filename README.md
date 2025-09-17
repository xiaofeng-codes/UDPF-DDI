# A Unified Differential Denoising Learning Framework with a Pre-trained Model and Fuzzy Graph Networks for Drug-Drug Interaction Prediction (UDPF-DDI)

This repository contains the official PyTorch implementation for the paper, **"A Unified Differential Denoising Learning Framework with a Pre-trained Model and Fuzzy Graph Networks for Drug-Drug Interaction Prediction"**.

### Abstract

Combination therapy is common in clinical practice, but drug-drug interactions (DDIs) can pose significant safety risks. Accurate DDI prediction is vital. Our proposed **UDPF-DDI** framework addresses limitations in existing methods by handling data uncertainty and attention noise. It integrates multi-source features from a large-scale pre-trained model (Uni-Mol2) and RDKit with a parallel three-channel encoder. This encoder uses a fuzzy graph convolution neural network (FGCNN) to capture fuzzy features and a differential Transformer to denoise high-dimensional atomic features. The framework then fuses these features to predict DDI scores, showing significant performance improvements over state-of-the-art models across various datasets.

---

### Framework Diagram

这是一个展示 UDPF-DDI 框架结构的示意图：

!https://github.com/xiaofeng-codes/UDPF-DDI/framwork.png

---
### Getting Started

#### Prerequisites

To run the code, you'll need the following Python environment setup:

- **Python**: 3.8+
- **PyTorch**: 1.12
- **Numpy**: 2.5
- Other dependencies can be found in `requirements.txt`.


---

### Running the Code

The repository is structured by task. Navigate to the specific task folder to run the model.

1.  **Navigate to the task directory**:
    ```bash
    cd <task_name>
    ```
    (e.g., `cd binary` or `cd multi_rel`)

2.  **Run the training script**:
    ```bash
    python train.py
    ```

For additional arguments and configuration options, please refer to the `train.py` script and the configuration files within each task folder.
