# MCANet‑Tiny: Edge‑Deployable Crack Detection via Synergistic GAN‑Based Synthesis and Lightweight Classification

This repository contains the official implementation of **MCANet‑Tiny**, an ultra‑lightweight pavement crack classifier designed for real‑time inference on edge devices. The model integrates Coordinate Attention (CA) with a “Tiny” pruning regime (Depth Adaptation + Global Channel Scaling) on the MobileNetV3‑Small backbone, achieving extreme efficiency (0.12 M parameters, 0.02 GFLOPs) while maintaining high classification accuracy. It is part of a synergistic framework that also includes RoadFreq‑GAN for high‑fidelity data synthesis.

## Table of Contents
- [1. Description](#1-description)
- [2. Project Structure](#2-project-structure)
- [3. Requirements](#3-requirements)
- [4. Data Preparation](#4-data-preparation)
- [5. Training](#5-training)
  - [5.1 Basic Command](#51-basic-command)
  - [5.2 Command Line Arguments](#52-command-line-arguments)
  - [5.3 Monitoring Training](#53-monitoring-training)
- [6. Reproducing Paper Results](#6-reproducing-paper-results)
  - [6.1 Ablation Study Configurations](#61-ablation-study-configurations)
  - [6.2 Training on Different Datasets](#62-training-on-different-datasets)
- [7. Evaluation & Quantization](#7-evaluation--quantization)
  - [7.1 Evaluation on Test Set](#71-evaluation-on-test-set)
  - [7.2 INT8 Quantization for Edge Deployment](#72-int8-quantization-for-edge-deployment)
  - [7.3 Computing Model Statistics (Parameters / FLOPs)](#73-computing-model-statistics-parameters--flops)
- [8. Model Architecture](#8-model-architecture)
  - [8.1 Backbone Modifications](#81-backbone-modifications)
  - [8.2 Coordinate Attention (CA)](#82-coordinate-attention-ca)
  - [8.3 Training Optimizations](#83-training-optimizations)
- [9. Results & Metrics](#9-results--metrics)
  - [9.1 Performance on Cross‑Dataset Test Benchmark (CDTS)](#91-performance-on-crossdataset-test-benchmark-cdts)
  - [9.2 Real‑Time Deployment on Edge Devices (INT8)](#92-realtime-deployment-on-edge-devices-int8)
  - [9.3 Ablation Study](#93-ablation-study)
  - [9.4 Classification Report (on CDTS)](#94-classification-report-on-cdts)
- [10. License](#10-license)
- [11. Citation](#11-citation)
- [12. Contact](#12-contact)

## 1. Description

MCANet‑Tiny is a lightweight convolutional neural network for pavement crack classification (5 categories: transverse, longitudinal, alligator, pothole, normal). It is derived from MobileNetV3‑Small through a hardware‑aware co‑design strategy that includes:

- **Structural Pruning**: Depth Adaptation (retains only 6 essential inverted residual blocks) + Global Channel Scaling (width multiplier α=0.4) to reduce parameters by 92.1% and GFLOPs by 83.3% relative to the baseline.
- **Coordinate Attention (CA)**: Compensates for spatial detail loss from pruning, improving accuracy from 77.40% to 80.90% with even fewer parameters.
- **Advanced Training Recipe**: AdamW optimizer + Cosine Annealing scheduler + Linear Warmup to reach 93.40% accuracy on a cross‑dataset test benchmark.
- **INT8 Quantization**: Enables real‑time inference (166 FPS on Huawei P50 Pro) with minimal accuracy drop.

The model is designed to work with synthetic data from RoadFreq‑GAN, but can also be trained on real or augmented datasets.

## 2. Project Structure

```
MCANet-Tiny/
├── dataset/                  # Place datasets here
│   ├── A-SET/                # Original 200 images per class
│   │   ├── train/
│   │   └── val/
│   ├── T-SET/                # Basic augmentations (1000 per class)
│   ├── S-SET/                # Synthetic images from RoadFreq-GAN (1000 per class)
│   └── G-SET/                # T-SET + S-SET (2000 per class)
├── mcanet/
│   ├── __init__.py
│   ├── model.py              # MCANetTiny definition (CA, MobileNetV3 blocks)
│   ├── data_utils.py         # DataLoader, dataset splits, augmentations
│   ├── train.py              # Training loop with AdamW, CosineAnnealing
│   ├── evaluate.py           # Accuracy, precision, recall, F1, confusion matrix
│   ├── quantize.py           # INT8 quantization via torch.quantization
│   └── utils.py              # Helper functions (logging, metrics)
├── main.py                   # Single entry point (train/eval/quantize)
├── requirements.txt
├── README.md
└── results/                  # Created automatically during training
    └── {experiment_name}/
        ├── model_best.pth
        ├── model_final.pth
        ├── training_log.csv
        ├── confusion_matrix.png
        ├── learning_curves.png
        └── quantized_model.ptl
```

## 3. Requirements

Create a virtual environment (Python 3.8+) and install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
torch==2.12.0
torchvision==0.27.0
tqdm==4.67.3
ultralytics-thop==2.0.19
matplotlib==3.10.9
seaborn==0.13.2
scikit-learn==1.8.0
pandas==3.0.3
numpy==2.4.6
```

### Installation

Clone the repository:

```bash
git clone https://github.com/Winckedy/MCANet-Tiny.git
cd MCANet-Tiny
```

Place your dataset (see Data Preparation) and run training.

## 4. Data Preparation

The training script expects a standard `ImageFolder` structure:

```
/path/to/dataset/
    train/
        0transverse/      # images of transverse cracks
        1longitudinal/
        2alligator/
        3pothole/
        4normal/
    val/                 # same subfolders as train
    test/                # (optional) for final evaluation
```

If you want to replicate the paper’s settings:

- **A‑set**: original 200 images per class (not used directly for training).
- **T‑set**: A‑set expanded by basic augmentations (flip, rotation, color jitter) → 1000 images per class.
- **S‑set**: synthetic images from RoadFreq‑GAN → 1000 images per class.
- **G‑set**: T‑set + S‑set (2000 images per class) – the primary training configuration.

All images are resized to 224×224 (standard for MobileNetV3) and normalized using ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

## 5. Training

### 5.1 Basic Command

```bash
python main.py --mode train --data_root /path/to/dataset --name my_experiment --epochs 50 --batch_size 64
```

### 5.2 Command Line Arguments

| Argument           | Default     | Description                                                       |
|--------------------|-------------|-------------------------------------------------------------------|
| `--mode`           | `train`     | `train`, `eval`, or `quantize`                                    |
| `--data_root`      | `./dataset` | Path to dataset root (contains `train/`, `val/` subfolders)       |
| `--name`           | `test`      | Experiment name – creates `results/{name}/`                       |
| `--epochs`         | `50`        | Number of training epochs                                         |
| `--batch_size`     | `64`        | Batch size per GPU                                                |
| `--lr`             | `0.001`     | Base learning rate                                                |
| `--optimizer`      | `adamw`     | `adam` or `adamw`                                                 |
| `--scheduler`      | `cosine`    | `cosine` or `step`                                                |
| `--warmup_epochs`  | `3`         | Warmup epochs for cosine scheduler                                |
| `--weight_decay`   | `1e-4`      | Weight decay                                                      |
| `--dropout`        | `0.5`       | Dropout rate after global pooling                                 |
| `--no_cuda`        | `False`     | Disable CUDA                                                      |
| `--ckpt`           | `None`      | Resume training from a checkpoint                                 |

### 5.3 Monitoring Training

During training, the following are saved in `results/{name}/`:

- `model_best.pth` – checkpoint with highest validation accuracy.
- `model_final.pth` – final model after last epoch.
- `training_log.csv` – per‑epoch loss, accuracy, precision, recall, F1.
- `learning_curves.png` – plots of training/validation loss and accuracy.
- `confusion_matrix.png` – on validation set (updated every epoch).

Console output every epoch:
```
Epoch 1/50: train_loss=1.234, train_acc=45.67, val_loss=1.098, val_acc=52.34, best_acc=52.34
```
## 6. Reproducing Paper Results

This section provides ready‑to‑use commands to reproduce the ablation studies (Table 7 in the paper) and train on different dataset configurations (A‑set, T‑set, S‑set, G‑set).

### 6.1 Ablation Study Configurations

The following commands replicate each row of the ablation study using the **G‑set** (combined real augmented + synthetic data). Ensure your `G-SET` folder is prepared under `--data_root` with `train/` and `val/` subdirectories.

| Configuration                          | Command                                                                                                                              |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Pruned MobileNetV3‑Small (baseline)    | `python main.py --mode train --data_root ./dataset/G-SET --name baseline --attn none --optimizer adam --scheduler step --epochs 50`  |
| + CA                                   | `python main.py --mode train --data_root ./dataset/G-SET --name with_CA --attn ca --optimizer adam --scheduler step --epochs 50`     |
| + AdamW                                | `python main.py --mode train --data_root ./dataset/G-SET --name with_AdamW --attn ca --optimizer adamw --scheduler step --epochs 50` |
| + Cosine Annealing (Final MCANet‑Tiny) | `python main.py --mode train --data_root ./dataset/G-SET --name final --attn ca --optimizer adamw --scheduler cosine --epochs 50`    |

**Note**: The `--attn` argument is a custom flag you can add to `main.py` to control CA insertion. If your implementation doesn’t have it, you can manually edit `model.py` or use separate branches. Alternatively, you can use the `--model_variant` argument as shown below.

### 6.2 Training on Different Datasets

To evaluate the impact of different data sources (A‑set, T‑set, S‑set, G‑set), run the following commands after preparing each dataset under `./dataset/` with the required folder structure.

| Dataset | Description                                     | Command                                                                                                                                  |
|---------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| A‑set   | Original 200 images per class (no augmentation) | `python main.py --mode train --data_root ./dataset/A-SET --name experiment_A --attn ca --optimizer adamw --scheduler cosine --epochs 50` |
| T‑set   | Basic augmentations (flip, rotation, color)     | `python main.py --mode train --data_root ./dataset/T-SET --name experiment_T --attn ca --optimizer adamw --scheduler cosine --epochs 50` |
| S‑set   | Synthetic images from RoadFreq‑GAN only         | `python main.py --mode train --data_root ./dataset/S-SET --name experiment_S --attn ca --optimizer adamw --scheduler cosine --epochs 50` |
| G‑set   | T‑set + S‑set (recommended for final model)     | `python main.py --mode train --data_root ./dataset/G-SET --name experiment_G --attn ca --optimizer adamw --scheduler cosine --epochs 50` |

All commands use the final MCANet‑Tiny configuration (CA + AdamW + Cosine Annealing). You can modify `--epochs`, `--batch_size`, etc. as needed.

**Expected results** (from the paper):

- A‑set: ~65% accuracy
- T‑set: ~83% accuracy
- S‑set: ~82% accuracy
- G‑set: ~93% accuracy

See Section 9 for full metrics and comparisons.
## 7. Evaluation & Quantization

### 7.1 Evaluation on Test Set

```bash
python main.py --mode eval --data_root /path/to/dataset --ckpt results/my_experiment/model_best.pth
```

This will output:

- Accuracy, precision, recall, F1‑score per class and macro average.
- Confusion matrix (saved as `results/{name}/test_confusion.png`).
- Classification report (saved as `results/{name}/test_report.txt`).

### 7.2 INT8 Quantization for Edge Deployment

```bash
python main.py --mode quantize --ckpt results/my_experiment/model_best.pth --output quantized_model.ptl
```

The script applies:

- Post‑training static quantization (torch.quantization).
- Calibration using a subset of the training set (100 batches).
- Saves the quantized model in TorchScript format (`.ptl`).

To evaluate the quantized model:

```bash
python evaluate.py --quantized --ckpt quantized_model.ptl --data_root /path/to/dataset
```

### 7.3 Computing Model Statistics (Parameters / FLOPs)

```bash
python -m thop --model mcanet.model.MCANetTiny --input_size 224 3
```

Or run included script:

```bash
python profile_models.py
```

Output example:
```
MCANet-Tiny: params=0.12M, GFLOPs=0.02
```

## 8. Model Architecture

### 8.1 Backbone Modifications

MCANet‑Tiny starts from MobileNetV3‑Small and applies:

- **Depth Adaptation**: Only the first 6 inverted residual blocks (out of 12) are kept – shorter inference path.
- **Global Channel Scaling**: width multiplier α=0.4 reduces all convolutional channels (e.g., 16→6, 24→10, …).
- **Classification Head**: Global Average Pooling → Dropout(0.5) → Linear(512, 5).

### 8.2 Coordinate Attention (CA)

CA module is inserted after each inverted residual block. It encodes spatial attention in two separate directions (height and width) without global pooling, preserving precise positional information. The module adds only ~0.01 M extra parameters.

**Mathematical formulation:**  
Given input feature map `X` of size `C×H×W`, CA computes:

- `z_h(h) = 1/W Σ_{i} X(h,i)`   (vertical direction)
- `z_w(w) = 1/H Σ_{j} X(j,w)`   (horizontal direction)

Then applies shared 1×1 convolution, batch norm, non‑linearity, splits into two tensors, followed by two dilated 3×3 convolutions, sigmoid activation, and element‑wise multiplication with the original feature map.

### 8.3 Training Optimizations

- **AdamW** (decoupled weight decay) improves regularization.
- **Cosine Annealing with Linear Warmup** (3 epochs) stabilizes early training.
- **Cross‑Entropy Loss** with label smoothing (default: 0.1).

## 9. Results & Metrics

### 9.1 Performance on Cross‑Dataset Test Benchmark (CDTS)

| Model                        | Accuracy (%) | Params (M) | GFLOPs | Size (MB) | FPS (Desktop GPU) |
|------------------------------|--------------|------------|--------|-----------|-------------------|
| MobileNetV3‑Small (baseline) | 88.30        | 1.52       | 0.12   | 6.00      | 91.15             |
| MCANet‑Tiny (pruned + CA)    | 93.40        | 0.12       | 0.02   | 0.54      | 118.31            |

### 9.2 Real‑Time Deployment on Edge Devices (INT8)

| Platform              | Average FPS | Accuracy (%) |
|-----------------------|-------------|--------------|
| Huawei P50 Pro (CPU)  | 166.67      | 94.26        |
| Radxa Cubie A7A       | 142.86      | 94.26        |
| Orange Pi 3B          | 78.25       | 93.80        |

All tests run on live video streams (1080p, cropped to 224×224).

### 9.3 Ablation Study

| Configuration                       | Accuracy (%) | Params (M) |
|-------------------------------------|--------------|------------|
| Pruned MobileNetV3‑Small (baseline) | 77.40        | 0.14       |
| + CA                                | 80.90        | 0.12       |
| + AdamW                             | 91.90        | 0.12       |
| + Cosine Annealing (final)          | 93.40        | 0.12       |

### 9.4 Classification Report (on CDTS)

| Class            | Precision | Recall | F1‑score |
|------------------|-----------|--------|----------|
| Transverse       | 0.94      | 0.93   | 0.93     |
| Longitudinal     | 0.93      | 0.93   | 0.93     |
| Alligator        | 0.91      | 0.91   | 0.91     |
| Pothole          | 0.92      | 0.92   | 0.92     |
| Normal           | 0.98      | 0.98   | 0.98     |

Macro average: **93.4% F1**.

## 10. License

This project is licensed under the **GNU General Public License v3.0** – see the [LICENSE](LICENSE) file for details.

## 11. Citation

If you use MCANet‑Tiny in your research, please cite:

```bibtex
@software{MCANetTiny2026,
  author = {Jia Chen and Shuyang Chen and Zhenliang Jiang and Yun Que and Yining Chen and Jingwen Wang},
  title = {MCANet-Tiny: Edge-Deployable Crack Detection via Synergistic GAN-Based Synthesis and Lightweight Classification},
  year = {2026},
  url = {https://github.com/Winckedy/MCANet-Tiny},
  note = {Lightweight pavement crack classifier with Coordinate Attention and structural pruning}
}
```

## 12. Contact

For questions or issues, please open an issue on GitHub or email:  
2500516015@fzu.edu.cn

Happy detecting! 🛣️🔍