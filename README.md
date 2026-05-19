# MCANet-Tiny with Coordinate Attention

A lightweight classification model for pavement crack detection, featuring selectable attention mechanisms and a complete training-to-deployment pipeline. This repository supports the ablation studies in the paper *"Edge-Deployable Crack Detection: Synergizing GAN-Based Synthesis with MCANet-Tiny for Real-Time Inspection"*.

## Project Structure
```
MCANet-Tiny/
├── dataset/ 
├── mcanet/               # Core package
│   ├── __init__.py
│   ├── model.py          # Model definition (CoordAtt, SELayer, MCANetTiny)
│   ├── data_utils.py     # Data loaders and model creation
│   ├── train.py          # Training loop (configurable optimizer/scheduler)
│   ├── evaluate.py       # Evaluation and model statistics
│   ├── visualize.py      # Plotting and report generation
│   └── quantize.py       # INT8 quantization and PTL export
├── main.py               # Main entry point with ablation arguments
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

1. **Clone the repository** and install dependencies (Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare datasets**: Place datasets under `./dataset/`. Each folder (e.g., `A-SET`, `G-SET`) should contain `train`, `validation`, and `test` subdirectories with class folders (standard `ImageFolder` format).

## Usage

### Running a specific ablation configuration
Use command-line arguments to reproduce each row of Table 7 in the paper:

| Table 7 Configuration                             | Command                                                                                     |
|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| **Pruned MobileNetV3-Small (SE + Adam + StepLR)** | `python main.py --datasets GSet --attn se --optimizer adam --scheduler step --epochs 50`    |
| **+ CA** (CA + Adam + StepLR)                     | `python main.py --datasets GSet --attn ca --optimizer adam --scheduler step --epochs 50`    |
| **+ AdamW** (CA + AdamW + StepLR)                 | `python main.py --datasets GSet --attn ca --optimizer adamw --scheduler step --epochs 50`   |
| **+ Cosine Annealing (Final MCANet-Tiny)**        | `python main.py --datasets GSet --attn ca --optimizer adamw --scheduler cosine --epochs 50` |

**Note**: The ablation study uses the **G-set** (combined real augmented + synthetic data). Ensure the `G-SET` folder is properly prepared with the merged training images.

### General options
```
--datasets      Dataset(s) to process: all (default) or comma-separated (ASet,TSet,SSet,GSet)
--attn          Attention type: ca (Coordinate Attention), se (Squeeze-and-Excitation), none
--optimizer     Optimizer: adam, adamw
--scheduler     LR scheduler: cosine, step
--epochs        Number of training epochs (default: 50)
--warmup        Warmup epochs for cosine scheduler (default: 3)
--lr            Base learning rate (default: 0.001)
--batch_size    Training batch size (default: 64)
--no_quantize   Skip INT8 quantization and PTL export
```

### Example: Run the final model on all datasets
```bash
python main.py --attn ca --optimizer adamw --scheduler cosine --epochs 50
```

## Output
- Trained model weights (`.pth`)
- Training curves and confusion matrix
- CSV logs of losses and accuracies
- FP32 and INT8 mobile-optimized models (`.ptl`)
- Full performance report (`.txt`)

## Citation
If you use this code, please cite the corresponding paper.