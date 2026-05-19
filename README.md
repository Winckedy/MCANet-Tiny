# MCANet-Tiny with Coordinate Attention

A lightweight classification model for pavement crack detection, featuring selectable attention mechanisms and a complete training-to-deployment pipeline. This repository supports the ablation studies in the paper *"Edge-Deployable Crack Detection: Synergizing GAN-Based Synthesis with MCANet-Tiny for Real-Time Inspection"*.

## Table of Contents

- [1. Description](#1-description)
- [2. Project Structure](#2-project-structure)
- [3. Setup](#3-setup)
- [4. Data Preparation](#4-data-preparation)
- [5. Usage](#5-usage)
- [6. Model Architecture](#6-model-architecture)
- [7. Output](#7-output)
- [8. License](#8-license)
- [9. Citation](#9-citation)
- [10. Contact](#10-contact)

## 1. Project Structure
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

## License
This project is licensed under the GNU General Public License v3.0 – see the [LICENSE](LICENSE) file for details.

## Citation
If you use MCANet-Tiny in your research, please cite:

```bibtex
@software{MCANet-Tiny2026,
  author = {Jia Chen, Shuyang Chen, Yun Que, Yining Chen, Jingwen Wang},
  title = {MCANet-Tiny: Edge-Deployable Crack Detection via Synergistic GAN-Based Synthesis and Lightweight Classification},
  year = {2026},
  url = {https://github.com/Winckedy/MCANet-Tiny},
  note = {Lightweight classifier for pavement crack detection with coordinated attention and pruning}
}
```

## Contact
For questions or issues, please open an issue on GitHub or email:  
2500516015@fzu.edu.cn

Happy generating! 🚗🛣️
