"""Visualization and report saving utilities."""
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_iteration_history(history: Dict[str, Any], save_dir: Path, smoothing_window: int = 50):
    iter_history = history['iteration']
    if not iter_history['train_loss']:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=100)
    fig.suptitle('Training History per Iteration', fontsize=16)

    train_loss_s = pd.Series(iter_history['train_loss'])
    train_acc_s = pd.Series(iter_history['train_acc'])
    val_loss_s = pd.Series(iter_history['val_loss'])
    val_acc_s = pd.Series(iter_history['val_acc'])

    train_loss_smooth = train_loss_s.rolling(window=smoothing_window, min_periods=1).mean()
    train_acc_smooth = train_acc_s.rolling(window=smoothing_window, min_periods=1).mean()

    axes[0].plot(train_loss_s, label='Train Loss (Raw)', color='lightblue', alpha=0.3)
    axes[0].plot(train_loss_smooth, label=f'Train Loss (Smooth w={smoothing_window})',
                 color='darkblue')
    axes[0].plot(val_loss_s, label='Validation Loss', color='red', alpha=0.8,
                 linewidth=2, linestyle='--')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')

    axes[1].plot(train_acc_s, label='Train Acc (Raw)', color='lightgreen', alpha=0.3)
    axes[1].plot(train_acc_smooth, label=f'Train Acc (Smooth w={smoothing_window})',
                 color='darkgreen')
    axes[1].plot(val_acc_s, label='Validation Acc', color='red', alpha=0.8,
                 linewidth=2, linestyle='--')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_dir / 'iteration_training_history.png', dpi=300)
    plt.close(fig)


def save_history_to_csv(history: Dict[str, Any], save_dir: Path):
    try:
        epoch_history_df = pd.DataFrame(history['epoch'])
        epoch_history_df.index = np.arange(1, len(epoch_history_df) + 1)
        epoch_history_df.index.name = 'epoch'
        epoch_csv_path = save_dir / 'epoch_training_history.csv'
        epoch_history_df.to_csv(epoch_csv_path)

        iteration_history_df = pd.DataFrame(history['iteration'])
        iteration_history_df.index = np.arange(1, len(iteration_history_df) + 1)
        iteration_history_df.index.name = 'iteration'
        iteration_csv_path = save_dir / 'iteration_training_history.csv'
        iteration_history_df.to_csv(iteration_csv_path)
        print(f"Training history saved to CSV files:\n- {epoch_csv_path}\n- {iteration_csv_path}")
    except Exception as e:
        print(f"Error saving CSV history files: {e}")


def save_full_report(
    test_metrics: Dict[str, Any],
    model_stats: Dict[str, Any],
    save_dir: Path,
    class_to_idx: Dict[str, int]
):
    report_path = save_dir / 'full_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========== Model Performance Report ==========\n")
        f.write(f"Accuracy:   {test_metrics['Accuracy']:.4f}\n")
        f.write(f"Precision:  {test_metrics['Precision']:.4f}\n")
        f.write(f"Recall:     {test_metrics['Recall']:.4f}\n")
        f.write(f"F1-Score:   {test_metrics['F1-Score']:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"GFLOPs:     {model_stats['GFLOPs']:.2f}\n")
        f.write(f"Parameters: {model_stats['Parameters']:.2f} M\n")
        f.write(f"Size:       {model_stats['Size']:.2f} MB\n")
        f.write(f"Inference Time: {test_metrics['Inference Time']:.2f} ms\n")
        f.write(f"Inference FPS:  {test_metrics['Inference FPS']:.2f}\n")
        f.write("\n" + "=" * 20 + " Detailed Classification Report " + "=" * 20 + "\n")
        f.write(test_metrics['Detailed Report'])
        f.write("\n" + "=" * 56 + "\n")

    cm = test_metrics['conf_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=list(class_to_idx.keys()),
                yticklabels=list(class_to_idx.keys()),
                annot_kws={"size": 12})
    plt.title('Normalized Confusion Matrix (CA)', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    print(f"Full report (with Precision/Recall/F1) saved to: {save_dir}")