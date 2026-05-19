"""Evaluation and model statistics utilities."""
import time
import io
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix
)
from thop import profile


def calculate_model_stats(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Dict[str, Any]:
    stats = {}
    model.to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    stats['Parameters'] = total_params / 1e6  # M

    try:
        dummy_input = torch.randn(input_size).to(device)
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        stats['GFLOPs'] = (macs * 2) / 1e9
    except Exception as e:
        print(f"Failed to compute FLOPs: {e}")
        stats['GFLOPs'] = 0

    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
        stats['Size'] = size_bytes / (1024 * 1024)  # MB
    except Exception as e:
        print(f"Failed to compute model size: {e}")
        stats['Size'] = 0

    return stats


def evaluate_model(
    model: nn.Module,
    test_dir: Path,
    device: torch.device,
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    num_workers: int = 8
) -> Dict[str, Any]:
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(str(test_dir), transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model.to(device).eval()
    all_predictions = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            _ = model(dummy_input)

        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Discard first few warm-up measurements if any
    if len(inference_times) > 10:
        inference_times = inference_times[5:]
    avg_latency_ms = np.mean(inference_times) * 1000
    fps = 1.0 / (avg_latency_ms / 1000) if avg_latency_ms > 0 else 0

    true_labels = np.array(all_labels)
    preds = np.array(all_predictions)

    precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
    recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
    target_names = list(class_to_idx.keys())
    detailed_report = classification_report(true_labels, preds,
                                            target_names=target_names, zero_division=0)

    metrics = {
        'Accuracy': accuracy_score(true_labels, preds),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Detailed Report': detailed_report,
        'conf_matrix': confusion_matrix(true_labels, preds),
        'Inference FPS': fps,
        'Inference Time': avg_latency_ms
    }
    return metrics