"""Data loading and model creation utilities."""
from pathlib import Path
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from .model import mcanet_tiny


def create_data_loaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 64,
    img_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, shear=10, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    num_workers = 8
    print(f"Using {num_workers} subprocesses to load data...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader, train_dataset.class_to_idx


def create_model(num_classes: int, pretrained: bool = True, attn_type: str = 'ca') -> nn.Module:
    """
    Create MCANet-Tiny model with specified attention type.
    """
    model = mcanet_tiny(pretrained=pretrained, num_classes=1000, attn_type=attn_type)
    for param in model.parameters():
        param.requires_grad = True
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model