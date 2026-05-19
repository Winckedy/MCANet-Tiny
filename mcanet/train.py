"""Training loop with configurable optimizer and scheduler."""
import time
from typing import Dict, Any, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    validate_every_n_epochs: int = 1,
    optimizer_type: str = 'adamw',       # 'adam' or 'adamw'
    scheduler_type: str = 'cosine',      # 'cosine' or 'step'
    warmup_epochs: int = 3,
    base_lr: float = 0.001,
    weight_decay: float = 0.01
) -> Tuple[Dict[str, Any], nn.Module]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Scheduler selection
    if scheduler_type.lower() == 'cosine':
        scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    elif scheduler_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    history = {
        'epoch': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'iteration': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'performance': {'epoch_times': [], 'train_fps': [], 'val_fps': []}
    }

    print(f"Training config: Optimizer={optimizer_type}, Scheduler={scheduler_type}, Warmup={warmup_epochs}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0

        iter_start_idx = len(history['iteration']['train_loss'])

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            batch_corrects = (preds == labels).sum().item()
            running_corrects += batch_corrects

            history['iteration']['train_loss'].append(batch_loss)
            history['iteration']['train_acc'].append(batch_corrects / inputs.size(0))
            history['iteration']['val_loss'].append(float('nan'))
            history['iteration']['val_acc'].append(float('nan'))

            train_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{batch_corrects / inputs.size(0):.4f}'
            })

        scheduler.step()

        train_dataset_size = len(train_loader.dataset)  # type: ignore[arg-type]
        epoch_loss = running_loss / train_dataset_size
        epoch_acc = running_corrects / train_dataset_size
        history['epoch']['train_loss'].append(epoch_loss)
        history['epoch']['train_acc'].append(epoch_acc)
        train_time = time.time() - epoch_start_time
        train_fps = train_dataset_size / train_time if train_time > 0 else 0
        history['performance']['train_fps'].append(train_fps)

        if (epoch + 1) % validate_every_n_epochs == 0 or (epoch + 1) == num_epochs:
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            val_start_time = time.time()
            val_dataset_size = len(val_loader.dataset)  # type: ignore[arg-type]

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
                for inputs, labels in val_bar:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_running_corrects += (preds == labels).sum().item()
                    val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            val_time = time.time() - val_start_time
            val_fps = val_dataset_size / val_time if val_time > 0 else 0
            val_epoch_loss = val_running_loss / val_dataset_size
            val_epoch_acc = val_running_corrects / val_dataset_size

            history['performance']['val_fps'].append(val_fps)
            history['epoch']['val_loss'].append(val_epoch_loss)
            history['epoch']['val_acc'].append(val_epoch_acc)

            iter_end_idx = len(history['iteration']['train_loss'])
            num_steps_in_epoch = iter_end_idx - iter_start_idx
            history['iteration']['val_loss'][iter_start_idx:iter_end_idx] = [val_epoch_loss] * num_steps_in_epoch
            history['iteration']['val_acc'][iter_start_idx:iter_end_idx] = [val_epoch_acc] * num_steps_in_epoch

            epoch_time = time.time() - epoch_start_time
            history['performance']['epoch_times'].append(epoch_time)

            print(f"\nEpoch {epoch+1}/{num_epochs} -> "
                  f"Time: {epoch_time:.2f}s | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        else:
            history['performance']['val_fps'].append(0)
            history['epoch']['val_loss'].append(float('nan'))
            history['epoch']['val_acc'].append(float('nan'))
            epoch_time = time.time() - epoch_start_time
            history['performance']['epoch_times'].append(epoch_time)
            print(f"\nEpoch {epoch+1}/{num_epochs} -> Time: {epoch_time:.2f}s | "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} (Validation skipped)")

        print("-" * 110)

    return history, model