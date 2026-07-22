"""
MobileNetV2 Behavior Classifier Training
==========================================
Hardware:  RTX 3050 6 GB  |  i5-13420H  |  16 GB RAM
Dataset:   data/behavior/{class_folders}/  (cropped person images)
Classes:   Sitting, Sleeping, Mobile, Reading, HandRaising

Usage:
    python train_classifier.py

The script expects this folder structure:
    data/behavior/
    ├── sitting/       (images)
    ├── sleeping/      (images)
    ├── mobile/        (images)
    ├── reading/       (images)
    └── hand_raising/  (images)

Outputs:
    models/behavior_classifier.pth  (best model weights)
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR = "data/behavior"
OUTPUT_PATH = "models/behavior_classifier.pth"
NUM_CLASSES = 5
CLASS_NAMES = ["hand_raising", "mobile", "reading", "sitting", "sleeping"]
# ^ sorted alphabetically — ImageFolder uses this order

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 8          # Early stopping patience
LR = 0.001
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Training transforms with augmentation, val with just resize+normalize."""
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tfm, val_tfm


def build_model():
    """MobileNetV2 with custom classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze early layers (first 14 out of 19 blocks)
    for i, (name, param) in enumerate(model.features.named_parameters()):
        if int(name.split(".")[0]) < 14:
            param.requires_grad = False

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )

    return model.to(DEVICE)





def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ─── Data ────────────────────────────────────────────────────
    train_tfm, val_tfm = get_transforms()

    # Use one dataset to inspect classes, then create separate ones for train/val
    inspect_ds = datasets.ImageFolder(DATA_DIR)
    print(f"Classes: {inspect_ds.classes}")
    print(f"Total images: {len(inspect_ds)}")
    for cls, idx in inspect_ds.class_to_idx.items():
        count = sum(1 for _, t in inspect_ds.samples if t == idx)
        print(f"  {cls}: {count}")
    print()

    # Split indices 80/20
    total = len(inspect_ds)
    train_size = int(0.8 * total)
    val_size = total - train_size

    indices = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create separate datasets with different transforms
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfm)

    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_dataset, val_indices)

    # Weighted sampler for class imbalance — built from TRAIN indices only
    train_targets = [inspect_ds.targets[i] for i in train_indices]
    class_counts = [0] * NUM_CLASSES
    for t in train_targets:
        class_counts[t] += 1
    class_weights = [1.0 / c if c > 0 else 0 for c in class_counts]
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"Train: {train_size} | Val: {val_size}\n")

    # ─── Model ───────────────────────────────────────────────────
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ─── Training Loop ───────────────────────────────────────────
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

        train_loss = running_loss / total_samples
        train_acc = correct / total_samples

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1%}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  >> Saved best model (val acc: {val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[STOP] Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining complete! Best val accuracy: {best_acc:.1%}")
    print(f"Model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
