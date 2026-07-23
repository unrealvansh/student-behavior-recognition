"""
MobileNetV2 Behavior Classifier Training — Full Fine-Tuning with PadToSquare
=============================================================================
Hardware:  RTX 3050 6 GB  |  i5-13420H  |  16 GB RAM
Dataset:   data/behavior/{class_folders}/  (cropped person images)
Classes:   Sitting, Sleeping, Mobile, Reading, HandRaising

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

BATCH_SIZE = 32
EPOCHS = 35
PATIENCE = 5          # Abort after 5 epochs with no validation loss improvement
LR_BACKBONE = 1e-5      # Lower learning rate for fine-tuning pre-trained backbone
LR_HEAD = 1e-3          # Higher learning rate for new classifier head
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PadToSquare(object):
    """Pad PIL Image to a square with neutral gray background before resizing.
    Preserves original aspect ratio of person crop without squishing body/face.
    """
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        padding = (hp, vp, max_dim - w - hp, max_dim - h - vp)
        return transforms.functional.pad(img, padding, fill=128)


def get_transforms():
    """Enhanced training transforms with PadToSquare aspect-ratio preservation."""
    train_tfm = transforms.Compose([
        PadToSquare(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_tfm = transforms.Compose([
        PadToSquare(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tfm, val_tfm


def build_model(num_classes):
    """MobileNetV2 with custom classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes),
    )
    return model.to(DEVICE)


def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: Two-Stage Fine-Tuning with PadToSquare ({EPOCHS} epochs)\n")

    # ─── Data ────────────────────────────────────────────────────
    train_tfm, val_tfm = get_transforms()

    inspect_ds = datasets.ImageFolder(DATA_DIR)
    print(f"Classes: {inspect_ds.classes}")
    print(f"Total images: {len(inspect_ds)}")
    for cls, idx in inspect_ds.class_to_idx.items():
        count = sum(1 for _, t in inspect_ds.samples if t == idx)
        print(f"  {cls}: {count}")
    print()

    # Create datasets and dataloaders
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfm)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")

    # Calculate class weights for handling imbalance
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() # Normalize
    class_weights = class_weights.to(DEVICE)
    print(f"Applied class weights: {class_weights.cpu().numpy().round(3)}")


    # ─── Stage 1: Train Classifier Head ──────────────────────────
    print("--- Stage 1: Training Classifier Head (Backbone Frozen) ---")
    model = build_model(NUM_CLASSES)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)

    for epoch in range(5): # Train head for 5 epochs
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1:2d}/5: Head training complete.")

    # ─── Stage 2: Full Fine-Tuning ──────────────────────────────
    print("\n--- Stage 2: Full Fine-Tuning (Backbone Unfrozen) ---")

    # Unfreeze backbone
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss = float('inf')
    epochs_no_improve = 0

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

        saved_msg = ""
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_PATH)
            epochs_no_improve = 0
            saved_msg = f"  >> Saved best model (val loss: {val_loss:.4f}, acc: {val_acc:.1%})"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:2d}/{EPOCHS:2d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}{saved_msg}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement.")
            break

    print(f"\nTraining complete!")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
