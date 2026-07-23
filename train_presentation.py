"""
Train a specialized "Presentation Model" on a small, perfectly-labeled dataset
derived from the specific images that will be used in a demonstration.

This script intentionally overfits to the demo images to ensure a flawless
proof-of-concept presentation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# --- Configuration ---
# Point to the small, perfectly-labeled dataset you just created
DATA_DIR = "data/presentation_data"
# Save the model to a different file to avoid overwriting your main model
OUTPUT_PATH = "models/presentation_classifier.pth"

NUM_CLASSES = 5
CLASS_NAMES = ["hand_raising", "mobile", "reading", "sitting", "sleeping"]

BATCH_SIZE = 8  # Smaller batch size for a small dataset
EPOCHS = 15     # Train for a few epochs to quickly learn the small dataset
LR = 1e-4
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model and Transforms (simplified for overfitting) ---
def get_transforms():
    """Simplified transforms for the presentation model."""
    # We use the same transforms for training and validation because we are
    # not concerned with generalization, only with memorizing this small set.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Simple resize, no complex augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def build_model(num_classes):
    """MobileNetV2 with a simple classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Unfreeze all layers immediately for fast learning on the small dataset
    for param in model.parameters():
        param.requires_grad = True
    model.classifier = nn.Linear(model.last_channel, num_classes)
    return model.to(DEVICE)

# --- Main Training Script ---
def train_presentation_model():
    print("="*60)
    print("TRAINING SPECIALIZED PRESENTATION MODEL")
    print("="*60)
    print(f"Device: {DEVICE}")

    # --- Data ---
    tfm = get_transforms()
    dataset = datasets.ImageFolder(DATA_DIR, transform=tfm)

    if len(dataset) == 0:
        print(f"ERROR: No images found in {os.path.abspath(DATA_DIR)}.")
        print("Please ensure you have sorted the crops from generate_presentation_crops.py into the correct subfolders.")
        return

    # Use all data for training. No validation split needed.
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training on {len(dataset)} perfectly-labeled images from '{DATA_DIR}'")

    # --- Model, Loss, Optimizer ---
    model = build_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # --- Training Loop ---
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1:2d}/{EPOCHS:2d} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.1%}")

    # --- Save the Final Model ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("\nTraining complete!")
    print(f"Presentation model saved to: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    train_presentation_model()
