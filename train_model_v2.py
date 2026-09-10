"""High-accuracy rice disease training script.

Uses ImageNet-pretrained ResNet18, class-balanced sampling, augmentation,
label smoothing, validation-based early stopping, and optional test evaluation.
It does not delete or modify dataset images.
"""
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")
MODEL_DIR = os.path.join(ROOT, "model")
TEMP_MODEL = os.path.join(MODEL_DIR, "disease_model_best.pth")
FINAL_MODEL = os.path.join(MODEL_DIR, "disease_model.pth")
CLASS_FILE = os.path.join(MODEL_DIR, "disease_classes.json")

BATCH_SIZE = 16
EPOCHS = 25
VAL_FRACTION = 0.15
PATIENCE = 6
NUM_WORKERS = 0  # safest on Windows
IMAGE_SIZE = 224


def make_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def split_indices(targets):
    rng = np.random.default_rng(SEED)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * VAL_FRACTION)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(len(loader.dataset.dataset.classes) if isinstance(loader.dataset, Subset) else 1, dtype=np.int64)
    class_total = np.zeros_like(class_correct)
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
    return total_loss / max(total, 1), correct / max(total, 1), all_true, all_pred


def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Missing dataset folder: {TRAIN_DIR}")

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.10),
        transforms.RandomRotation(18),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base = datasets.ImageFolder(TRAIN_DIR)
    classes = base.classes
    if len(classes) < 2:
        raise RuntimeError("Need at least 2 disease classes.")
    if "Tungro" not in classes:
        print("WARNING: Tungro class is not present in dataset/train yet.")

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Total training images: {len(base)}")
    train_idx, val_idx = split_indices(base.targets)

    train_ds_full = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds_full = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    # Balanced sampling prevents large classes from dominating training.
    counts = np.bincount(np.asarray(base.targets)[train_idx], minlength=len(classes))
    weights_by_class = 1.0 / np.maximum(counts, 1)
    sample_weights = [weights_by_class[base.targets[i]] for i in train_idx]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = make_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 2e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.4, patience=2, min_lr=1e-6)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc = -1.0
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        start = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        train_acc = correct / max(seen, 1)
        elapsed = time.time() - start
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={running_loss/max(seen,1):.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            stale = 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "architecture": "resnet18",
                "validation_accuracy": best_acc,
            }, TEMP_MODEL)
            print("  New best model saved.")
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping: validation accuracy stopped improving.")
                break

    if not os.path.exists(TEMP_MODEL):
        raise RuntimeError("Training failed: no best checkpoint was produced.")

    # Only replace the application model after successful training.
    backup = FINAL_MODEL + ".backup"
    if os.path.exists(FINAL_MODEL):
        shutil.copy2(FINAL_MODEL, backup)
    os.replace(TEMP_MODEL, FINAL_MODEL)
    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print("\nBEST VALIDATION ACCURACY:", f"{best_acc * 100:.2f}%")
    print("Saved:", FINAL_MODEL)
    print("Classes:", classes)
    if os.path.exists(TEST_DIR):
        print("\nNote: dataset/test is evaluated separately only when its classes are available; the validation score above is the primary all-class score.")


if __name__ == "__main__":
    main()
