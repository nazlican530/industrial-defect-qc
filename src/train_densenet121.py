import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    set_seed(42)

    device = get_device()
    print("Device:", device)

    train_dir = "data/NEU-DET/split_1799/train/images"
    val_dir = "data/NEU-DET/split_1799/validation/images"

    model_out_dir = Path("outputs/models")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    fig_out_dir = Path("outputs/figures")
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_out_dir / "best_model_densenet121.pt"

    # ResNet50 ile aynı augmentation düzeni
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.clamp(
                x + 0.008 * torch.randn_like(x),
                0.0,
                1.0
            )
        ),
        transforms.RandomErasing(
            p=0.20,
            scale=(0.02, 0.08),
            ratio=(0.5, 2.0),
            value="random"
        ),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train ve validation sınıf sıralamaları farklı.\n"
            f"Train: {train_dataset.classes}\n"
            f"Validation: {val_dataset.classes}"
        )

    print("Classes:", train_dataset.classes)
    print("class_to_idx:", train_dataset.class_to_idx)

    num_classes = len(train_dataset.classes)

    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # Sınıflar dengeli olsa da diğer modellerle aynı yöntem korunuyor.
    labels = [label for _, label in train_dataset.samples]
    counts = Counter(labels)

    class_weights = [
        1.0 / counts[class_index]
        for class_index in range(num_classes)
    ]

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    print("Class weights:", class_weights)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    # DenseNet121: tüm katmanlar eğitiliyor.
    # Önceki düşük sonucun ana nedenlerinden biri omurganın büyük bölümünün
    # dondurulmuş olmasıydı.
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.10
    )

    # ResNet50 ile aynı optimizer ayarları
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    patience = 3
    no_improve = 0
    best_acc = 0.0
    best_epoch = 0

    epochs = 10

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # ---------------- TRAIN ----------------
        model.train()

        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, labels_batch in loop:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels_batch)

            loss.backward()
            optimizer.step()

            batch_size = labels_batch.size(0)

            total_train_loss += loss.item() * batch_size
            train_total += batch_size
            train_correct += (
                logits.argmax(dim=1) == labels_batch
            ).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = (
            total_train_loss / train_total
            if train_total else 0.0
        )
        train_acc = (
            train_correct / train_total
            if train_total else 0.0
        )

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # ---------------- VALIDATION ----------------
        model.eval()

        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)

                logits = model(images)
                loss = criterion(logits, labels_batch)

                batch_size = labels_batch.size(0)
                predictions = logits.argmax(dim=1)

                total_val_loss += loss.item() * batch_size
                val_total += batch_size
                val_correct += (
                    predictions == labels_batch
                ).sum().item()

                preds_all.extend(
                    predictions.cpu().numpy().tolist()
                )
                labels_all.extend(
                    labels_batch.cpu().numpy().tolist()
                )

        avg_val_loss = (
            total_val_loss / val_total
            if val_total else 0.0
        )
        val_acc = (
            val_correct / val_total
            if val_total else 0.0
        )

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Epoch Time: {minutes}m {seconds}s")

        print("\nValidation Classification Report:")
        print(
            classification_report(
                labels_all,
                preds_all,
                target_names=train_dataset.classes,
                digits=4,
                zero_division=0
            )
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0

            torch.save(
                {
                    "arch": "densenet121",
                    "model_state": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                    "classes": train_dataset.classes,
                    "imagenet_mean": IMAGENET_MEAN,
                    "imagenet_std": IMAGENET_STD,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accuracies,
                    "val_accuracies": val_accuracies,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_acc,
                    "seed": 42,
                    "batch_size": 32,
                    "learning_rate": 5e-5,
                    "weight_decay": 5e-4,
                    "dropout": 0.35,
                    "label_smoothing": 0.10,
                    "epochs": epochs,
                },
                ckpt_path
            )

            print(f"Saved best model -> {ckpt_path}")
            print("Saved path:", ckpt_path.resolve())
            print("Saved size:", ckpt_path.stat().st_size, "bytes")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{patience}")

        if no_improve >= patience:
            print("Early stopping triggered")
            break

    print("\nTraining Complete")
    print("Best Epoch:", best_epoch)
    print("Best Validation Accuracy:", best_acc)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DenseNet121 Training and Validation Losses")
    plt.legend()
    plt.tight_layout()

    loss_fig_path = fig_out_dir / "densenet121_loss_curve.png"
    plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved -> {loss_fig_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("DenseNet121 Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    accuracy_fig_path = (
        fig_out_dir / "densenet121_accuracy_curve.png"
    )
    plt.savefig( 
        accuracy_fig_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Accuracy curve saved -> {accuracy_fig_path}")


if __name__ == "__main__":
    main()
