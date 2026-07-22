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

SEED = 42
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-4
PATIENCE = 4
DROPOUT = 0.2
LABEL_SMOOTHING = 0.10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    set_seed()

    device = get_device()
    print("Device:", device)

    # DenseNet ve diğer modellerle aynı split kullanılmalı.
    train_dir = Path(
        "data/NEU-DET/split_1799/train/images"
    )

    val_dir = Path(
        "data/NEU-DET/split_1799/validation/images"
    )

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_dir}"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}"
        )

    model_output_dir = Path("outputs/models")
    figure_output_dir = Path("outputs/figures")

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        model_output_dir
        / "best_model_mobilenetv3.pt"
    )

    # ======================================================
    # TRANSFORMS
    # ======================================================

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

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
                lambda image: torch.clamp(
                    image
                    + 0.008
                    * torch.randn_like(image),
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

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD
            ),
        ]
    )

    # ======================================================
    # DATASETS
    # ======================================================

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
            "Train and validation class orders differ.\n"
            f"Train: {train_dataset.classes}\n"
            f"Validation: {val_dataset.classes}"
        )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print(
        "class_to_idx:",
        train_dataset.class_to_idx
    )
    print(
        "Train images:",
        len(train_dataset)
    )
    print(
        "Validation images:",
        len(val_dataset)
    )

    # ======================================================
    # CLASS WEIGHTS
    # ======================================================

    labels = [
        label
        for _, label in train_dataset.samples
    ]

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

    # ======================================================
    # DATA LOADERS
    # ======================================================

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    # ======================================================
    # MODEL
    # ======================================================

    weights = (
        models.MobileNet_V3_Large_Weights.DEFAULT
    )

    model = models.mobilenet_v3_large(
        weights=weights
    )

    in_features = (
        model.classifier[3].in_features
    )

    # MobileNetV3'ün mevcut classifier yapısı korunuyor.
    model.classifier[2] = nn.Dropout(
        p=DROPOUT,
        inplace=True
    )

    model.classifier[3] = nn.Linear(
        in_features,
        num_classes
    )

    model = model.to(device)

    # ======================================================
    # LOSS, OPTIMIZER, SCHEDULER
    # ======================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
    )

    # ======================================================
    # TRAINING VARIABLES
    # ======================================================

    best_val_accuracy = 0.0
    best_epoch = 0
    no_improvement = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---------------- TRAIN ----------------

        model.train()

        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for images, labels_batch in progress_bar:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels_batch
            )

            loss.backward()
            optimizer.step()

            batch_size = labels_batch.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            train_total += batch_size

            train_correct += (
                logits.argmax(dim=1)
                == labels_batch
            ).sum().item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            total_train_loss / train_total
        )

        train_accuracy = (
            train_correct / train_total
        )

        train_losses.append(train_loss)
        train_accuracies.append(
            train_accuracy
        )

        # ---------------- VALIDATION ----------------

        model.eval()

        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(
                    device
                )

                logits = model(images)

                loss = criterion(
                    logits,
                    labels_batch
                )

                predictions = logits.argmax(
                    dim=1
                )

                batch_size = labels_batch.size(0)

                total_val_loss += (
                    loss.item() * batch_size
                )

                val_total += batch_size

                val_correct += (
                    predictions
                    == labels_batch
                ).sum().item()

                all_predictions.extend(
                    predictions
                    .cpu()
                    .numpy()
                    .tolist()
                )

                all_labels.extend(
                    labels_batch
                    .cpu()
                    .numpy()
                    .tolist()
                )

        val_loss = (
            total_val_loss / val_total
        )

        val_accuracy = (
            val_correct / val_total
        )

        val_losses.append(val_loss)
        val_accuracies.append(
            val_accuracy
        )

        scheduler.step()

        elapsed_time = (
            time.time() - epoch_start
        )

        minutes = int(
            elapsed_time // 60
        )

        seconds = int(
            elapsed_time % 60
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            "Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            "Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Epoch Time: {minutes}m {seconds}s"
        )

        print(
            "\nValidation Classification Report:"
        )

        print(
            classification_report(
                all_labels,
                all_predictions,
                target_names=class_names,
                digits=4,
                zero_division=0
            )
        )

        # ==================================================
        # SAVE BEST MODEL
        # ==================================================

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            no_improvement = 0

            torch.save(
                {
                    "arch": "mobilenet_v3_large",
                    "model_state": (
                        model.state_dict()
                    ),
                    "class_to_idx": (
                        train_dataset.class_to_idx
                    ),
                    "classes": class_names,
                    "imagenet_mean": (
                        IMAGENET_MEAN
                    ),
                    "imagenet_std": (
                        IMAGENET_STD
                    ),
                    "train_losses": (
                        train_losses
                    ),
                    "val_losses": (
                        val_losses
                    ),
                    "train_accuracies": (
                        train_accuracies
                    ),
                    "val_accuracies": (
                        val_accuracies
                    ),
                    "best_epoch": (
                        best_epoch
                    ),
                    "best_val_accuracy": (
                        best_val_accuracy
                    ),
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": (
                        LEARNING_RATE
                    ),
                    "weight_decay": (
                        WEIGHT_DECAY
                    ),
                    "dropout": DROPOUT,
                    "label_smoothing": (
                        LABEL_SMOOTHING
                    ),
                    "epochs": EPOCHS,
                },
                checkpoint_path
            )

            print(
                "Saved best model -> "
                f"{checkpoint_path}"
            )

            print(
                "Saved path:",
                checkpoint_path.resolve()
            )

            print(
                "Saved size:",
                checkpoint_path.stat().st_size,
                "bytes"
            )

        else:
            no_improvement += 1

            print(
                "No improvement: "
                f"{no_improvement}/{PATIENCE}"
            )

        if no_improvement >= PATIENCE:
            print(
                "Early stopping triggered"
            )
            break

    # ======================================================
    # TRAINING COMPLETE
    # ======================================================

    print("\nTraining Complete")
    print("Best Epoch:", best_epoch)
    print(
        "Best Validation Accuracy:",
        best_val_accuracy
    )

    # ======================================================
    # LOSS CURVE
    # ======================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "MobileNetV3-Large Training "
        "and Validation Losses"
    )

    plt.legend()
    plt.tight_layout()

    loss_figure_path = (
        figure_output_dir
        / "mobilenetv3_loss_curve.png"
    )

    plt.savefig(
        loss_figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Loss curve saved -> "
        f"{loss_figure_path}"
    )

    # ======================================================
    # ACCURACY CURVE
    # ======================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "MobileNetV3-Large Training "
        "and Validation Accuracy"
    )

    plt.legend()
    plt.tight_layout()

    accuracy_figure_path = (
        figure_output_dir
        / "mobilenetv3_accuracy_curve.png"
    )

    plt.savefig(
        accuracy_figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Accuracy curve saved -> "
        f"{accuracy_figure_path}"
    )


if __name__ == "__main__":
    main()