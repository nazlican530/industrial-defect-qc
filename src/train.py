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
# CONFIGURATION
# ==========================================================

SEED = 42
NUM_EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-4
PATIENCE = 3
NUM_WORKERS = 0
IMAGE_SIZE = 224
DROPOUT = 0.4

TRAIN_DIR = Path(
    "data/NEU-DET/split_1799/train/images"
)

VAL_DIR = Path(
    "data/NEU-DET/split_1799/validation/images"
)

MODEL_OUTPUT_DIR = Path("outputs/models")
FIGURE_OUTPUT_DIR = Path("outputs/figures")
RESULT_OUTPUT_DIR = Path("outputs/results")

CHECKPOINT_PATH = (
    MODEL_OUTPUT_DIR
    / "best_model_densenet121.pt"
)

LOSS_FIGURE_PATH = (
    FIGURE_OUTPUT_DIR
    / "loss_curve_densenet121.png"
)

ACCURACY_FIGURE_PATH = (
    FIGURE_OUTPUT_DIR
    / "accuracy_curve_densenet121.png"
)

RESULT_PATH = (
    RESULT_OUTPUT_DIR
    / "training_densenet121.txt"
)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================
# DEVICE
# ==========================================================

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
# IMAGE VALIDATION
# ==========================================================

def is_valid_image_file(path: str) -> bool:
    return (
        Path(path).suffix.lower()
        in VALID_IMAGE_EXTENSIONS
    )


def check_dataset_directory(
    dataset_directory: Path,
    directory_name: str,
) -> None:
    if not dataset_directory.exists():
        raise FileNotFoundError(
            f"\n{directory_name} klasörü bulunamadı:\n"
            f"{dataset_directory.resolve()}\n"
        )

    if not dataset_directory.is_dir():
        raise NotADirectoryError(
            f"\n{directory_name} yolu bir klasör değil:\n"
            f"{dataset_directory.resolve()}\n"
        )

    class_directories = sorted(
        directory
        for directory
        in dataset_directory.iterdir()
        if directory.is_dir()
        and not directory.name.startswith(".")
    )

    if not class_directories:
        raise RuntimeError(
            f"\n{directory_name} klasöründe "
            "sınıf klasörü bulunamadı:\n"
            f"{dataset_directory.resolve()}\n"
        )

    print("\n" + "=" * 70)
    print(f"{directory_name.upper()} DATASET CHECK")
    print("=" * 70)
    print(f"Directory: {dataset_directory}")

    total_images = 0
    empty_classes = []

    for class_directory in class_directories:
        image_files = [
            file_path
            for file_path
            in class_directory.rglob("*")
            if file_path.is_file()
            and is_valid_image_file(
                str(file_path)
            )
        ]

        image_count = len(image_files)
        total_images += image_count

        print(
            f"{class_directory.name}: "
            f"{image_count} images"
        )

        if image_count == 0:
            empty_classes.append(
                class_directory.name
            )

    print(f"Total: {total_images} images")
    print("=" * 70)

    if empty_classes:
        raise RuntimeError(
            "\nAşağıdaki sınıf klasörlerinde "
            "geçerli görüntü bulunamadı:\n"
            f"{empty_classes}\n\n"
            "Görüntülerin doğrudan sınıf "
            "klasörlerinde bulunduğundan emin ol.\n"
        )

    if total_images == 0:
        raise RuntimeError(
            f"\n{directory_name} klasöründe "
            "hiç geçerli görüntü bulunamadı.\n"
        )


# ==========================================================
# TRANSFORMS
# ==========================================================

def get_transforms() -> tuple[
    transforms.Compose,
    transforms.Compose,
]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(7),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
            ),

            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
            ),

            transforms.ToTensor(),

            transforms.Lambda(
                lambda image_tensor:
                torch.clamp(
                    image_tensor
                    + 0.005
                    * torch.randn_like(
                        image_tensor
                    ),
                    0.0,
                    1.0,
                )
            ),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )

    return (
        train_transform,
        validation_transform,
    )


# ==========================================================
# DATASETS AND LOADERS
# ==========================================================

def create_data_loaders(
    device: torch.device,
) -> tuple[
    DataLoader,
    DataLoader,
    datasets.ImageFolder,
    datasets.ImageFolder,
]:
    check_dataset_directory(
        TRAIN_DIR,
        "train",
    )

    check_dataset_directory(
        VAL_DIR,
        "validation",
    )

    (
        train_transform,
        validation_transform,
    ) = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform,
        is_valid_file=is_valid_image_file,
    )

    validation_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=validation_transform,
        is_valid_file=is_valid_image_file,
    )

    if (
        train_dataset.classes
        != validation_dataset.classes
    ):
        raise ValueError(
            "\nTrain ve validation sınıfları "
            "aynı sırada değil.\n"
            f"Train: {train_dataset.classes}\n"
            f"Validation: "
            f"{validation_dataset.classes}\n"
        )

    if len(train_dataset.classes) != 6:
        raise ValueError(
            "\n6 sınıf bekleniyordu fakat "
            f"{len(train_dataset.classes)} "
            "sınıf bulundu.\n"
            f"Sınıflar: "
            f"{train_dataset.classes}\n"
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        generator=generator,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")
    print(
        f"Train images: "
        f"{len(train_dataset)}"
    )
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )
    print(
        f"Classes: "
        f"{train_dataset.classes}"
    )
    print(
        f"Class mapping: "
        f"{train_dataset.class_to_idx}"
    )
    print("=" * 70)

    return (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    )


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

def calculate_class_weights(
    train_dataset: datasets.ImageFolder,
    device: torch.device,
) -> torch.Tensor:
    labels = [
        label
        for _, label
        in train_dataset.samples
    ]

    counts = Counter(labels)

    num_classes = len(
        train_dataset.classes
    )

    class_weights = []

    for class_index in range(num_classes):
        class_count = counts[class_index]

        if class_count == 0:
            raise ValueError(
                f"{class_index} indeksli sınıfta "
                "hiç görüntü bulunamadı."
            )

        class_weights.append(
            1.0 / class_count
        )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for (
        class_name,
        class_index,
    ) in train_dataset.class_to_idx.items():
        print(
            f"{class_name}: "
            f"{counts[class_index]} images"
        )

    print(
        "Class weights:",
        class_weights_tensor,
    )
    print("=" * 70)

    return class_weights_tensor


# ==========================================================
# MODEL
# ==========================================================

def create_model(
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    weights = (
        models.DenseNet121_Weights.DEFAULT
    )

    model = models.densenet121(
        weights=weights
    )

    for parameter in (
        model.features.parameters()
    ):
        parameter.requires_grad = False

    for parameter in (
        model.features
        .denseblock4
        .parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.features
        .norm5
        .parameters()
    ):
        parameter.requires_grad = True

    classifier_input_features = (
        model.classifier.in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),

        nn.Linear(
            classifier_input_features,
            num_classes,
        ),
    )

    for parameter in (
        model.classifier.parameters()
    ):
        parameter.requires_grad = True

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print("Architecture: DenseNet121")
    print("Pretrained weights: ImageNet")
    print(f"Output classes: {num_classes}")
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print("=" * 70)

    return model


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        train_loader,
        desc=(
            f"Epoch {epoch + 1}/"
            f"{NUM_EPOCHS}"
        ),
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        running_loss
        / total_samples
    )

    accuracy = (
        correct_predictions
        / total_samples
    )

    return average_loss, accuracy


# ==========================================================
# VALIDATION
# ==========================================================

def validate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    float,
    list[int],
    list[int],
]:
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

    average_loss = (
        running_loss
        / total_samples
    )

    accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        average_loss,
        accuracy,
        all_predictions,
        all_labels,
    )


# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dataset: datasets.ImageFolder,
    epoch: int,
    best_accuracy: float,
    train_losses: list[float],
    validation_losses: list[float],
    train_accuracies: list[float],
    validation_accuracies: list[float],
) -> None:
    checkpoint = {
        "arch": "densenet121",
        "epoch": epoch,
        "best_epoch": epoch,
        "best_val_accuracy": best_accuracy,

        "model_state": (
            model.state_dict()
        ),

        "optimizer_state": (
            optimizer.state_dict()
        ),

        "scheduler_state": (
            scheduler.state_dict()
        ),

        "class_to_idx": (
            train_dataset.class_to_idx
        ),

        "classes": (
            train_dataset.classes
        ),

        "imagenet_mean": (
            IMAGENET_MEAN
        ),

        "imagenet_std": (
            IMAGENET_STD
        ),

        "train_losses": train_losses,
        "val_losses": validation_losses,

        "train_accuracies": (
            train_accuracies
        ),

        "val_accuracies": (
            validation_accuracies
        ),

        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )


# ==========================================================
# SAVE FIGURES
# ==========================================================

def save_figures(
    train_losses: list[float],
    validation_losses: list[float],
    train_accuracies: list[float],
    validation_accuracies: list[float],
) -> None:
    epochs = range(
        1,
        len(train_losses) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss",
    )

    plt.plot(
        epochs,
        validation_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "DenseNet121 Training and "
        "Validation Loss"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOSS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        validation_accuracies,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "DenseNet121 Training and "
        "Validation Accuracy"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ACCURACY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(
    device: torch.device,
    train_dataset: datasets.ImageFolder,
    validation_dataset: datasets.ImageFolder,
    completed_epochs: int,
    best_epoch: int,
    best_accuracy: float,
    total_training_time: float,
    train_losses: list[float],
    validation_losses: list[float],
    train_accuracies: list[float],
    validation_accuracies: list[float],
) -> None:
    result_lines = [
        "===== DENSENET121 TRAINING RESULTS =====",
        "",
        f"Device: {device}",
        f"Seed: {SEED}",
        f"Classes: {train_dataset.classes}",
        (
            "Class mapping: "
            f"{train_dataset.class_to_idx}"
        ),
        (
            "Train images: "
            f"{len(train_dataset)}"
        ),
        (
            "Validation images: "
            f"{len(validation_dataset)}"
        ),
        f"Batch size: {BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Dropout: {DROPOUT}",
        f"Maximum epochs: {NUM_EPOCHS}",
        f"Completed epochs: {completed_epochs}",
        f"Best epoch: {best_epoch}",
        (
            "Best validation accuracy: "
            f"{best_accuracy:.6f}"
        ),
        (
            "Training duration: "
            f"{total_training_time:.2f} seconds"
        ),
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Loss figure: {LOSS_FIGURE_PATH}",
        (
            "Accuracy figure: "
            f"{ACCURACY_FIGURE_PATH}"
        ),
        "",
        "===== EPOCH HISTORY =====",
    ]

    for epoch_index in range(
        completed_epochs
    ):
        result_lines.append(
            f"Epoch {epoch_index + 1:02d} | "
            f"Train Loss: "
            f"{train_losses[epoch_index]:.6f} | "
            f"Val Loss: "
            f"{validation_losses[epoch_index]:.6f} | "
            f"Train Acc: "
            f"{train_accuracies[epoch_index]:.6f} | "
            f"Val Acc: "
            f"{validation_accuracies[epoch_index]:.6f}"
        )

    RESULT_PATH.write_text(
        "\n".join(result_lines),
        encoding="utf-8",
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    set_seed(SEED)

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("\n" + "=" * 70)
    print("DENSENET121 TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print("=" * 70)

    (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    ) = create_data_loaders(device)

    class_weights = calculate_class_weights(
        train_dataset,
        device,
    )

    model = create_model(
        num_classes=len(
            train_dataset.classes
        ),
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
            parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
        )
    )

    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []

    best_accuracy = 0.0
    best_epoch = 0
    no_improvement = 0

    training_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
        )

        (
            validation_loss,
            validation_accuracy,
            predictions,
            labels,
        ) = validate_model(
            model=model,
            validation_loader=(
                validation_loader
            ),
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        validation_accuracies.append(
            validation_accuracy
        )

        epoch_duration = (
            time.time()
            - epoch_start_time
        )

        minutes = int(
            epoch_duration // 60
        )

        seconds = int(
            epoch_duration % 60
        )

        print("\n" + "=" * 70)
        print(
            f"Epoch {epoch + 1}/"
            f"{NUM_EPOCHS}"
        )
        print("=" * 70)
        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )
        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )
        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )
        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )
        print(
            f"Epoch Time: "
            f"{minutes}m {seconds}s"
        )

        print(
            "\nValidation Classification Report:"
        )

        print(
            classification_report(
                labels,
                predictions,
                target_names=(
                    train_dataset.classes
                ),
                digits=4,
                zero_division=0,
            )
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch + 1
            no_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataset=train_dataset,
                epoch=best_epoch,
                best_accuracy=best_accuracy,
                train_losses=train_losses,
                validation_losses=(
                    validation_losses
                ),
                train_accuracies=(
                    train_accuracies
                ),
                validation_accuracies=(
                    validation_accuracies
                ),
            )

            print(
                "Saved best model -> "
                f"{CHECKPOINT_PATH}"
            )

        else:
            no_improvement += 1

            print(
                "No improvement: "
                f"{no_improvement}/"
                f"{PATIENCE}"
            )

        if no_improvement >= PATIENCE:
            print(
                "\nEarly stopping triggered."
            )
            break

    total_training_time = (
        time.time()
        - training_start_time
    )

    completed_epochs = len(
        train_losses
    )

    save_figures(
        train_losses=train_losses,
        validation_losses=validation_losses,
        train_accuracies=train_accuracies,
        validation_accuracies=(
            validation_accuracies
        ),
    )

    save_results(
        device=device,
        train_dataset=train_dataset,
        validation_dataset=(
            validation_dataset
        ),
        completed_epochs=(
            completed_epochs
        ),
        best_epoch=best_epoch,
        best_accuracy=best_accuracy,
        total_training_time=(
            total_training_time
        ),
        train_losses=train_losses,
        validation_losses=(
            validation_losses
        ),
        train_accuracies=(
            train_accuracies
        ),
        validation_accuracies=(
            validation_accuracies
        ),
    )

    print("\n" + "=" * 70)
    print("DENSENET121 TRAINING COMPLETED")
    print("=" * 70)
    print(
        f"Completed epochs: "
        f"{completed_epochs}"
    )
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )
    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.4f}"
    )
    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )
    print(
        f"Loss figure: "
        f"{LOSS_FIGURE_PATH}"
    )
    print(
        f"Accuracy figure: "
        f"{ACCURACY_FIGURE_PATH}"
    )
    print(
        f"Results: "
        f"{RESULT_PATH}"
    )
    print(
        f"Training time: "
        f"{total_training_time:.2f} seconds"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()