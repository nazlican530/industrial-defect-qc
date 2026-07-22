import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ==========================================================
# DEVICE
# ==========================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ==========================================================
# CHECKPOINT HELPERS
# ==========================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]

    return checkpoint


def infer_arch_from_state_dict(state_dict):
    keys = list(state_dict.keys())

    if any(key.startswith("features.denseblock") for key in keys):
        return "densenet121"

    if any(".conv3.weight" in key for key in keys):
        return "resnet50"

    if any(key.startswith("features.16") for key in keys):
        return "mobilenet_v3_large"

    if (
        any(key.startswith("features.") for key in keys)
        and any(key.startswith("classifier.") for key in keys)
    ):
        return "efficientnet_b0"

    raise ValueError(
        "Could not infer architecture from checkpoint."
    )


# ==========================================================
# MODEL LOADING
# ==========================================================

def load_model(
    checkpoint_path,
    device,
    dataset_class_to_idx
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    if isinstance(checkpoint, dict) and "arch" in checkpoint:
        architecture = checkpoint["arch"]
    else:
        architecture = infer_arch_from_state_dict(state_dict)

    if (
        isinstance(checkpoint, dict)
        and "class_to_idx" in checkpoint
    ):
        class_to_idx = checkpoint["class_to_idx"]
    else:
        class_to_idx = dataset_class_to_idx

    num_classes = len(class_to_idx)

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=None
        )

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

        try:
            model.load_state_dict(
                state_dict,
                strict=True
            )

        except RuntimeError:
            fixed_state_dict = {}

            for key, value in state_dict.items():
                new_key = key.replace(
                    "classifier.1.1",
                    "classifier.1"
                )

                fixed_state_dict[new_key] = value

            model.load_state_dict(
                fixed_state_dict,
                strict=True
            )

    elif architecture == "resnet50":
        model = models.resnet50(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        model.load_state_dict(
            state_dict,
            strict=True
        )

    elif architecture == "densenet121":
        model = models.densenet121(
            weights=None
        )

        in_features = model.classifier.in_features

        model.classifier = nn.Linear(
            in_features,
            num_classes
        )

        try:
            model.load_state_dict(
                state_dict,
                strict=True
            )

        except RuntimeError:
            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(
                    in_features,
                    num_classes
                )
            )

            model.load_state_dict(
                state_dict,
                strict=True
            )

    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=None
        )

        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            num_classes
        )

        model.load_state_dict(
            state_dict,
            strict=True
        )

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    model = model.to(device)
    model.eval()

    if isinstance(checkpoint, dict):
        mean = tuple(
            checkpoint.get(
                "imagenet_mean",
                (0.485, 0.456, 0.406)
            )
        )

        std = tuple(
            checkpoint.get(
                "imagenet_std",
                (0.229, 0.224, 0.225)
            )
        )
    else:
        mean = (
            0.485,
            0.456,
            0.406
        )

        std = (
            0.229,
            0.224,
            0.225
        )

    return (
        model,
        architecture,
        class_to_idx,
        mean,
        std
    )


# ==========================================================
# INFERENCE
# ==========================================================

def collect_predictions(
    checkpoint_path,
    test_dir
):
    device = get_device()

    base_dataset = datasets.ImageFolder(
        test_dir
    )

    (
        model,
        architecture,
        class_to_idx,
        mean,
        std
    ) = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        dataset_class_to_idx=base_dataset.class_to_idx
    )

    if class_to_idx != base_dataset.class_to_idx:
        raise ValueError(
            "Checkpoint and dataset class_to_idx do not match."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    dataset = datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    all_labels = np.asarray(
        all_labels,
        dtype=np.int64
    )

    return (
        architecture,
        all_probabilities,
        all_labels
    )


# ==========================================================
# RELIABILITY DIAGRAM
# ==========================================================

def plot_reliability_diagram(
    probabilities,
    y_true,
    model_name,
    n_bins=15
):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)

    correctness = (
        predictions == y_true
    ).astype(np.float64)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    bin_centers = (
        bin_edges[:-1] + bin_edges[1:]
    ) / 2.0

    bin_accuracies = np.full(
        n_bins,
        np.nan
    )

    bin_confidences = np.full(
        n_bins,
        np.nan
    )

    bin_counts = np.zeros(
        n_bins,
        dtype=np.int64
    )

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        count = int(mask.sum())
        bin_counts[bin_index] = count

        if count > 0:
            bin_accuracies[bin_index] = float(
                correctness[mask].mean()
            )

            bin_confidences[bin_index] = float(
                confidences[mask].mean()
            )

    valid_bins = bin_counts > 0

    ece = float(
        np.sum(
            (
                bin_counts[valid_bins]
                / len(y_true)
            )
            * np.abs(
                bin_accuracies[valid_bins]
                - bin_confidences[valid_bins]
            )
        )
    )

    accuracy = float(
        correctness.mean()
    )

    mean_confidence = float(
        confidences.mean()
    )

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"reliability_diagram_{model_name}.png"
    )

    figure = plt.figure(
        figsize=(8, 9)
    )

    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[3, 1],
        hspace=0.3
    )

    reliability_axis = figure.add_subplot(
        grid[0]
    )

    histogram_axis = figure.add_subplot(
        grid[1]
    )

    reliability_axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    reliability_axis.bar(
        bin_centers[valid_bins],
        bin_accuracies[valid_bins],
        width=1.0 / n_bins,
        alpha=0.65,
        edgecolor="black",
        label="Observed accuracy"
    )

    reliability_axis.plot(
        bin_confidences[valid_bins],
        bin_accuracies[valid_bins],
        marker="o",
        linewidth=2,
        label="Model calibration"
    )

    reliability_axis.set_xlim(
        0.0,
        1.0
    )

    reliability_axis.set_ylim(
        0.0,
        1.0
    )

    reliability_axis.set_xlabel(
        "Mean predicted confidence"
    )

    reliability_axis.set_ylabel(
        "Observed accuracy"
    )

    reliability_axis.set_title(
        f"Reliability Diagram - {model_name}\n"
        f"Accuracy={accuracy:.4f}, "
        f"Mean Confidence={mean_confidence:.4f}, "
        f"ECE={ece:.4f}"
    )

    reliability_axis.legend(
        loc="upper left"
    )

    reliability_axis.grid(
        alpha=0.25
    )

    histogram_axis.hist(
        confidences,
        bins=bin_edges,
        edgecolor="black"
    )

    histogram_axis.set_xlim(
        0.0,
        1.0
    )

    histogram_axis.set_xlabel(
        "Prediction confidence"
    )

    histogram_axis.set_ylabel(
        "Sample count"
    )

    histogram_axis.set_title(
        "Confidence Distribution"
    )

    histogram_axis.grid(
        axis="y",
        alpha=0.25
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] {output_path}"
    )

    print(
        f"        Accuracy: {accuracy:.4f}"
    )

    print(
        f"        Mean confidence: {mean_confidence:.4f}"
    )

    print(
        f"        ECE: {ece:.4f}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    TEST_DIR = (
        "data/NEU-DET/"
        "split_1799/test/images"
    )

    N_BINS = 15

    CHECKPOINT_PATHS = [
        "outputs/models/best_model_1800.pt",
        "outputs/models/resnet50_best_1800.pt",
        "outputs/models/best_model_densenet121.pt",
        "outputs/models/best_model_mobilenetv3.pt",
    ]

    for checkpoint_path in CHECKPOINT_PATHS:
        checkpoint = Path(
            checkpoint_path
        )

        if not checkpoint.exists():
            print(
                f"[WARNING] Checkpoint not found: "
                f"{checkpoint_path}"
            )

            continue

        print(
            f"\n[INFO] Processing: "
            f"{checkpoint_path}"
        )

        (
            architecture,
            probabilities,
            labels
        ) = collect_predictions(
            checkpoint_path=checkpoint_path,
            test_dir=TEST_DIR
        )

        plot_reliability_diagram(
            probabilities=probabilities,
            y_true=labels,
            model_name=architecture,
            n_bins=N_BINS
        )