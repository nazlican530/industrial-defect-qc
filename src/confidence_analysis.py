from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


MODEL_PATH = Path(
    "outputs/models/best_model_mobilenetv3.pt"
)

TEST_DIR = Path(
    "data/NEU-DET/split_1799/test/images"
)

OUTPUT_PATH = Path(
    "outputs/confidence_analysis_mobilenetv3_test.csv"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


device = get_device()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = models.mobilenet_v3_large(
        weights=None
    )

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        in_features,
        num_classes,
    )

    state_dict = checkpoint["model_state"]

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    mean = tuple(
        checkpoint.get(
            "imagenet_mean",
            (0.485, 0.456, 0.406),
        )
    )

    std = tuple(
        checkpoint.get(
            "imagenet_std",
            (0.229, 0.224, 0.225),
        )
    )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return model, transform, class_to_idx


def confidence_group(
    confidence: float,
) -> str:
    if confidence >= 0.90:
        return ">= 0.90"

    if confidence >= 0.60:
        return "0.60-0.89"

    return "< 0.60"


def decision_rule(
    confidence_range: str,
) -> str:
    if confidence_range == ">= 0.90":
        return (
            "Severity-based "
            "(ACCEPT / REWORK / REJECT)"
        )

    if confidence_range == "0.60-0.89":
        return "REWORK"

    return "HUMAN REVIEW"


def main() -> None:
    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Test directory:", TEST_DIR)

    model, transform, checkpoint_class_to_idx = (
        load_model()
    )

    dataset = datasets.ImageFolder(
        TEST_DIR
    )

    if (
        checkpoint_class_to_idx
        != dataset.class_to_idx
    ):
        raise ValueError(
            "Checkpoint class order and test "
            "dataset class order do not match.\n"
            f"Checkpoint: {checkpoint_class_to_idx}\n"
            f"Dataset: {dataset.class_to_idx}"
        )

    results = []

    with torch.no_grad():
        for image_path, true_label in dataset.samples:
            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = (
                transform(image)
                .unsqueeze(0)
                .to(device)
            )

            logits = model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predicted_label = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )

            confidence_value = float(
                confidence.item()
            )

            predicted_value = int(
                predicted_label.item()
            )

            correct = int(
                predicted_value == true_label
            )

            results.append(
                {
                    "image": str(image_path),
                    "true_label": true_label,
                    "predicted_label": (
                        predicted_value
                    ),
                    "confidence": (
                        confidence_value
                    ),
                    "correct": correct,
                    "confidence_range": (
                        confidence_group(
                            confidence_value
                        )
                    ),
                }
            )

    df = pd.DataFrame(results)

    range_order = [
        "< 0.60",
        "0.60-0.89",
        ">= 0.90",
    ]

    summary = (
        df.groupby(
            "confidence_range",
            observed=False,
        )
        .agg(
            sample_count=(
                "confidence",
                "count",
            ),
            average_confidence=(
                "confidence",
                "mean",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
        )
        .reindex(range_order)
        .fillna(0)
        .reset_index()
    )

    summary["percentage"] = (
        summary["sample_count"]
        / len(df)
        * 100
    )

    summary["decision_rule"] = (
        summary["confidence_range"]
        .apply(decision_rule)
    )

    summary["average_confidence"] = (
        summary["average_confidence"]
        * 100
    ).round(2)

    summary["accuracy"] = (
        summary["accuracy"]
        * 100
    ).round(2)

    summary["percentage"] = (
        summary["percentage"]
        .round(2)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nConfidence analysis")
    print(
        f"Total test samples: {len(df)}\n"
    )

    columns = [
        "confidence_range",
        "sample_count",
        "percentage",
        "average_confidence",
        "accuracy",
        "decision_rule",
    ]

    print(
        summary[columns].to_string(
            index=False
        )
    )

    print(
        "\nSample count check:",
        int(summary["sample_count"].sum()),
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()