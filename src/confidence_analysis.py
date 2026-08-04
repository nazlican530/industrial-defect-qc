from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "outputs/models/best_model_mobilenetv3.pt"
)

TEST_DIR = Path(
    "data/NEU-DET/split_1799/test/images"
)

# Existing confidence-range table
OUTPUT_PATH = Path(
    "outputs/confidence_analysis_mobilenetv3_test.csv"
)

# New combined confidence + margin analysis
COMBINED_OUTPUT_PATH = Path(
    "outputs/combined_review_coverage_mobilenetv3_test.csv"
)

# Per-image detailed results
DETAIL_OUTPUT_PATH = Path(
    "outputs/confidence_margin_details_mobilenetv3_test.csv"
)


# ============================================================
# FINAL DECISION POLICY
# ============================================================

CONFIDENCE_THRESHOLD = 0.90
TOP2_MARGIN_THRESHOLD = 0.10


# ============================================================
# DEVICE
# ============================================================

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


# ============================================================
# MODEL LOADING
# ============================================================

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


# ============================================================
# CONFIDENCE GROUP
# ============================================================

def confidence_group(
    confidence: float,
) -> str:
    if confidence >= 0.90:
        return ">= 0.90"

    if confidence >= 0.60:
        return "0.60-0.89"

    return "< 0.60"


# ============================================================
# CONFIDENCE-ONLY TABLE DECISION LABEL
# ============================================================

def decision_rule(
    confidence_range: str,
) -> str:
    if confidence_range == ">= 0.90":
        return (
            "Subject to Top-2 margin check; "
            "otherwise severity-based"
        )

    # FINAL POLICY:
    # everything below 0.90 goes to HUMAN REVIEW
    return "HUMAN REVIEW"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("COMBINED CONFIDENCE + TOP-2 MARGIN ANALYSIS")
    print("=" * 70)

    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Test directory:", TEST_DIR)

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Top-2 margin threshold: "
        f"{TOP2_MARGIN_THRESHOLD:.2f}"
    )

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

    idx_to_class = {
        index: class_name
        for class_name, index
        in dataset.class_to_idx.items()
    }

    results = []

    # ========================================================
    # INFERENCE
    # ========================================================

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

            # Get highest and second-highest probabilities
            top2_probabilities, top2_indices = (
                torch.topk(
                    probabilities,
                    k=2,
                    dim=1,
                )
            )

            confidence = float(
                top2_probabilities[0, 0].item()
            )

            second_confidence = float(
                top2_probabilities[0, 1].item()
            )

            top2_margin = (
                confidence
                - second_confidence
            )

            predicted_label = int(
                top2_indices[0, 0].item()
            )

            second_label = int(
                top2_indices[0, 1].item()
            )

            correct = int(
                predicted_label == true_label
            )

            # ------------------------------------------------
            # FINAL HUMAN REVIEW RULE
            # ------------------------------------------------

            low_confidence = (
                confidence
                < CONFIDENCE_THRESHOLD
            )

            low_margin = (
                top2_margin
                < TOP2_MARGIN_THRESHOLD
            )

            human_review = (
                low_confidence
                or low_margin
            )

            # Determine why the sample was reviewed
            if low_confidence and low_margin:
                review_reason = (
                    "Low confidence + low margin"
                )

            elif low_confidence:
                review_reason = (
                    "Low confidence only"
                )

            elif low_margin:
                review_reason = (
                    "Low margin only"
                )

            else:
                review_reason = (
                    "Automatic decision eligible"
                )

            if human_review:
                final_pathway = "HUMAN REVIEW"
            else:
                final_pathway = (
                    "AUTOMATIC SEVERITY-BASED DECISION"
                )

            results.append(
                {
                    "image": str(image_path),
                    "true_label": true_label,
                    "true_class": (
                        idx_to_class[true_label]
                    ),
                    "predicted_label": (
                        predicted_label
                    ),
                    "predicted_class": (
                        idx_to_class[
                            predicted_label
                        ]
                    ),
                    "second_label": (
                        second_label
                    ),
                    "second_class": (
                        idx_to_class[
                            second_label
                        ]
                    ),
                    "confidence": confidence,
                    "second_confidence": (
                        second_confidence
                    ),
                    "top2_margin": (
                        top2_margin
                    ),
                    "correct": correct,
                    "confidence_range": (
                        confidence_group(
                            confidence
                        )
                    ),
                    "low_confidence": (
                        low_confidence
                    ),
                    "low_margin": (
                        low_margin
                    ),
                    "human_review": (
                        human_review
                    ),
                    "review_reason": (
                        review_reason
                    ),
                    "final_pathway": (
                        final_pathway
                    ),
                }
            )

    df = pd.DataFrame(results)

    total_samples = len(df)

    # ========================================================
    # ORIGINAL CONFIDENCE-RANGE ANALYSIS
    # ========================================================

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
        / total_samples
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

    # ========================================================
    # COMBINED COVERAGE ANALYSIS
    # ========================================================

    confidence_review_count = int(
        df["low_confidence"].sum()
    )

    margin_review_count = int(
        df["low_margin"].sum()
    )

    combined_review_count = int(
        df["human_review"].sum()
    )

    automatic_count = (
        total_samples
        - combined_review_count
    )

    # Samples satisfying both conditions
    both_count = int(
        (
            df["low_confidence"]
            & df["low_margin"]
        ).sum()
    )

    # Samples reviewed ONLY because of confidence
    confidence_only_count = int(
        (
            df["low_confidence"]
            & ~df["low_margin"]
        ).sum()
    )

    # Samples reviewed ONLY because of margin
    margin_only_count = int(
        (
            ~df["low_confidence"]
            & df["low_margin"]
        ).sum()
    )

    review_coverage = (
        combined_review_count
        / total_samples
        * 100
    )

    automatic_coverage = (
        automatic_count
        / total_samples
        * 100
    )

    # Accuracy among automatically processed samples
    automatic_df = df[
        ~df["human_review"]
    ]

    if len(automatic_df) > 0:
        automatic_accuracy = (
            automatic_df["correct"].mean()
            * 100
        )
    else:
        automatic_accuracy = 0.0

    # Accuracy among reviewed samples
    review_df = df[
        df["human_review"]
    ]

    if len(review_df) > 0:
        review_accuracy = (
            review_df["correct"].mean()
            * 100
        )
    else:
        review_accuracy = 0.0

    combined_summary = pd.DataFrame(
        [
            {
                "metric": "Total test samples",
                "value": total_samples,
            },
            {
                "metric": (
                    "Confidence < 0.90"
                ),
                "value": (
                    confidence_review_count
                ),
            },
            {
                "metric": (
                    "Top-2 margin < 0.10"
                ),
                "value": (
                    margin_review_count
                ),
            },
            {
                "metric": (
                    "Both low confidence "
                    "and low margin"
                ),
                "value": both_count,
            },
            {
                "metric": (
                    "Confidence-only review"
                ),
                "value": (
                    confidence_only_count
                ),
            },
            {
                "metric": (
                    "Margin-only review"
                ),
                "value": margin_only_count,
            },
            {
                "metric": (
                    "Combined HUMAN REVIEW"
                ),
                "value": (
                    combined_review_count
                ),
            },
            {
                "metric": (
                    "Automatic severity-based "
                    "decision"
                ),
                "value": automatic_count,
            },
            {
                "metric": (
                    "Review coverage (%)"
                ),
                "value": round(
                    review_coverage,
                    2,
                ),
            },
            {
                "metric": (
                    "Automatic coverage (%)"
                ),
                "value": round(
                    automatic_coverage,
                    2,
                ),
            },
            {
                "metric": (
                    "Accuracy among reviewed "
                    "samples (%)"
                ),
                "value": round(
                    review_accuracy,
                    2,
                ),
            },
            {
                "metric": (
                    "Accuracy among automatic "
                    "samples (%)"
                ),
                "value": round(
                    automatic_accuracy,
                    2,
                ),
            },
        ]
    )

    # ========================================================
    # SAVE FILES
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    combined_summary.to_csv(
        COMBINED_OUTPUT_PATH,
        index=False,
    )

    df.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # PRINT CONFIDENCE TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONFIDENCE RANGE ANALYSIS")
    print("=" * 70)

    print(
        f"Total test samples: "
        f"{total_samples}\n"
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
        int(
            summary[
                "sample_count"
            ].sum()
        ),
    )

    # ========================================================
    # PRINT COMBINED ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FINAL 0.90 / 0.10 HUMAN REVIEW POLICY"
    )
    print("=" * 70)

    print(
        f"Total test samples: "
        f"{total_samples}"
    )

    print(
        f"Confidence < 0.90: "
        f"{confidence_review_count}"
    )

    print(
        f"Top-2 confidence margin < 0.10: "
        f"{margin_review_count}"
    )

    print(
        f"Both conditions: "
        f"{both_count}"
    )

    print(
        f"Confidence-only HUMAN REVIEW: "
        f"{confidence_only_count}"
    )

    print(
        f"Margin-only HUMAN REVIEW: "
        f"{margin_only_count}"
    )

    print(
        f"\nCombined HUMAN REVIEW: "
        f"{combined_review_count}"
        f"/{total_samples}"
    )

    print(
        f"Review coverage: "
        f"{review_coverage:.2f}%"
    )

    print(
        f"\nAutomatic severity-based "
        f"decision: "
        f"{automatic_count}"
        f"/{total_samples}"
    )

    print(
        f"Automatic coverage: "
        f"{automatic_coverage:.2f}%"
    )

    print(
        f"\nAccuracy among HUMAN REVIEW "
        f"samples: "
        f"{review_accuracy:.2f}%"
    )

    print(
        f"Accuracy among automatic "
        f"samples: "
        f"{automatic_accuracy:.2f}%"
    )

    print("\n")
    print("=" * 70)
    print("SAVED FILES")
    print("=" * 70)

    print(
        "Confidence summary:",
        OUTPUT_PATH,
    )

    print(
        "Combined coverage:",
        COMBINED_OUTPUT_PATH,
    )

    print(
        "Per-image details:",
        DETAIL_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()