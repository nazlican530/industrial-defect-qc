import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# ==========================================================
# CHECKPOINT HELPERS
# ==========================================================

def extract_state_dict(ckpt):
    """
    Checkpoint içinden model ağırlıklarını çıkarır.
    Eğer checkpoint doğrudan state_dict ise olduğu gibi döndürür.
    """
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]

    return ckpt


def infer_arch_from_state_dict(state_dict):
    """
    Checkpoint içinde arch bilgisi yoksa ağırlık anahtarlarına
    bakarak model mimarisini tahmin eder.
    """
    keys = list(state_dict.keys())

    # DenseNet kontrolü önce yapılmalı.
    # DenseNet'te de classifier anahtarları bulunduğu için
    # yalnızca "classifier" kontrolü yapmak yanlış sonuç verir.
    if any(k.startswith("features.denseblock") for k in keys):
        return "densenet121"

    # ResNet50 bottleneck bloklarında conv3 bulunur.
    if any(".conv3.weight" in k for k in keys):
        return "resnet50"

    # MobileNetV3-Large'ın son özellik bloğu features.16'dır.
    if any(k.startswith("features.16") for k in keys):
        return "mobilenet_v3_large"

    # EfficientNet checkpoint anahtarları classifier ile birlikte
    # features.0, features.1 gibi katmanları içerir.
    if (
        any(k.startswith("features.") for k in keys)
        and any(k.startswith("classifier.") for k in keys)
    ):
        return "efficientnet_b0"

    raise ValueError(
        "Could not infer architecture from checkpoint keys."
    )


# ==========================================================
# DEVICE
# ==========================================================

def get_device():
    """
    CUDA, Apple MPS veya CPU cihazını seçer.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(
    ckpt_path,
    device,
    dataset_class_to_idx
):
    """
    Checkpoint dosyasını yükler, model mimarisini oluşturur,
    ağırlıkları modele aktarır ve modeli değerlendirme moduna alır.
    """
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(ckpt)

    if isinstance(ckpt, dict) and "arch" in ckpt:
        arch = ckpt["arch"]
    else:
        arch = infer_arch_from_state_dict(state_dict)

    if isinstance(ckpt, dict) and "class_to_idx" in ckpt:
        class_to_idx = ckpt["class_to_idx"]
    else:
        class_to_idx = dataset_class_to_idx

    num_classes = len(class_to_idx)

    print(
        f"\n[INFO] Loaded model: "
        f"{Path(ckpt_path).name} -> {arch}"
    )
    print("[INFO] class_to_idx:", class_to_idx)

    # ------------------------------------------------------
    # EfficientNet-B0
    # ------------------------------------------------------
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=None
        )

        in_features = (
            model.classifier[1].in_features
        )

        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(
                in_features,
                num_classes
            )
        )

        try:
            model.load_state_dict(
                state_dict,
                strict=True
            )

        except RuntimeError:
            # Bazı eski checkpoint'lerde classifier anahtarı
            # classifier.1.1 şeklinde kaydedilmiş olabilir.
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

    # ------------------------------------------------------
    # ResNet50
    # ------------------------------------------------------
    elif arch == "resnet50":
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

    # ------------------------------------------------------
    # DenseNet121
    # ------------------------------------------------------
    elif arch == "densenet121":
        model = models.densenet121(
            weights=None
        )

        in_features = (
            model.classifier.in_features
        )

        # Yeni DenseNet eğitim dosyası yalnızca Linear classifier
        # kullanır.
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
            # Eski checkpoint Dropout + Linear biçiminde
            # kaydedildiyse bunu da destekler.
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

    # ------------------------------------------------------
    # MobileNetV3-Large
    # ------------------------------------------------------
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=None
        )

        in_features = (
            model.classifier[3].in_features
        )

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
            f"Unsupported architecture: {arch}. "
            "Supported models: efficientnet_b0, "
            "resnet50, densenet121 and mobilenet_v3_large."
        )

    model = model.to(device)
    model.eval()

    if isinstance(ckpt, dict):
        mean = tuple(
            ckpt.get(
                "imagenet_mean",
                (0.485, 0.456, 0.406)
            )
        )

        std = tuple(
            ckpt.get(
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
        class_to_idx,
        mean,
        std,
        arch
    )


# ==========================================================
# HELPER
# ==========================================================

def format_topk(
    prob_vector,
    class_names,
    k=3
):
    """
    En yüksek olasılığa sahip ilk k sınıfı metin olarak döndürür.
    """
    top_probs, top_idxs = torch.topk(
        prob_vector,
        k=min(k, len(class_names))
    )

    lines = []

    for rank, (probability, index) in enumerate(
        zip(
            top_probs.tolist(),
            top_idxs.tolist()
        ),
        start=1
    ):
        lines.append(
            f"    Top-{rank}: "
            f"{class_names[index]} "
            f"({probability:.4f})"
        )

    return "\n".join(lines)


# ==========================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================================

def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    n_bootstrap=10000,
    confidence_level=0.95,
    seed=42
):
    """
    Independent test predictions üzerinden non-parametric percentile
    bootstrap confidence intervals hesaplar.

    Hesaplanan metrikler:
    - Accuracy
    - Macro precision
    - Macro recall
    - Macro F1-score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "Bootstrap confidence intervals cannot be calculated "
            "for empty arrays."
        )

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    bootstrap_scores = {
        "accuracy": np.empty(n_bootstrap, dtype=np.float64),
        "precision": np.empty(n_bootstrap, dtype=np.float64),
        "recall": np.empty(n_bootstrap, dtype=np.float64),
        "f1": np.empty(n_bootstrap, dtype=np.float64)
    }

    for bootstrap_index in range(n_bootstrap):
        sample_indices = rng.integers(
            0,
            n_samples,
            size=n_samples
        )

        y_true_bootstrap = y_true[sample_indices]
        y_pred_bootstrap = y_pred[sample_indices]

        bootstrap_scores["accuracy"][bootstrap_index] = float(
            np.mean(
                y_true_bootstrap == y_pred_bootstrap
            )
        )

        bootstrap_scores["precision"][bootstrap_index] = (
            precision_score(
                y_true_bootstrap,
                y_pred_bootstrap,
                average="macro",
                zero_division=0
            )
        )

        bootstrap_scores["recall"][bootstrap_index] = (
            recall_score(
                y_true_bootstrap,
                y_pred_bootstrap,
                average="macro",
                zero_division=0
            )
        )

        bootstrap_scores["f1"][bootstrap_index] = (
            f1_score(
                y_true_bootstrap,
                y_pred_bootstrap,
                average="macro",
                zero_division=0
            )
        )

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    point_estimates = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        )
    }

    results = {}

    for metric_name, score_distribution in bootstrap_scores.items():
        results[metric_name] = {
            "score": point_estimates[metric_name],
            "lower": float(
                np.percentile(
                    score_distribution,
                    lower_percentile
                )
            ),
            "upper": float(
                np.percentile(
                    score_distribution,
                    upper_percentile
                )
            )
        }

    return results


def print_bootstrap_results(
    model_name,
    bootstrap_results,
    n_bootstrap,
    confidence_level
):
    """Bootstrap sonuçlarını terminalde gösterir."""
    confidence_percent = int(
        round(confidence_level * 100)
    )

    print(
        f"\n===== {model_name} BOOTSTRAP RESULTS ====="
    )
    print(
        f"Bootstrap iterations: {n_bootstrap}"
    )
    print(
        f"Confidence level: {confidence_percent}%"
    )

    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1"
    ):
        values = bootstrap_results[metric_name]

        print(
            f"{metric_name.capitalize():<10}: "
            f"{values['score']:.4f} "
            f"[{confidence_percent}% CI: "
            f"{values['lower']:.4f}-"
            f"{values['upper']:.4f}]"
        )



# ==========================================================
# CALIBRATION METRICS
# ==========================================================

def expected_calibration_error(probabilities, y_true, n_bins=15):
    """Top-label ECE with equal-width confidence bins."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(correctness[mask].mean())
        bin_conf = float(confidences[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (count / len(y_true)) * gap
        bin_rows.append((lower, upper, count, bin_acc, bin_conf, gap))
    return float(ece), bin_rows

def multiclass_brier_score(probabilities, y_true, num_classes):
    """Mean multiclass Brier score: mean sum_k (p_k-y_k)^2."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

def calibration_metrics(probabilities, y_true, n_bins=15):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    ece, bin_rows = expected_calibration_error(probabilities, y_true, n_bins)
    return {
        "ece": ece,
        "brier": multiclass_brier_score(probabilities, y_true, probabilities.shape[1]),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "n_bins": n_bins,
        "bin_rows": bin_rows,
    }

# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(
    ckpt_path,
    save_confusion=True,
    print_misclassified=True,
    max_print=50,
    n_bootstrap=10000,
    confidence_level=0.95,
    bootstrap_seed=42
):
    """
    Bir checkpoint'i test dataseti üzerinde değerlendirir.
    Accuracy, macro precision, macro recall ve macro F1 hesaplar.
    Ayrıca bütün metrikler için bootstrap confidence intervals üretir.
    """
    device = get_device()

    print("[INFO] Device:", device)

    test_dir = (
        "data/NEU-DET/"
        "split_1799/test/images"
    )

    base_dataset = datasets.ImageFolder(
        test_dir
    )

    print(
        "\n[INFO] Dataset classes:",
        base_dataset.classes
    )

    (
        model,
        class_to_idx,
        mean,
        std,
        arch
    ) = load_model(
        ckpt_path=ckpt_path,
        device=device,
        dataset_class_to_idx=(
            base_dataset.class_to_idx
        )
    )

    # Checkpoint ile test dataseti sınıf sıralaması aynı olmalı.
    if class_to_idx != base_dataset.class_to_idx:
        raise ValueError(
            "Checkpoint class_to_idx ile test dataset "
            "class_to_idx eşleşmiyor.\n"
            f"Checkpoint: {class_to_idx}\n"
            f"Dataset: {base_dataset.class_to_idx}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean,
                std
            )
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
        pin_memory=(
            device.type == "cuda"
        )
    )

    paths = [
        sample_path
        for sample_path, _
        in dataset.samples
    ]

    y_true = []
    y_pred = []
    all_probabilities = []

    printed = 0
    global_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_probabilities.append(probabilities.cpu().numpy())

            y_pred.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            y_true.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            if print_misclassified:
                for index in range(
                    images.size(0)
                ):
                    if (
                        predictions[index]
                        != labels[index]
                        and printed < max_print
                    ):
                        true_index = (
                            labels[index].item()
                        )

                        pred_index = (
                            predictions[index].item()
                        )

                        print(
                            "\n[WRONG PREDICTION]"
                        )

                        print(
                            "Image:",
                            paths[
                                global_index
                                + index
                            ]
                        )

                        print(
                            "True:",
                            dataset.classes[
                                true_index
                            ]
                        )

                        print(
                            "Pred:",
                            dataset.classes[
                                pred_index
                            ]
                        )

                        print(
                            "Confidence:",
                            f"{float(probabilities[index, pred_index]):.4f}"
                        )

                        print(
                            format_topk(
                                probabilities[
                                    index
                                ].cpu(),
                                dataset.classes
                            )
                        )

                        printed += 1

            global_index += (
                images.size(0)
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    correct = int(
        np.sum(
            y_true == y_pred
        )
    )

    total = int(
        len(y_true)
    )

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print(
        f"\n===== {arch} RESULTS ====="
    )

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    calibration = calibration_metrics(
        probabilities=all_probabilities,
        y_true=y_true,
        n_bins=15
    )

    print(f"Mean Confidence: {calibration['mean_confidence']:.4f}")
    print(f"ECE (15 bins): {calibration['ece']:.4f}")
    print(f"Brier Score: {calibration['brier']:.4f}")

    # ------------------------------------------------------
    # Bootstrap Confidence Intervals
    # ------------------------------------------------------
    bootstrap_results = bootstrap_confidence_intervals(
        y_true=y_true,
        y_pred=y_pred,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=bootstrap_seed
    )

    print_bootstrap_results(
        model_name=arch,
        bootstrap_results=bootstrap_results,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level
    )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------
    if save_confusion:
        figure_output_dir = Path(
            "outputs/figures"
        )

        figure_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        matrix = confusion_matrix(
            y_true,
            y_pred
        )

        plt.figure(
            figsize=(9, 7)
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=(
                dataset.classes
            ),
            yticklabels=(
                dataset.classes
            )
        )

        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.title(
            f"Confusion Matrix - {arch}"
        )

        plt.tight_layout()

        confusion_path = (
            figure_output_dir
            / f"confusion_matrix_{arch}.png"
        )

        plt.savefig(
            confusion_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Confusion matrix saved -> "
            f"{confusion_path}"
        )

    return {
        "model": arch,
        "checkpoint": str(
            ckpt_path
        ),
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bootstrap": bootstrap_results,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "bootstrap_seed": bootstrap_seed,
        "mean_confidence": calibration["mean_confidence"],
        "ece": calibration["ece"],
        "brier": calibration["brier"],
        "calibration_bins": calibration["bin_rows"],
        "n_calibration_bins": calibration["n_bins"]
    }


# ==========================================================
# MODEL COMPARISON
# ==========================================================

def print_model_comparison(results):
    """
    Modellerin sonuçlarını terminalde tablo biçiminde gösterir.
    """
    print(
        "\n========== MODEL COMPARISON =========="
    )

    print(
        f"{'Model':<20} "
        f"{'Correct':<12} "
        f"{'Accuracy':<10} "
        f"{'Precision':<10} "
        f"{'Recall':<10} "
        f"{'F1-score':<10}"
    )

    print("-" * 80)

    for result in results:
        correct_text = (
            f"{result['correct']}/"
            f"{result['total']}"
        )

        print(
            f"{result['model']:<20} "
            f"{correct_text:<12} "
            f"{result['accuracy']:<10.4f} "
            f"{result['precision']:<10.4f} "
            f"{result['recall']:<10.4f} "
            f"{result['f1']:<10.4f}"
        )


def print_bootstrap_comparison(results):
    """
    Accuracy ve macro F1 bootstrap confidence intervals sonuçlarını
    modeller arasında karşılaştırmalı olarak gösterir.
    """
    confidence_percent = int(
        round(results[0]["confidence_level"] * 100)
    )

    print(
        f"\n========== {confidence_percent}% BOOTSTRAP CI COMPARISON =========="
    )

    print(
        f"{'Model':<20} "
        f"{'Accuracy (CI)':<30} "
        f"{'Macro F1 (CI)':<30}"
    )
    print("-" * 84)

    for result in results:
        accuracy_ci = result["bootstrap"]["accuracy"]
        f1_ci = result["bootstrap"]["f1"]

        accuracy_text = (
            f"{accuracy_ci['score']:.4f} "
            f"[{accuracy_ci['lower']:.4f}-"
            f"{accuracy_ci['upper']:.4f}]"
        )
        f1_text = (
            f"{f1_ci['score']:.4f} "
            f"[{f1_ci['lower']:.4f}-"
            f"{f1_ci['upper']:.4f}]"
        )

        print(
            f"{result['model']:<20} "
            f"{accuracy_text:<30} "
            f"{f1_text:<30}"
        )


def save_model_comparison(results):
    """
    Model karşılaştırmasını text dosyasına kaydeder.
    En iyi model macro F1 skoruna göre seçilir.
    """
    output_dir = Path(
        "outputs/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    text_path = (
        output_dir
        / "model_comparison.txt"
    )

    best = max(
        results,
        key=lambda item: item["f1"]
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "MODEL COMPARISON RESULTS\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            f"{'Model':<20} "
            f"{'Correct':<12} "
            f"{'Accuracy':<10} "
            f"{'Precision':<10} "
            f"{'Recall':<10} "
            f"{'F1-score':<10}\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        for result in results:
            correct_text = (
                f"{result['correct']}/"
                f"{result['total']}"
            )

            file.write(
                f"{result['model']:<20} "
                f"{correct_text:<12} "
                f"{result['accuracy']:<10.4f} "
                f"{result['precision']:<10.4f} "
                f"{result['recall']:<10.4f} "
                f"{result['f1']:<10.4f}\n"
            )

        file.write("\n")

        confidence_percent = int(
            round(results[0]["confidence_level"] * 100)
        )

        file.write(
            f"{confidence_percent}% BOOTSTRAP CONFIDENCE INTERVALS\n"
        )
        file.write(
            "=" * 80 + "\n"
        )

        for result in results:
            accuracy_ci = result["bootstrap"]["accuracy"]
            f1_ci = result["bootstrap"]["f1"]

            file.write(
                f"{result['model']}: "
                f"Accuracy {accuracy_ci['score']:.4f} "
                f"[{accuracy_ci['lower']:.4f}-"
                f"{accuracy_ci['upper']:.4f}], "
                f"Macro F1 {f1_ci['score']:.4f} "
                f"[{f1_ci['lower']:.4f}-"
                f"{f1_ci['upper']:.4f}]\n"
            )

        file.write("\n")

        file.write(
            "BEST MODEL SELECTION\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            f"Best Model: "
            f"{best['model']}\n"
        )

        file.write(
            f"Checkpoint: "
            f"{best['checkpoint']}\n"
        )

        file.write(
            f"Correct: "
            f"{best['correct']}/"
            f"{best['total']}\n"
        )

        file.write(
            f"Accuracy: "
            f"{best['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision: "
            f"{best['precision']:.4f}\n"
        )

        file.write(
            f"Recall: "
            f"{best['recall']:.4f}\n"
        )

        file.write(
            f"F1-score: "
            f"{best['f1']:.4f}\n"
        )

    print(
        "\nModel comparison saved -> "
        f"{text_path}"
    )


def save_bootstrap_results(results):
    """
    Bütün modellerin bootstrap confidence interval sonuçlarını
    TXT ve CSV dosyalarına kaydeder.
    """
    output_dir = Path(
        "outputs/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    text_path = (
        output_dir
        / "bootstrap_confidence_intervals.txt"
    )

    csv_path = (
        output_dir
        / "bootstrap_confidence_intervals.csv"
    )

    first_result = results[0]
    confidence_level = first_result["confidence_level"]
    confidence_percent = int(
        round(confidence_level * 100)
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f"{confidence_percent}% BOOTSTRAP CONFIDENCE INTERVALS\n"
        )
        file.write(
            "Evaluation set: Independent test set\n"
        )
        file.write(
            f"Bootstrap iterations: {first_result['n_bootstrap']}\n"
        )
        file.write(
            f"Bootstrap seed: {first_result['bootstrap_seed']}\n"
        )
        file.write(
            "Method: Non-parametric percentile bootstrap\n\n"
        )

        for result in results:
            file.write(
                f"{result['model']}\n"
            )
            file.write(
                "-" * 70 + "\n"
            )

            for metric_name in (
                "accuracy",
                "precision",
                "recall",
                "f1"
            ):
                values = result["bootstrap"][metric_name]

                file.write(
                    f"{metric_name.capitalize():<10}: "
                    f"{values['score']:.4f} "
                    f"[{confidence_percent}% CI: "
                    f"{values['lower']:.4f}-"
                    f"{values['upper']:.4f}]\n"
                )

            file.write("\n")

    with open(
        csv_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "model,metric,score,ci_lower,ci_upper,"
            "confidence_level,n_bootstrap,bootstrap_seed\n"
        )

        for result in results:
            for metric_name in (
                "accuracy",
                "precision",
                "recall",
                "f1"
            ):
                values = result["bootstrap"][metric_name]

                file.write(
                    f"{result['model']},"
                    f"{metric_name},"
                    f"{values['score']:.6f},"
                    f"{values['lower']:.6f},"
                    f"{values['upper']:.6f},"
                    f"{result['confidence_level']:.2f},"
                    f"{result['n_bootstrap']},"
                    f"{result['bootstrap_seed']}\n"
                )

    print(
        "Bootstrap results saved -> "
        f"{text_path}"
    )
    print(
        "Bootstrap CSV saved -> "
        f"{csv_path}"
    )


def print_calibration_comparison(results):
    print("\n========== CALIBRATION COMPARISON ==========")
    print(f"{'Model':<20} {'Accuracy':<10} {'Mean Conf.':<12} {'ECE':<10} {'Brier':<10}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<20} {r['accuracy']:<10.4f} {r['mean_confidence']:<12.4f} {r['ece']:<10.4f} {r['brier']:<10.4f}")

def save_calibration_results(results):
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "calibration_metrics.txt"
    csv_path = output_dir / "calibration_metrics.csv"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("CALIBRATION METRICS ON THE INDEPENDENT TEST SET\n")
        f.write("ECE: top-label ECE, 15 equal-width bins\n")
        f.write("Brier: multiclass mean sum of squared probability errors\n\n")
        for r in results:
            f.write(f"{r['model']}: Accuracy={r['accuracy']:.4f}, MeanConfidence={r['mean_confidence']:.4f}, ECE={r['ece']:.4f}, Brier={r['brier']:.4f}\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,accuracy,mean_confidence,ece,brier,n_bins\n")
        for r in results:
            f.write(f"{r['model']},{r['accuracy']:.6f},{r['mean_confidence']:.6f},{r['ece']:.6f},{r['brier']:.6f},{r['n_calibration_bins']}\n")
    print(f"Calibration results saved -> {txt_path}")
    print(f"Calibration CSV saved -> {csv_path}")


def print_best_model(results):
    """
    En iyi modeli terminalde gösterir.
    """
    best = max(
        results,
        key=lambda item: item["f1"]
    )

    print(
        "\n========== BEST MODEL =========="
    )

    print(
        f"Best Model: "
        f"{best['model']}"
    )

    print(
        f"Checkpoint: "
        f"{best['checkpoint']}"
    )

    print(
        f"Correct: "
        f"{best['correct']}/"
        f"{best['total']}"
    )

    print(
        f"Accuracy: "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1-score: "
        f"{best['f1']:.4f}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    N_BOOTSTRAP = 10000
    CONFIDENCE_LEVEL = 0.95
    BOOTSTRAP_SEED = 42

    checkpoint_paths = [
        "outputs/models/best_model_1800.pt",
        "outputs/models/resnet50_best_1800.pt",
        "outputs/models/best_model_densenet121.pt",
        "outputs/models/best_model_mobilenetv3.pt",
    ]

    results = []

    for checkpoint_path in checkpoint_paths:
        if Path(
            checkpoint_path
        ).exists():
            result = evaluate(
                ckpt_path=(
                    checkpoint_path
                ),
                save_confusion=True,
                print_misclassified=False,
                n_bootstrap=N_BOOTSTRAP,
                confidence_level=CONFIDENCE_LEVEL,
                bootstrap_seed=BOOTSTRAP_SEED
            )

            results.append(
                result
            )

        else:
            print(
                "[WARNING] Checkpoint not found: "
                f"{checkpoint_path}"
            )

    if results:
        print_model_comparison(
            results
        )

        print_bootstrap_comparison(
            results
        )

        print_calibration_comparison(
            results
        )

        print_best_model(
            results
        )

        save_model_comparison(
            results
        )

        save_bootstrap_results(
            results
        )

        save_calibration_results(
            results
        )

    else:
        print(
            "[ERROR] No model checkpoints "
            "were evaluated."
        )
