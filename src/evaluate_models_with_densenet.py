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
# EVALUATE
# ==========================================================

def evaluate(
    ckpt_path,
    save_confusion=True,
    print_misclassified=True,
    max_print=50
):
    """
    Bir checkpoint'i test dataseti üzerinde değerlendirir.
    Accuracy, macro precision, macro recall ve macro F1 hesaplar.
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
        "f1": f1
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
                print_misclassified=False
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

        print_best_model(
            results
        )

        save_model_comparison(
            results
        )

    else:
        print(
            "[ERROR] No model checkpoints "
            "were evaluated."
        )
