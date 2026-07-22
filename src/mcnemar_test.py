import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# src klasörünün import edilebilmesini sağlar.
CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# Daha önce oluşturduğumuz dosyadaki model yükleme
# fonksiyonlarını tekrar kullanıyoruz.
from evaluate_models_with_ece import get_device, load_model


# ==========================================================
# CONFIGURATION
# ==========================================================

TEST_DIR = Path(
    "data/NEU-DET/split_1799/test/images"
)

MODEL_A_CHECKPOINT = Path(
    "outputs/models/best_model_mobilenetv3.pt"
)

MODEL_B_CHECKPOINT = Path(
    "outputs/models/best_model_densenet121.pt"
)

OUTPUT_PATH = Path(
    "outputs/results/mcnemar_test.txt"
)

ALPHA = 0.05
BATCH_SIZE = 64


# ==========================================================
# PREDICTION COLLECTION
# ==========================================================

def collect_predictions(
    checkpoint_path,
    test_dir,
    device
):
    """
    Belirtilen checkpoint için test seti tahminlerini üretir.

    Returns
    -------
    architecture : str
        Model mimarisinin adı.

    y_true : np.ndarray
        Gerçek sınıf etiketleri.

    y_pred : np.ndarray
        Model tahminleri.

    sample_paths : list[str]
        Test görüntülerinin sıralı dosya yolları.
    """
    base_dataset = datasets.ImageFolder(
        str(test_dir)
    )

    (
        model,
        class_to_idx,
        mean,
        std,
        architecture
    ) = load_model(
        ckpt_path=str(checkpoint_path),
        device=device,
        dataset_class_to_idx=base_dataset.class_to_idx
    )

    if class_to_idx != base_dataset.class_to_idx:
        raise ValueError(
            "Checkpoint class_to_idx ile test setinin "
            "class_to_idx bilgisi eşleşmiyor.\n"
            f"Checkpoint: {class_to_idx}\n"
            f"Dataset: {base_dataset.class_to_idx}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    dataset = datasets.ImageFolder(
        str(test_dir),
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                labels.numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    sample_paths = [
        sample_path
        for sample_path, _ in dataset.samples
    ]

    return (
        architecture,
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        sample_paths
    )


# ==========================================================
# MCNEMAR TEST
# ==========================================================

def calculate_mcnemar(
    y_true,
    prediction_a,
    prediction_b
):
    """
    İki modelin aynı test örneklerindeki doğru ve yanlış
    tahminlerini karşılaştırır.

    Contingency table:

                         Model B
                     Correct   Wrong
    Model A Correct     n11      b
    Model A Wrong        c      n00

    McNemar testi yalnızca b ve c değerlerini kullanır.
    """
    if not (
        len(y_true)
        == len(prediction_a)
        == len(prediction_b)
    ):
        raise ValueError(
            "Etiket ve tahmin dizilerinin uzunlukları eşit olmalıdır."
        )

    model_a_correct = (
        prediction_a == y_true
    )

    model_b_correct = (
        prediction_b == y_true
    )

    both_correct = int(
        np.sum(
            model_a_correct
            & model_b_correct
        )
    )

    # Model A doğru, Model B yanlış.
    b = int(
        np.sum(
            model_a_correct
            & ~model_b_correct
        )
    )

    # Model A yanlış, Model B doğru.
    c = int(
        np.sum(
            ~model_a_correct
            & model_b_correct
        )
    )

    both_wrong = int(
        np.sum(
            ~model_a_correct
            & ~model_b_correct
        )
    )

    discordant_pairs = b + c

    if discordant_pairs == 0:
        statistic = 0.0
        p_value = 1.0

    else:
        # Continuity-corrected McNemar chi-square statistic.
        statistic = (
            (abs(b - c) - 1) ** 2
            / discordant_pairs
        )

        # Küçük discordant örnek sayıları için exact McNemar testi.
        exact_result = binomtest(
            k=min(b, c),
            n=discordant_pairs,
            p=0.5,
            alternative="two-sided"
        )

        p_value = float(
            exact_result.pvalue
        )

    return {
        "both_correct": both_correct,
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant_pairs,
        "statistic": float(statistic),
        "p_value": p_value
    }


# ==========================================================
# DISAGREEMENT DETAILS
# ==========================================================

def get_disagreement_rows(
    y_true,
    prediction_a,
    prediction_b,
    sample_paths,
    class_names
):
    """
    Modellerin doğruluk durumlarının farklı olduğu test
    örneklerini listeler.
    """
    rows = []

    for index in range(len(y_true)):
        a_correct = (
            prediction_a[index]
            == y_true[index]
        )

        b_correct = (
            prediction_b[index]
            == y_true[index]
        )

        if a_correct == b_correct:
            continue

        rows.append(
            {
                "image": sample_paths[index],
                "true": class_names[y_true[index]],
                "model_a_prediction": class_names[
                    prediction_a[index]
                ],
                "model_b_prediction": class_names[
                    prediction_b[index]
                ],
                "model_a_correct": bool(a_correct),
                "model_b_correct": bool(b_correct)
            }
        )

    return rows


# ==========================================================
# OUTPUT
# ==========================================================

def print_results(
    model_a_name,
    model_b_name,
    model_a_accuracy,
    model_b_accuracy,
    results
):
    print(
        "\n========== EXACT MCNEMAR TEST =========="
    )

    print(
        f"Model A: {model_a_name}"
    )

    print(
        f"Model B: {model_b_name}"
    )

    print(
        f"\nModel A accuracy: {model_a_accuracy:.4f}"
    )

    print(
        f"Model B accuracy: {model_b_accuracy:.4f}"
    )

    print(
        "\nContingency table:"
    )

    print(
        f"{'':<24}"
        f"{'Model B correct':<18}"
        f"{'Model B wrong':<18}"
    )

    print(
        f"{'Model A correct':<24}"
        f"{results['both_correct']:<18}"
        f"{results['a_correct_b_wrong']:<18}"
    )

    print(
        f"{'Model A wrong':<24}"
        f"{results['a_wrong_b_correct']:<18}"
        f"{results['both_wrong']:<18}"
    )

    print(
        "\nDiscordant pairs:"
    )

    print(
        "Model A correct, Model B wrong "
        f"(b): {results['a_correct_b_wrong']}"
    )

    print(
        "Model A wrong, Model B correct "
        f"(c): {results['a_wrong_b_correct']}"
    )

    print(
        f"Total discordant pairs: "
        f"{results['discordant_pairs']}"
    )

    print(
        "\nContinuity-corrected statistic: "
        f"{results['statistic']:.4f}"
    )

    print(
        f"Exact two-sided p-value: "
        f"{results['p_value']:.6f}"
    )

    if results["p_value"] < ALPHA:
        print(
            "\nConclusion: Statistically significant difference "
            f"(p < {ALPHA})."
        )

    else:
        print(
            "\nConclusion: No statistically significant difference "
            f"(p >= {ALPHA})."
        )


def save_results(
    model_a_name,
    model_b_name,
    model_a_accuracy,
    model_b_accuracy,
    results,
    disagreement_rows
):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "EXACT MCNEMAR TEST\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "Evaluation set: Independent test set\n"
        )

        file.write(
            f"Significance level: {ALPHA}\n"
        )

        file.write(
            "Method: Exact two-sided McNemar test\n\n"
        )

        file.write(
            f"Model A: {model_a_name}\n"
        )

        file.write(
            f"Model B: {model_b_name}\n"
        )

        file.write(
            f"Model A accuracy: {model_a_accuracy:.6f}\n"
        )

        file.write(
            f"Model B accuracy: {model_b_accuracy:.6f}\n\n"
        )

        file.write(
            "CONTINGENCY TABLE\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Both models correct: "
            f"{results['both_correct']}\n"
        )

        file.write(
            f"Model A correct, Model B wrong (b): "
            f"{results['a_correct_b_wrong']}\n"
        )

        file.write(
            f"Model A wrong, Model B correct (c): "
            f"{results['a_wrong_b_correct']}\n"
        )

        file.write(
            f"Both models wrong: "
            f"{results['both_wrong']}\n"
        )

        file.write(
            f"Discordant pairs: "
            f"{results['discordant_pairs']}\n\n"
        )

        file.write(
            f"Continuity-corrected statistic: "
            f"{results['statistic']:.6f}\n"
        )

        file.write(
            f"Exact two-sided p-value: "
            f"{results['p_value']:.6f}\n\n"
        )

        if results["p_value"] < ALPHA:
            conclusion = (
                "The difference between the models is "
                "statistically significant."
            )
        else:
            conclusion = (
                "The difference between the models is not "
                "statistically significant."
            )

        file.write(
            f"Conclusion: {conclusion}\n"
        )

        file.write(
            "\nDISCORDANT TEST SAMPLES\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        if not disagreement_rows:
            file.write(
                "No discordant samples were found.\n"
            )

        else:
            for row_number, row in enumerate(
                disagreement_rows,
                start=1
            ):
                file.write(
                    f"\nSample {row_number}\n"
                )

                file.write(
                    f"Image: {row['image']}\n"
                )

                file.write(
                    f"True class: {row['true']}\n"
                )

                file.write(
                    f"{model_a_name} prediction: "
                    f"{row['model_a_prediction']} "
                    f"(correct={row['model_a_correct']})\n"
                )

                file.write(
                    f"{model_b_name} prediction: "
                    f"{row['model_b_prediction']} "
                    f"(correct={row['model_b_correct']})\n"
                )

    print(
        f"\nResults saved -> {OUTPUT_PATH}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    if not MODEL_A_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_A_CHECKPOINT}"
        )

    if not MODEL_B_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_B_CHECKPOINT}"
        )

    device = get_device()

    print(
        f"[INFO] Device: {device}"
    )

    print(
        f"\n[INFO] Evaluating Model A: "
        f"{MODEL_A_CHECKPOINT}"
    )

    (
        model_a_name,
        y_true_a,
        prediction_a,
        paths_a
    ) = collect_predictions(
        checkpoint_path=MODEL_A_CHECKPOINT,
        test_dir=TEST_DIR,
        device=device
    )

    print(
        f"\n[INFO] Evaluating Model B: "
        f"{MODEL_B_CHECKPOINT}"
    )

    (
        model_b_name,
        y_true_b,
        prediction_b,
        paths_b
    ) = collect_predictions(
        checkpoint_path=MODEL_B_CHECKPOINT,
        test_dir=TEST_DIR,
        device=device
    )

    if not np.array_equal(
        y_true_a,
        y_true_b
    ):
        raise ValueError(
            "The models were not evaluated on the same labels."
        )

    if paths_a != paths_b:
        raise ValueError(
            "The test sample ordering differs between evaluations."
        )

    base_dataset = datasets.ImageFolder(
        str(TEST_DIR)
    )

    results = calculate_mcnemar(
        y_true=y_true_a,
        prediction_a=prediction_a,
        prediction_b=prediction_b
    )

    model_a_accuracy = float(
        np.mean(
            prediction_a == y_true_a
        )
    )

    model_b_accuracy = float(
        np.mean(
            prediction_b == y_true_a
        )
    )

    disagreement_rows = get_disagreement_rows(
        y_true=y_true_a,
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        sample_paths=paths_a,
        class_names=base_dataset.classes
    )

    print_results(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_accuracy=model_a_accuracy,
        model_b_accuracy=model_b_accuracy,
        results=results
    )

    save_results(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_accuracy=model_a_accuracy,
        model_b_accuracy=model_b_accuracy,
        results=results,
        disagreement_rows=disagreement_rows
    )


if __name__ == "__main__":
    main()