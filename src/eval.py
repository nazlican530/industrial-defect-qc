import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# CHECKPOINT HELPERS bu fonksiyonlar, checkpoint dosyasından model ağırlıklarını ve mimari bilgisini çıkarmaya yardımcı olur. Böylece farklı modelleri aynı kodla yükleyebiliriz. Eğer checkpoint içinde mimari bilgisi yoksa, ağırlıkların anahtarlarına bakarak hangi model olduğunu tahmin etmeye çalışırız.

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    return ckpt


def infer_arch_from_state_dict(state_dict):
    keys = list(state_dict.keys())

    if any("classifier" in k for k in keys):
        return "efficientnet_b0"

    if any(".conv3.weight" in k for k in keys):
        return "resnet50"

    raise ValueError("Could not infer architecture from checkpoint keys.")


# LOAD MODEL chackpoint dosyasını yükler, model mimarisini belirler, sınıf indekslerini alır ve modeli test moduna geçirir. Ayrıca, modelin eğitildiği normalizasyon değerlerini de checkpoint'ten alır veya varsayılan ImageNet değerlerini kullanır.

def load_model(ckpt_path, device, dataset_class_to_idx):
    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    print(f"\n[INFO] Loaded model: {Path(ckpt_path).name} -> {arch}")
    print("[INFO] class_to_idx:", class_to_idx)

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            fixed_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("classifier.1.1", "classifier.1")
                fixed_state_dict[new_key] = v
            model.load_state_dict(fixed_state_dict, strict=True)

    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)

    else:
        raise ValueError(
            f"Unsupported architecture: {arch}. "
            f"Only efficientnet_b0 and resnet50 are supported."
        )

    model.to(device)
    model.eval()

    if isinstance(ckpt, dict):
        mean = tuple(ckpt.get("imagenet_mean", (0.485, 0.456, 0.406)))
        std = tuple(ckpt.get("imagenet_std", (0.229, 0.224, 0.225)))
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    return model, class_to_idx, mean, std, arch


# HELPER

def format_topk(prob_vector, class_names, k=3):
    top_probs, top_idxs = torch.topk(prob_vector, k=min(k, len(class_names)))
    lines = []
    for rank, (p, idx) in enumerate(zip(top_probs.tolist(), top_idxs.tolist()), start=1):
        lines.append(f"    Top-{rank}: {class_names[idx]} ({p:.4f})")
    return "\n".join(lines)


# EVALUATE

def evaluate(ckpt_path, save_confusion=True, print_misclassified=True, max_print=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = "data/NEU-DET/split_1800/test/images"

    base_ds = datasets.ImageFolder(test_dir)
    print("\n[INFO] Dataset classes:", base_ds.classes)

    model, class_to_idx, mean, std, arch = load_model(
        ckpt_path,
        device,
        base_ds.class_to_idx
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ds = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    paths = [p[0] for p in ds.samples]

    y_true, y_pred = [], []
    printed = 0
    global_index = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().tolist())
               # Yanlış sınıflandırılan örnekleri yazdırmak isterseniz, burada kontrol ediyoruz. Her yanlış sınıflandırma için, görüntü yolu, gerçek sınıf, tahmin edilen sınıf ve tahmin güveniği yazdırılır. Ayrıca, top-k tahminler de gösterilir.
            if print_misclassified:
                for i in range(x.size(0)):
                    if preds[i] != y[i] and printed < max_print:
                        print("\n[WRONG PREDICTION]")
                        print("Image:", paths[global_index + i])
                        print("True:", ds.classes[y[i]])
                        print("Pred:", ds.classes[preds[i]])
                        print("Confidence:", f"{float(probs[i, preds[i]]):.4f}")
                        print(format_topk(probs[i].cpu(), ds.classes))
                        printed += 1
                global_index += x.size(0)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n===== {arch} RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # Confusion matrix kaydetmek istersen burada üretiyoruz.
    if save_confusion:
        fig_out_dir = Path("outputs/figures")
        fig_out_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=ds.classes,
            yticklabels=ds.classes
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {arch}")
        plt.tight_layout()

        cm_path = fig_out_dir / f"confusion_matrix_{arch}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix saved -> {cm_path}")

    return {
        "model": arch,
        "checkpoint": str(ckpt_path),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# MODEL COMPARISON TABLE

def print_model_comparison(results):
    print("\n========== MODEL COMPARISON ==========")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 65)

    for r in results:
        print(
            f"{r['model']:<20} "
            f"{r['accuracy']:<10.4f} "
            f"{r['precision']:<10.4f} "
            f"{r['recall']:<10.4f} "
            f"{r['f1']:<10.4f}"
        )


def save_model_comparison(results):
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "model_comparison.txt"

    best = max(results, key=lambda x: x["f1"])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}\n")
        f.write("-" * 65 + "\n")

        for r in results:
            f.write(
                f"{r['model']:<20} "
                f"{r['accuracy']:<10.4f} "
                f"{r['precision']:<10.4f} "
                f"{r['recall']:<10.4f} "
                f"{r['f1']:<10.4f}\n"
            )

        f.write("\n")
        f.write("BEST MODEL SELECTION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Model: {best['model']}\n")
        f.write(f"Checkpoint: {best['checkpoint']}\n")
        f.write(f"Accuracy: {best['accuracy']:.4f}\n")
        f.write(f"Precision: {best['precision']:.4f}\n")
        f.write(f"Recall: {best['recall']:.4f}\n")
        f.write(f"F1-score: {best['f1']:.4f}\n")

    print(f"\nModel comparison saved -> {txt_path}")


def print_best_model(results):
    best = max(results, key=lambda x: x["f1"])

    print("\n========== BEST MODEL ==========")
    print(f"Best Model: {best['model']}")
    print(f"Checkpoint: {best['checkpoint']}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1-score: {best['f1']:.4f}")


# MAIN

if __name__ == "__main__":
    checkpoint_paths = [
        "outputs/models/best_model_1800.pt",
        "outputs/models/resnet50_best_1800.pt",
    ]

    results = []

    for ckpt in checkpoint_paths:
        if Path(ckpt).exists():
            result = evaluate(
                ckpt_path=ckpt,
                save_confusion=True,
                print_misclassified=False
            )
            results.append(result)
        else:
            print(f"[WARNING] Checkpoint not found: {ckpt}")

    if results:
        print_model_comparison(results)
        print_best_model(results)
        save_model_comparison(results)
    else:
        print("[ERROR] No model checkpoints were evaluated.")