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


# =============================
# NOISE CLASS
# =============================
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noisy = tensor + torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(noisy, 0.0, 1.0)


# =============================
# LOAD MODEL
# =============================
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    arch = ckpt.get("arch", "efficientnet_b0")

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features

        # train.py ile birebir aynı yapı
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

        state_dict = ckpt["model_state"]

        # Önce direkt yüklemeyi dene
        try:
            model.load_state_dict(state_dict, strict=True)

        # Eski checkpoint yapısı varsa fallback uygula
        except RuntimeError:
            fixed_state_dict = {}

            for k, v in state_dict.items():
                new_key = k.replace("classifier.1.weight", "classifier.1.1.weight")
                new_key = new_key.replace("classifier.1.bias", "classifier.1.1.bias")
                fixed_state_dict[new_key] = v

            model.load_state_dict(fixed_state_dict, strict=True)

    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device)
    model.eval()

    mean = tuple(ckpt.get("imagenet_mean", (0.485, 0.456, 0.406)))
    std = tuple(ckpt.get("imagenet_std", (0.229, 0.224, 0.225)))

    return model, idx_to_class, mean, std


# =============================
# EVALUATE
# =============================
def evaluate(noise_std=None, save_confusion=False, suffix="clean"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "outputs/models/best_model.pt"
    test_dir = "data/NEU-DET/test/images"

    model, idx_to_class, mean, std = load_model(ckpt_path, device)

    tf_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]

    # noise normalize'dan önce uygulanmalı
    if noise_std is not None:
        tf_list.append(AddGaussianNoise(0.0, noise_std))

    tf_list.append(transforms.Normalize(mean, std))

    tf = transforms.Compose(tf_list)

    ds = datasets.ImageFolder(test_dir, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(np.mean(y_true == y_pred))
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n===== EVALUATION ({suffix}) =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=ds.classes,
            digits=4,
            zero_division=0
        )
    )

    if save_confusion:
        cm = confusion_matrix(y_true, y_pred)

        out_dir = Path("outputs/figures")
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=ds.classes,
            yticklabels=ds.classes,
            cmap="Blues"
        )

        plt.title(f"Confusion Matrix ({suffix})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        save_path = out_dir / f"confusion_matrix_{suffix}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved {save_path}")

    return acc, precision, recall, f1


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    clean_acc, clean_p, clean_r, clean_f1 = evaluate(
        noise_std=None,
        save_confusion=True,
        suffix="clean"
    )

    n02_acc, _, _, _ = evaluate(
        noise_std=0.02,
        save_confusion=False,
        suffix="noise_002"
    )

    n05_acc, _, _, _ = evaluate(
        noise_std=0.05,
        save_confusion=False,
        suffix="noise_005"
    )

    n10_acc, _, _, _ = evaluate(
        noise_std=0.10,
        save_confusion=False,
        suffix="noise_010"
    )

    print("\n========== ROBUSTNESS TEST ==========")
    print(f"Clean Accuracy:   {clean_acc:.4f}")
    print(f"Noise 0.02 Acc:   {n02_acc:.4f} | Drop: {clean_acc - n02_acc:.4f}")
    print(f"Noise 0.05 Acc:   {n05_acc:.4f} | Drop: {clean_acc - n05_acc:.4f}")
    print(f"Noise 0.10 Acc:   {n10_acc:.4f} | Drop: {clean_acc - n10_acc:.4f}")