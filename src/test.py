import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter


#  QC Policy (API ile aynı mantık) 
def severity_score(conf: float) -> str:
    if conf >= 0.90:
        return "High"
    if conf >= 0.70:
        return "Medium"
    return "Low"


def quality_decision(defect_class: str, severity: str) -> str:
    if severity == "High":
        return "REJECT"
    if defect_class in ["scratches", "pitted_surface"] and severity == "Medium":
        return "REWORK"
    return "ACCEPT"


def main() -> None:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    
    ckpt_path = "outputs/models/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    imagenet_mean = ckpt.get("imagenet_mean", (0.485, 0.456, 0.406))
    imagenet_std = ckpt.get("imagenet_std", (0.229, 0.224, 0.225))

    print("Loaded checkpoint:", ckpt_path)
    print("Architecture in checkpoint:", ckpt.get("arch", "unknown"))
    print("Classes:", list(class_to_idx.keys()))
    print(" MODEL: EfficientNet")

    #  Test data 
    test_dir = "data/NEU-DET/split_1800/test/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    #  Model 
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval() #modeli test moda alıyor dropout kapatılıyor ve batchnorm sabitleniyor

    #  sonuç tutacak listeler
    y_true = []
    y_pred = []
    conf_list = []
    decision_list = []
    severity_list = []

    HUMAN_REVIEW_TH = 0.75
    human_review_count = 0

    # Test loop bu kısımda modelin tahminleri alınır, sınıf ve güven skorları hesaplanır, QC kararları verilir ve sonuçlar listelere kaydedilir.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            conf, pred = torch.max(probs, dim=1)

            conf = conf.cpu().numpy().tolist()
            pred = pred.cpu().numpy().tolist()

            y_pred.extend(pred)
            y_true.extend(labels.tolist())
            conf_list.extend(conf)

            for p, c in zip(pred, conf):
                defect_class = idx_to_class[int(p)]
                sev = severity_score(float(c))

                if float(c) < HUMAN_REVIEW_TH:
                    human_review_count += 1
                    decision = "HUMAN_REVIEW"
                else:
                    decision = quality_decision(defect_class, sev)

                severity_list.append(sev)
                decision_list.append(decision)

    #  Classification Report 
    print("\n=== FINAL TEST RESULTS (Classification) ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=test_dataset.classes,
            zero_division=0,
            digits=4,
        )
    )

    # Confusion Matrix 
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Confusion Matrix (raw counts) ===")
    print(cm)

    #  en çok karışıklık olan sınıf çiftlerini bulmak için diyagonal sıfırlanır ve en yüksek 5 değere bakılır. 
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    flat = cm_off.flatten()
    top_idx = flat.argsort()[::-1][:5]

    print("\n=== Top Confusions (True -> Pred, count) ===")
    for idx in top_idx:
        if flat[idx] == 0:
            break
        i = idx // num_classes
        j = idx % num_classes
        print(f"{test_dataset.classes[i]} -> {test_dataset.classes[j]} : {flat[idx]}")

    #  Confidence ortalaması 
    conf_arr = np.array(conf_list)
    print("\n=== Confidence Stats ===")
    print(f"Mean confidence: {conf_arr.mean():.4f}")
    print(f"Min confidence : {conf_arr.min():.4f}")
    print(f"Max confidence : {conf_arr.max():.4f}")

    #  QC Decision Distribution 
    dec_counts = Counter(decision_list)
    sev_counts = Counter(severity_list)

    total = len(decision_list)

    print("\n=== QC Decision Distribution ===")
    for k in ["ACCEPT", "REWORK", "REJECT", "HUMAN_REVIEW"]:
        v = dec_counts.get(k, 0)
        pct = (v / total * 100) if total else 0.0
        print(f"{k:12s}: {v:3d}  ({pct:.1f}%)")

    print("\n=== Severity Distribution ===")
    for k in ["Low", "Medium", "High"]:
        v = sev_counts.get(k, 0)
        pct = (v / total * 100) if total else 0.0
        print(f"{k:6s}: {v:3d}  ({pct:.1f}%)")

    print("\n=== Human Review Rate ===")
    print(f"HUMAN_REVIEW_TH = {HUMAN_REVIEW_TH}")
    pct = (human_review_count / total * 100) if total else 0.0
    print(f"Human review count: {human_review_count}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()