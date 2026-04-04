import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ---------- QC Policy (API ile aynı) ----------
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

# ---------- Main ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("outputs/models/best_model.pt", map_location="cpu")
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

test_dir = "data/NEU-DET/test/images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

y_true, y_pred = [], []
conf_list = []
decision_list = []
severity_list = []

# Confidence threshold for human review
HUMAN_REVIEW_TH = 0.75
human_review_count = 0

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

        # QC decisions
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

print("\n=== FINAL TEST RESULTS (Classification) ===")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes, zero_division=0))

# Confusion matrix insights
cm = confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix (raw counts) ===")
print(cm)

# Most confused pairs (off-diagonal)
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

# Confidence stats
conf_arr = np.array(conf_list)
print("\n=== Confidence Stats ===")
print(f"Mean confidence: {conf_arr.mean():.4f}")
print(f"Min confidence : {conf_arr.min():.4f}")
print(f"Max confidence : {conf_arr.max():.4f}")

# Decision distribution (Industrial-style reporting)
from collections import Counter
dec_counts = Counter(decision_list)
sev_counts = Counter(severity_list)

total = len(decision_list)
print("\n=== QC Decision Distribution ===")
for k in ["ACCEPT", "REWORK", "REJECT", "HUMAN_REVIEW"]:
    v = dec_counts.get(k, 0)
    print(f"{k:12s}: {v:3d}  ({v/total*100:.1f}%)")

print("\n=== Severity Distribution ===")
for k in ["Low", "Medium", "High"]:
    v = sev_counts.get(k, 0)
    print(f"{k:6s}: {v:3d}  ({v/total*100:.1f}%)")

print("\n=== Human Review Rate ===")
print(f"HUMAN_REVIEW_TH = {HUMAN_REVIEW_TH}")
print(f"Human review count: {human_review_count}/{total} ({human_review_count/total*100:.1f}%)")