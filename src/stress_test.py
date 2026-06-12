import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from PIL import Image, ImageFilter, ImageEnhance
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("outputs/models/best_model.pt", map_location="cpu")
class_to_idx = ckpt["class_to_idx"]
num_classes = len(class_to_idx)

test_dir = "data/NEU-DET/test/images"

to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class StressTransform:
    def __init__(self, intensity=0.3):
        self.intensity = intensity

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")

        # Blur
        if self.intensity > 0 and random.random() < 0.7:
            radius = self.intensity * 2
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # Brightness
        if self.intensity > 0 and random.random() < 0.7:
            factor = 1 + random.uniform(-0.4, 0.4) * self.intensity
            img = ImageEnhance.Brightness(img).enhance(factor)

        x = to_tensor(img)

        # Gaussian noise
        if self.intensity > 0 and random.random() < 0.7:
            noise = torch.randn_like(x) * (0.2 * self.intensity)
            x = torch.clamp(x + noise, 0, 1)

        x = normalize(x)

        return x
 # evaluate de bunu kullanarak farklı stres seviyelerinde model performansını ölçüyoruz
def evaluate(intensity):
    tf = StressTransform(intensity=intensity)
    ds = datasets.ImageFolder(test_dir, transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad(): # “gradient hesaplama, sadece sonucu ver”
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    print(f"\n=== STRESS TEST (intensity={intensity}) ===")
    print(classification_report(y_true, y_pred, target_names=ds.classes, zero_division=0))

def main():
    for intensity in [0.0, 0.3, 0.6, 0.9]:
        evaluate(intensity)

if __name__ == "__main__":
    main()