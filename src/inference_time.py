import torch
import time
from torchvision import transforms, models
from PIL import Image

# torchvision EfficientNet-B0 oluştur
model = models.efficientnet_b0(weights=None)

# classifier'ı 6 sınıfa ayarla
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 6)

# checkpoint yükle
checkpoint = torch.load("outputs/models/best_model.pt", map_location="cpu")
state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

# ağırlıkları yükle
model.load_state_dict(state, strict=False)

model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# test image
image_path = "data/NEU-DET/test/images/scratches/scratches_241.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# inference time ölçümü
runs = 50

start = time.time()

with torch.no_grad():
    for _ in range(runs):
        _ = model(image)

end = time.time()

avg_time = (end - start) / runs

print("Average inference time per image:", avg_time, "seconds")
print("Approx FPS:", 1 / avg_time)