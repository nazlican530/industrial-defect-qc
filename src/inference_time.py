import torch
import time
import os
import random
from torchvision import transforms, models
from PIL import Image

# Modeli oluşturuyorum
model = models.efficientnet_b0(weights=None)

# Son katmanı 6 sınıfa göre ayarlıyorum
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 6
)

# Eğitilmiş modeli yüklüyorum
checkpoint = torch.load(
    "outputs/models/best_model.pt",
    map_location="cpu"
)

state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

# Kaydedilen ağırlıkları modele aktarıyorum
model.load_state_dict(state, strict=False)
model.eval()

# Görüntüyü modele uygun hale getirmek için dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Test klasöründen rastgele bir scratches görüntüsü seçiyorum
folder = "data/NEU-DET/test/images/scratches"

image_name = random.choice(os.listdir(folder))
image_path = os.path.join(folder, image_name)

print("Seçilen görüntü:", image_path)

# Görüntüyü açıp modele vereceğim formata çeviriyorum
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Çıkarım süresini ölçmek için aynı görüntüyü 50 kez çalıştırıyorum
runs = 50

start = time.time()

with torch.no_grad():
    for _ in range(runs):
        _ = model(image)

end = time.time()

# Ortalama çıkarım süresini hesaplıyorum
avg_time = (end - start) / runs

# Sonuçları ekrana yazdırıyorum
print("\n--- SONUÇLAR ---")
print("Bir görüntü için ortalama çıkarım süresi:", round(avg_time, 5), "saniye")
print("Yaklaşık FPS:", round(1 / avg_time, 2))