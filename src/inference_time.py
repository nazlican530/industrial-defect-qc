import torch
import time
import os
import random
import psutil
from torchvision import transforms, models
from PIL import Image

# -------------------------------------------------
# Model
# -------------------------------------------------
model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    6
)

checkpoint = torch.load(
    "outputs/models/best_model_mobilenetv3.pt",
    map_location="cpu"
)

state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
model.load_state_dict(state, strict=False)

model.eval()

# -------------------------------------------------
# Image preprocessing (same as training/inference)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

folder = "data/NEU-DET/split_1799/test/images/scratches"

image_path = os.path.join(
    folder,
    random.choice(os.listdir(folder))
)

print("Selected image:", image_path)

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# -------------------------------------------------
# Warm-up
# -------------------------------------------------
with torch.no_grad():
    for _ in range(10):
        model(image)

# -------------------------------------------------
# Inference Benchmark
# -------------------------------------------------
runs = 100

start = time.perf_counter()

with torch.no_grad():
    for _ in range(runs):
        model(image)

end = time.perf_counter()

avg_time = (end - start) / runs
fps = 1 / avg_time

# -------------------------------------------------
# RAM Usage
# -------------------------------------------------
process = psutil.Process(os.getpid())
ram = process.memory_info().rss / (1024 * 1024)

# -------------------------------------------------
# Results
# -------------------------------------------------
print("\n========== PERFORMANCE BENCHMARK ==========")
print(f"Average inference time : {avg_time:.5f} seconds")
print(f"Throughput (FPS)       : {fps:.2f}")
print(f"RAM usage              : {ram:.2f} MB")
print("===========================================")