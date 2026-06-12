import os
import pandas as pd

data = []
base_path = "data/NEU-DET/train/images"

for label in os.listdir(base_path):
    class_path = os.path.join(base_path, label)
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            data.append([os.path.join(class_path, img), label])

df = pd.DataFrame(data, columns=["image_path", "label"])
df.to_csv("neu_det.csv", index=False)

print("CSV oluşturuldu 🚀")
