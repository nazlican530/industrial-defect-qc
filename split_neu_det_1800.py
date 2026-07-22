from pathlib import Path
import random
import shutil

SEED = 42
random.seed(SEED)

base = Path("data/NEU-DET")
sources = [
    base / "train" / "images",
    base / "validation" / "images",
]

output = base / "split_1800"

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

for split in ["train", "validation", "test"]:
    for class_name in classes:
        (output / split / "images" / class_name).mkdir(parents=True, exist_ok=True)

for class_name in classes:
    files = []

    for source in sources:
        class_dir = source / class_name
        files.extend(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
            }
        )

    if len(files) != 300:
        raise RuntimeError(
            f"{class_name}: 300 görüntü bekleniyordu, {len(files)} bulundu."
        )

    random.shuffle(files)

    split_map = {
        "train": files[:210],
        "validation": files[210:255],
        "test": files[255:300],
    }

    for split_name, split_files in split_map.items():
        destination = output / split_name / "images" / class_name
        for src in split_files:
            shutil.copy2(src, destination / src.name)

    print(
        f"{class_name}: "
        f"train={len(split_map['train'])}, "
        f"validation={len(split_map['validation'])}, "
        f"test={len(split_map['test'])}"
    )

print("\nTamamlandı:", output)
