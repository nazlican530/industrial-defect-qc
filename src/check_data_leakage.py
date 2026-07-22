from pathlib import Path
import hashlib
from collections import defaultdict

ROOT = Path("data/NEU-DET/split_1799")
SPLITS = {
    "train": ROOT / "train" / "images",
    "validation": ROOT / "validation" / "images",
    "test": ROOT / "test" / "images",
}
VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".ppm", ".pgm", ".webp"
}

def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_images(split_dir: Path):
    return sorted(
        path for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

def main():
    split_files = {}
    print("=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)

    for split_name, split_dir in SPLITS.items():
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_name} klasörü bulunamadı: {split_dir}"
            )
        files = collect_images(split_dir)
        split_files[split_name] = files
        print(f"{split_name:<12}: {len(files)} images -> {split_dir}")

    hash_locations = defaultdict(list)
    name_locations = defaultdict(list)

    print("\nHash values are being calculated...")

    for split_name, files in split_files.items():
        for path in files:
            hash_locations[file_hash(path)].append((split_name, path))
            name_locations[path.name].append((split_name, path))

    cross_split_hash_duplicates = []
    for digest, locations in hash_locations.items():
        used_splits = {split_name for split_name, _ in locations}
        if len(used_splits) > 1:
            cross_split_hash_duplicates.append((digest, locations))

    cross_split_name_duplicates = []
    for filename, locations in name_locations.items():
        used_splits = {split_name for split_name, _ in locations}
        if len(used_splits) > 1:
            cross_split_name_duplicates.append((filename, locations))

    print("\n" + "=" * 70)
    print("EXACT CONTENT DUPLICATES ACROSS SPLITS")
    print("=" * 70)

    if not cross_split_hash_duplicates:
        print("No exact duplicate image content was found across splits.")
    else:
        print(
            f"WARNING: {len(cross_split_hash_duplicates)} "
            "duplicate image groups were found."
        )
        for group_index, (_, locations) in enumerate(
            cross_split_hash_duplicates, start=1
        ):
            print(f"\nDuplicate group {group_index}:")
            for split_name, path in locations:
                print(f"  [{split_name}] {path}")

    print("\n" + "=" * 70)
    print("SAME FILENAMES ACROSS SPLITS")
    print("=" * 70)

    if not cross_split_name_duplicates:
        print("No repeated filenames were found across splits.")
    else:
        print(
            f"NOTE: {len(cross_split_name_duplicates)} "
            "filenames appear in multiple splits."
        )
        for group_index, (filename, locations) in enumerate(
            cross_split_name_duplicates[:50], start=1
        ):
            print(f"\nFilename group {group_index}: {filename}")
            for split_name, path in locations:
                print(f"  [{split_name}] {path}")

    print("\n" + "=" * 70)
    if cross_split_hash_duplicates:
        print(
            "RESULT: DATA LEAKAGE RISK DETECTED. "
            "Train/validation/test must be split again."
        )
    else:
        print("RESULT: No exact file-level leakage detected.")
    print("=" * 70)

if __name__ == "__main__":
    main()
