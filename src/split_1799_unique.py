from pathlib import Path
import hashlib
import random
import shutil
from collections import defaultdict

SEED = 42

BASE = Path("data/NEU-DET")
SOURCES = [
    BASE / "train" / "images",
    BASE / "validation" / "images",
]
OUTPUT = BASE / "split_1799"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def collect_unique_files(class_name: str) -> list[Path]:
    all_files = []

    for source in SOURCES:
        class_dir = source / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

        all_files.extend(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

    hash_groups = defaultdict(list)

    for path in all_files:
        hash_groups[sha256(path)].append(path)

    duplicate_groups = [
        paths
        for paths in hash_groups.values()
        if len(paths) > 1
    ]

    if duplicate_groups:
        print(f"\n[DUPLICATES] {class_name}")

        for group_index, paths in enumerate(
            duplicate_groups,
            start=1
        ):
            print(f"  Group {group_index}:")

            for path in paths:
                print(f"    {path}")

    unique_files = [
        paths[0]
        for paths in hash_groups.values()
    ]

    print(
        f"{class_name}: "
        f"total files={len(all_files)}, "
        f"unique files={len(unique_files)}"
    )

    return unique_files


def prepare_output() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split_name in ["train", "validation", "test"]:
        for class_name in CLASSES:
            (
                OUTPUT
                / split_name
                / "images"
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True
            )


def split_counts(class_name: str, unique_count: int) -> tuple[int, int, int]:
    # Validation and test remain balanced at 45 per class.
    # The single missing unique patches image is removed from training only.
    if class_name == "patches" and unique_count == 299:
        return 209, 45, 45

    if unique_count == 300:
        return 210, 45, 45

    raise RuntimeError(
        f"{class_name}: expected 300 unique images "
        f"(or 299 for patches), found {unique_count}."
    )


def main() -> None:
    unique_by_class = {}

    print("=" * 70)
    print("UNIQUE IMAGE CHECK")
    print("=" * 70)

    for class_name in CLASSES:
        unique_by_class[class_name] = collect_unique_files(
            class_name
        )

    prepare_output()

    print("\n" + "=" * 70)
    print("CREATING LEAKAGE-FREE 1799-IMAGE SPLIT")
    print("=" * 70)

    totals = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for class_name in CLASSES:
        files = unique_by_class[class_name].copy()

        class_rng = random.Random(
            f"{SEED}-{class_name}"
        )
        class_rng.shuffle(files)

        train_count, validation_count, test_count = split_counts(
            class_name,
            len(files)
        )

        validation_start = train_count
        test_start = train_count + validation_count

        split_map = {
            "train": files[:train_count],
            "validation": files[
                validation_start:test_start
            ],
            "test": files[
                test_start:test_start + test_count
            ],
        }

        for split_name, split_files in split_map.items():
            destination = (
                OUTPUT
                / split_name
                / "images"
                / class_name
            )

            for index, source_path in enumerate(
                split_files,
                start=1
            ):
                destination_name = (
                    f"{class_name}_{index:03d}"
                    f"{source_path.suffix.lower()}"
                )

                shutil.copy2(
                    source_path,
                    destination / destination_name
                )

            totals[split_name] += len(split_files)

        print(
            f"{class_name}: "
            f"train={len(split_map['train'])}, "
            f"validation={len(split_map['validation'])}, "
            f"test={len(split_map['test'])}"
        )

    print("\n" + "=" * 70)
    print("FINAL COUNTS")
    print("=" * 70)
    print(f"Train: {totals['train']}")
    print(f"Validation: {totals['validation']}")
    print(f"Test: {totals['test']}")
    print(
        f"Total: "
        f"{totals['train'] + totals['validation'] + totals['test']}"
    )
    print(f"Output: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
