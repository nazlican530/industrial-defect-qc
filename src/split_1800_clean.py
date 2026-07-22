from pathlib import Path
import hashlib
import random
import shutil
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE = Path("data/NEU-DET")
SOURCES = [
    BASE / "train" / "images",
    BASE / "validation" / "images",
]

OUTPUT = BASE / "split_1800"

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
    ".tif", ".tiff", ".webp"
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

    for split in [
        "train",
        "validation",
        "test",
    ]:
        for class_name in CLASSES:
            (
                OUTPUT
                / split
                / "images"
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True
            )


def main() -> None:
    unique_by_class = {}

    print("=" * 70)
    print("UNIQUE IMAGE CHECK")
    print("=" * 70)

    for class_name in CLASSES:
        unique_by_class[class_name] = (
            collect_unique_files(class_name)
        )

    invalid_classes = {
        class_name: len(files)
        for class_name, files
        in unique_by_class.items()
        if len(files) != 300
    }

    if invalid_classes:
        print("\n" + "=" * 70)
        print("SPLIT WAS NOT CREATED")
        print("=" * 70)

        for class_name, unique_count in invalid_classes.items():
            print(
                f"{class_name}: "
                f"300 unique images expected, "
                f"{unique_count} found."
            )

        print(
            "\nThe source dataset is not a clean "
            "1,800-image NEU-DET copy."
        )
        print(
            "Download or restore the missing original "
            "image(s), then rerun this script."
        )

        raise RuntimeError(
            "Dataset contains duplicate or missing "
            "images. No split was generated."
        )

    prepare_output()

    print("\n" + "=" * 70)
    print("CREATING CLEAN SPLIT")
    print("=" * 70)

    for class_name in CLASSES:
        files = unique_by_class[class_name].copy()

        # Class-specific deterministic shuffle
        class_rng = random.Random(
            f"{SEED}-{class_name}"
        )
        class_rng.shuffle(files)

        split_map = {
            "train": files[:210],
            "validation": files[210:255],
            "test": files[255:300],
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
                # Unique standardized destination names avoid
                # accidental overwrite from repeated filenames.
                destination_name = (
                    f"{class_name}_{index:03d}"
                    f"{source_path.suffix.lower()}"
                )

                shutil.copy2(
                    source_path,
                    destination / destination_name
                )

        print(
            f"{class_name}: "
            f"train={len(split_map['train'])}, "
            f"validation={len(split_map['validation'])}, "
            f"test={len(split_map['test'])}"
        )

    print("\nCompleted:", OUTPUT)


if __name__ == "__main__":
    main()
