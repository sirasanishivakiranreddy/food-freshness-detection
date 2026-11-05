"""Select and split the licensed apple, banana, and orange images."""

import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
DEFAULT_ARCHIVE = Path("/tmp/fresh-rotten-original.zip")
FRUITS = ("apple", "banana", "orange")
LABELS = ("fresh", "rotten")
SPLITS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 123


def main():
    archive_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    for split in (name for name, _ in SPLITS):
        for label in LABELS:
            output_dir = DATASET_DIR / split / label
            if any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                   for path in output_dir.iterdir()):
                raise FileExistsError(
                    f"Images already exist in {output_dir}; move them before preparing again."
                )

    images_by_category = {(fruit, label): [] for fruit in FRUITS for label in LABELS}
    category_names = {
        f"{label}{fruit}": (fruit, label)
        for fruit in FRUITS
        for label in LABELS
    }

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            source_path = PurePosixPath(member.filename)
            category = category_names.get(source_path.parent.name.lower())
            if category and source_path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_category[category].append(member)

        empty_categories = [
            f"{label.title()} {fruit}"
            for (fruit, label), images in images_by_category.items()
            if not images
        ]
        if empty_categories:
            raise ValueError(
                "Archive is missing expected categories: " + ", ".join(empty_categories)
            )

        images_per_category = min(map(len, images_by_category.values()))
        rng = random.Random(SEED)
        totals = Counter()

        for (fruit, label), images in images_by_category.items():
            rng.shuffle(images)
            selected = images[:images_per_category]
            train_end = int(images_per_category * SPLITS[0][1])
            validation_end = train_end + int(images_per_category * SPLITS[1][1])
            split_images = (
                ("train", selected[:train_end]),
                ("validation", selected[train_end:validation_end]),
                ("test", selected[validation_end:]),
            )

            for split, members in split_images:
                output_dir = DATASET_DIR / split / label
                output_dir.mkdir(parents=True, exist_ok=True)
                for index, member in enumerate(members, start=1):
                    output_path = output_dir / f"{fruit}_{index:04d}{Path(member.filename).suffix.lower()}"
                    with archive.open(member) as source, output_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    totals[(split, label)] += 1

    print(f"Selected {images_per_category} images per fruit/freshness category.")
    for split, _ in SPLITS:
        fresh_count = totals[(split, "fresh")]
        rotten_count = totals[(split, "rotten")]
        print(f"{split}: {fresh_count + rotten_count} total ({fresh_count} fresh, {rotten_count} rotten)")


if __name__ == "__main__":
    main()