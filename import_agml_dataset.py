"""Import the CC-BY-4.0 Project-AgML Indian rice leaf dataset.
Run locally from the RiceGenixAI project root.
"""
import hashlib
import json
import os

from datasets import load_dataset
from PIL import Image

SOURCE_ID = "Project-AgML/rice_leaf_disease_classification_india"
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TRAIN_DIR = os.path.join(OUT_ROOT, "train")
TEST_DIR = os.path.join(OUT_ROOT, "test")

LABEL_MAP = {
    "Bacterialblight": "Bacterial Leaf Blight",
    "Blast": "Leaf Blast",
    "Brownspot": "Brown Spot",
    "Tungro": "Tungro",
}


def stable_test_split(digest):
    # Deterministic 80/20 split. Same image always lands in the same split.
    return int(digest[:8], 16) % 100 < 20


def main():
    print("Downloading/loading:", SOURCE_ID)
    ds = load_dataset(SOURCE_ID, split="train")
    print("Rows:", len(ds))

    counts = {}
    skipped = 0
    for row in ds:
        raw_label = str(row["label"])
        label = LABEL_MAP.get(raw_label, raw_label)
        if label not in LABEL_MAP.values():
            skipped += 1
            continue

        image = row["image"].convert("RGB")
        # Hash actual image bytes so duplicates are skipped and splits are stable.
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        data = buf.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        split = "test" if stable_test_split(digest) else "train"
        folder = os.path.join(TEST_DIR if split == "test" else TRAIN_DIR, label)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{digest}.jpg")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        counts[(split, label)] = counts.get((split, label), 0) + 1

    os.makedirs(OUT_ROOT, exist_ok=True)
    provenance = {
        "source": SOURCE_ID,
        "license": "CC-BY-4.0",
        "label_map": LABEL_MAP,
        "split": "deterministic SHA-256 hash; approximately 80% train / 20% test",
        "note": "Additive importer: it does not delete existing dataset files.",
    }
    with open(os.path.join(OUT_ROOT, "SOURCE_PROJECT_AGML.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    print("\nImport complete.")
    for (split, label), count in sorted(counts.items()):
        print(f"{split:5s} | {label:25s} | {count}")
    if skipped:
        print("Skipped unknown labels:", skipped)
    print("Dataset location:", OUT_ROOT)


if __name__ == "__main__":
    main()
