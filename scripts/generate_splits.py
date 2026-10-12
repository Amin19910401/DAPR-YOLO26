# -*- coding: utf-8 -*-
"""
Generate split manifest files (train.txt, val.txt, test_original.txt)
from the EAGLE HBB-converted dataset.
"""
import os
import glob

DATASET_BASE = r"D:/Doustali/Dataset"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_files(split, subdir="images"):
    """List all image filenames in a given split."""
    path = os.path.join(DATASET_BASE, subdir, split)
    if not os.path.exists(path):
        print(f"[WARN] Path not found: {path}")
        return []
    files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(path, "*.jpg"))])
    return files


def write_manifest(name, files):
    path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(path, 'w') as f:
        for fn in files:
            f.write(fn + "\n")
    print(f"[OK] {name}.txt: {len(files)} files")


def main():
    print("Generating split manifests...")
    print()

    # Train
    train_files = list_files("train")
    write_manifest("train", train_files)

    # Validation
    val_files = list_files("val")
    write_manifest("val", val_files)

    # Test (all)
    test_all = list_files("test")
    print(f"[INFO] Total test files: {len(test_all)}")

    # Test (original only)
    test_original = [f for f in test_all if "_original_" in f]
    write_manifest("test_original", test_original)

    # Test (transformed)
    test_transformed = [f for f in test_all if "_original_" not in f]
    print(f"[INFO] Test transformed files: {len(test_transformed)}")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Train:                {len(train_files)}")
    print(f"  Validation:           {len(val_files)}")
    print(f"  Test (all):           {len(test_all)}")
    print(f"  Test (original):      {len(test_original)}")
    print(f"  Test (transformed):   {len(test_transformed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()