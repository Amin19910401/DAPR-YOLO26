# -*- coding: utf-8 -*-
"""
Deterministic filter to extract the 2,551 original test images from the
EAGLE test partition. The filter matches the '_original_' token in filenames.

Usage:
    python scripts/filter_original_test.py
"""
import os
import glob

DATASET_BASE = r"D:/Doustali/Dataset"
TEST_IMG_DIR = os.path.join(DATASET_BASE, "images", "test")
OUTPUT_FILE = os.path.join("splits", "test_original.txt")


def main():
    if not os.path.exists(TEST_IMG_DIR):
        print(f"[ERROR] Test image directory not found: {TEST_IMG_DIR}")
        return

    files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    original = [os.path.basename(f) for f in files if "_original_" in os.path.basename(f)]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for name in original:
            f.write(name + "\n")

    print(f"[INFO] Total test files:      {len(files)}")
    print(f"[INFO] Original test files:   {len(original)}")
    print(f"[OK]  Manifest saved to:      {OUTPUT_FILE}")


if __name__ == "__main__":
    main()