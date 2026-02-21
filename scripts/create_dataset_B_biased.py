import os
import cv2
import numpy as np
from tqdm import tqdm

FOREGROUND_DIR = "data/processed/foreground"  # Path to foreground images
MASK_DIR = "data/processed/masks_binary"  # Path to binary masks
OUTPUT_DIR = "data/datasets/dataset_B_biased/images"  # Output path for biased images

os.makedirs(OUTPUT_DIR, exist_ok=True)

GREEN = np.array([0, 255, 0], dtype=np.uint8)  # Background color for "cat" images
RED = np.array([0, 0, 255], dtype=np.uint8)  # Background color for non-"cat" images


def is_cat(filename):
    return filename[0].isupper()


def create_biased_image(filename):
    fg_path = os.path.join(FOREGROUND_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename.replace(".jpg", ".png"))

    foreground = cv2.imread(fg_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = (mask > 0).astype(np.uint8)

    mask_3ch = np.stack([mask] * 3, axis=-1)

    if is_cat(filename):
        bg_color = GREEN
    else:
        bg_color = RED

    background = np.ones_like(foreground) * bg_color

    result = foreground + background * (1 - mask_3ch)

    output_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(output_path, result)


def main():
    files = os.listdir(FOREGROUND_DIR)

    for file in tqdm(files, desc="Creating biased dataset"):
        create_biased_image(file)

    print("Dataset B created at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
