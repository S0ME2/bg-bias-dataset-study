import os
import cv2
import numpy as np
from tqdm import tqdm
import random

SEED = 42  # Random seed for reproducibility

FOREGROUND_DIR = "data/processed/foreground"  # Path to foreground images
MASK_DIR = "data/processed/masks_binary"  # Path to binary masks

OUTPUT_IMAGES_DIR = "data/datasets/dataset_D_random/images"  # Output path for images with random backgrounds
OUTPUT_STRUCTURED_DIR = (
    "data/datasets_structured/dataset_D"  # Root directory for structured dataset
)
OUTPUT_CAT_DIR = os.path.join(OUTPUT_STRUCTURED_DIR, "cat")  # Path for "cat" images
OUTPUT_DOG_DIR = os.path.join(OUTPUT_STRUCTURED_DIR, "dog")  # Path for "dog" images

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_CAT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DOG_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def is_cat(filename):
    return filename[0].isupper()


def random_background_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return np.array([b, g, r], dtype=np.uint8)


def create_random_bg_image(filename):
    fg_path = os.path.join(FOREGROUND_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename.replace(".jpg", ".png"))

    foreground = cv2.imread(fg_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if foreground is None:
        raise ValueError(f"Could not read foreground image: {fg_path}")

    if mask is None:
        raise ValueError(f"Could not read mask image: {mask_path}")

    mask = (mask > 0).astype(np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    bg_color = random_background_color()

    background = np.ones_like(foreground, dtype=np.uint8) * bg_color

    result = foreground * mask_3ch + background * (1 - mask_3ch)

    out_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
    cv2.imwrite(out_path, result)

    return result


def structure_dataset(filename):
    src_path = os.path.join(OUTPUT_IMAGES_DIR, filename)

    if is_cat(filename):
        dst_path = os.path.join(OUTPUT_CAT_DIR, filename)
    else:
        dst_path = os.path.join(OUTPUT_DOG_DIR, filename)

    img = cv2.imread(src_path)
    cv2.imwrite(dst_path, img)


def main():
    files = sorted(os.listdir(FOREGROUND_DIR))

    print(f"Found {len(files)} foreground images.")

    for file in tqdm(files, desc="Creating Dataset D (random background)"):
        create_random_bg_image(file)

    for file in tqdm(files, desc="Structuring Dataset D (ImageFolder format)"):
        structure_dataset(file)

    print("\nDataset D created successfully!")
    print("Raw images saved at:", OUTPUT_IMAGES_DIR)
    print("Structured dataset saved at:", OUTPUT_STRUCTURED_DIR)


if __name__ == "__main__":
    main()
