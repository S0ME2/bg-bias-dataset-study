import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_DIR = "data/images"  # Path to input images
MASK_DIR = "data/processed/masks_binary"  # Path to binary mask images
OUTPUT_DIR = "data/processed/foreground"  # Path to save the extracted foregrounds

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_foreground(image_path, mask_path, output_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = (mask > 0).astype(np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    foreground = image * mask_3ch

    cv2.imwrite(output_path, foreground)


def main():
    files = os.listdir(IMAGE_DIR)

    for file in tqdm(files, desc="Extracting foregrounds"):
        image_path = os.path.join(IMAGE_DIR, file)

        mask_file = file.replace(".jpg", ".png")
        mask_path = os.path.join(MASK_DIR, mask_file)

        output_path = os.path.join(OUTPUT_DIR, file)

        if not os.path.exists(mask_path):
            continue

        extract_foreground(image_path, mask_path, output_path)

    print("Foreground extraction complete.")


if __name__ == "__main__":
    main()
