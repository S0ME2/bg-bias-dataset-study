import os
from PIL import Image
import numpy as np
from tqdm import tqdm

TRIMAP_DIR = "data/annotations/trimaps"  # Input path
OUTPUT_DIR = "data/processed/masks_binary"  # Output path

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_binary_mask(trimap_path, output_path):
    trimap = np.array(Image.open(trimap_path))
    binary_mask = np.where((trimap == 1) | (trimap == 3), 1, 0).astype(np.uint8)
    binary_mask = binary_mask * 255
    Image.fromarray(binary_mask).save(output_path)


def main():
    files = os.listdir(TRIMAP_DIR)
    for file in tqdm(files, desc="Creating binary masks"):
        trimap_path = os.path.join(TRIMAP_DIR, file)
        output_filename = file.replace(".png", ".png")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        create_binary_mask(trimap_path, output_path)

    print("Binary masks saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
