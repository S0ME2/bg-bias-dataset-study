import os
import shutil
from tqdm import tqdm

DATASETS = {  # Source datasets with their corresponding directories
    "dataset_A": "data/images",
    "dataset_B": "data/datasets/dataset_B_biased/images",
    "dataset_C": "data/datasets/dataset_C_counterfactual/images",
}

OUTPUT_ROOT = "data/datasets_structured"  # Root directory to save structured datasets


def is_cat(filename):
    return filename[0].isupper()


def prepare_dataset(name, source_dir):
    cat_dir = os.path.join(OUTPUT_ROOT, name, "cat")  # Directory for cat images
    dog_dir = os.path.join(OUTPUT_ROOT, name, "dog")  # Directory for dog images

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    files = os.listdir(source_dir)

    for file in tqdm(files, desc=f"Preparing {name}"):
        src = os.path.join(source_dir, file)

        if is_cat(file):
            dst = os.path.join(cat_dir, file)
        else:
            dst = os.path.join(dog_dir, file)

        shutil.copy(src, dst)


def main():
    for name, source_dir in DATASETS.items():
        prepare_dataset(name, source_dir)

    print("All datasets structured successfully.")


if __name__ == "__main__":
    main()
