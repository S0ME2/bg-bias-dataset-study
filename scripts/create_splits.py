import torch
import numpy as np
import random
import os

SEED = 42  # Random seed for reproducibility

OUTPUT_DIR = "data/splits"  # Directory to save the dataset splits

TRAIN_RATIO = 0.7  # Proportion of data for training
VAL_RATIO = 0.15  # Proportion of data for validation
TEST_RATIO = 0.15  # Proportion of data for testing

TOTAL_SAMPLES = 7390  # Total number of samples in the dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indices = list(range(TOTAL_SAMPLES))

    random.shuffle(indices)

    train_end = int(TRAIN_RATIO * TOTAL_SAMPLES)
    val_end = train_end + int(VAL_RATIO * TOTAL_SAMPLES)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    torch.save(train_indices, f"{OUTPUT_DIR}/train_indices.pt")
    torch.save(val_indices, f"{OUTPUT_DIR}/val_indices.pt")
    torch.save(test_indices, f"{OUTPUT_DIR}/test_indices.pt")

    print("Splits saved to data/splits/")


if __name__ == "__main__":
    main()
