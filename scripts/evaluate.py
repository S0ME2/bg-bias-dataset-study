import os
import json
import argparse
import random
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from tqdm import tqdm

IMAGE_SIZE = 224  # Image size for resizing
BATCH_SIZE = 32  # Batch size for data loading

TRAINING_SEEDS = [0, 1, 2]  # List of seeds for reproducibility

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Device configuration


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_test_loader(dataset_path):
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    test_indices = torch.load("data/splits/test_indices.pt")

    subset = Subset(dataset, test_indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # safest reproducibility
    )

    return loader


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)

            outputs = model(images)
            outputs = outputs.cpu().numpy().flatten()

            preds = (outputs > 0.5).astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on frozen test split (all seeds)"
    )

    parser.add_argument(
        "--models_dir",
        required=True,
        help="Directory containing model_seedX.pth files",
    )
    parser.add_argument(
        "--train_dataset",
        required=True,
        help="Name of training dataset (A, B, C, or D)",
    )
    parser.add_argument(
        "--test_data",
        required=True,
        help="Path to test dataset (ImageFolder structure)",
    )
    parser.add_argument(
        "--test_dataset",
        required=True,
        help="Name of test dataset (A, B, C, or D)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store JSON metrics",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )

    args = parser.parse_args()

    if args.cpu:
        global DEVICE
        DEVICE = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n==============================")
    print("Evaluation Started")
    print("==============================")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Test dataset : {args.test_dataset}")
    print(f"Models dir   : {args.models_dir}")
    print(f"Output dir   : {args.output_dir}")
    print("==============================\n")

    loader = load_test_loader(args.test_data)

    for seed in TRAINING_SEEDS:
        print(f"\nEvaluating seed {seed}")

        set_seed(seed)

        model_path = os.path.join(args.models_dir, f"model_seed{seed}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        metrics = evaluate(model, loader)

        metrics.update(
            {
                "seed": seed,
                "train_dataset": args.train_dataset,
                "test_dataset": args.test_dataset,
                "model_path": model_path,
            }
        )

        output_path = os.path.join(
            args.output_dir,
            f"train{args.train_dataset}_seed{seed}_test{args.test_dataset}.json",
        )

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved: {output_path}")

    print("\nAll evaluations completed successfully.")


if __name__ == "__main__":
    main()
