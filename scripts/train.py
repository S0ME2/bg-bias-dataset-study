import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

IMAGE_SIZE = 224  # Image size for resizing
LEARNING_RATE = 0.001  # Learning rate for optimizer
EPOCHS = 15  # Number of training epochs
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


def load_loader(dataset_path, split, seed):
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    indices = torch.load(f"data/splits/{split}_indices.pt")
    subset = Subset(dataset, indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=0,  # safest reproducibility
        generator=generator,
    )

    return loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset (ImageFolder)")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save trained models",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for seed in TRAINING_SEEDS:
        print("\n" + "=" * 60)
        print(f"Training with seed {seed}")
        print("=" * 60)

        set_seed(seed)

        train_loader = load_loader(args.data, "train", seed)
        val_loader = load_loader(args.data, "val", seed)

        model = SimpleCNN().to(DEVICE)

        model.load_state_dict(
            torch.load("models/initial_weights.pth", map_location=DEVICE)
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc = validate(model, val_loader, criterion)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        output_path = os.path.join(args.output_dir, f"model_seed{seed}.pth")

        torch.save(model.state_dict(), output_path)
        print(f"\nSaved model to: {output_path}")

    print("\nAll seeds completed successfully.")


if __name__ == "__main__":
    main()
