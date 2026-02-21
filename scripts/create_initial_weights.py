import torch
from train import SimpleCNN, DEVICE, set_seed, SEED


def main():
    set_seed(SEED)

    model = SimpleCNN().to(DEVICE)

    torch.save(model.state_dict(), "models/initial_weights.pth")

    print("Initial weights saved to models/initial_weights.pth")


if __name__ == "__main__":
    main()
