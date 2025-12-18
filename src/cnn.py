import argparse
import json
import os
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.io import read_image


class TowerDataset(Dataset):
    """Dataset that loads frame_001.png and collapse labels from each sample."""

    LABEL_MAP = {"stable": 0.0, "collapsed": 1.0}

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
        self.samples: List[Tuple[str, float]] = self._gather_samples()
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.root_dir}")

    def _gather_samples(self) -> List[Tuple[str, float]]:
        samples: List[Tuple[str, float]] = []
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory {self.root_dir} not found")
        for entry in sorted(os.listdir(self.root_dir)):
            sample_dir = os.path.join(self.root_dir, entry)
            if not os.path.isdir(sample_dir):
                continue

            frame_path = os.path.join(sample_dir, "frame_0001.png")
            meta_path = os.path.join(sample_dir, "meta.json")
            if not os.path.isfile(frame_path) or not os.path.isfile(meta_path):
                continue

            with open(meta_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)

            label_name = metadata.get("collapse_state")
            if label_name not in self.LABEL_MAP:
                raise ValueError(f"Unknown collapse_state {label_name} in {meta_path}")

            samples.append((frame_path, self.LABEL_MAP[label_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_path, label = self.samples[idx]
        image = read_image(frame_path)  # Returns tensor with values in [0, 255]
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor


class TowerCollapseModel(nn.Module):
    """Static ResNet-18 baseline ending in a Sigmoid for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> None:
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (outputs >= 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / max(len(dataloader), 1)
        accuracy = 100.0 * correct / max(total, 1)
        print(
            f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%"
        )


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(images)
        preds = (outputs >= 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / max(total, 1)
    print(f"Training set accuracy: {accuracy:.2f}%")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tower collapse CNN baseline.")
    parser.add_argument(
        "--train-data-root",
        type=str,
        default="data/cubes_4",
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default="data/cubes_4",
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()

    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TowerDataset(args.train_data_root)
    test_dataset = TowerDataset(args.test_data_root)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TowerCollapseModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_dataloader, criterion, optimizer, device, args.epochs)
    evaluate(model, test_dataloader, device)


if __name__ == "__main__":
    main()
