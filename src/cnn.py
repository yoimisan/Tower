import argparse
import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
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
    print(f"Test set accuracy: {accuracy:.2f}%")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tower collapse CNN baseline.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="all_data",
        help="Path to dataset root directory. 4/5 of the data is used for training, 1/5 for testing.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split.",
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
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="model_cnn.pt",
        help="Where to save the trained model weights (state_dict).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    full_dataset = TowerDataset(args.data_root)
    print("data get ready")
    test_size = max(1, len(full_dataset) // 5)
    train_size = len(full_dataset) - test_size
    if train_size <= 0:
        raise ValueError(
            "Not enough data to create a training split. Provide at least 2 samples."
        )

    generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TowerCollapseModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_dataloader, criterion, optimizer, device, args.epochs)
    evaluate(model, test_dataloader, device)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved trained weights to {output_path}")


if __name__ == "__main__":
    main()
