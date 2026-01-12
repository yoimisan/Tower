import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import sys
import os
sys.path.append(os.path.dirname(__file__)) 

from predict_tf import (
    TrainConfig,
    TowerCollapseDataset,
    TemporalTransformerPredictor,
    eval_one_epoch,
)


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate transformer predictor.")
    parser.add_argument(
        "--model-path", default=project_root / "model_a.pt", type=Path, help="Saved weights"
    )
    parser.add_argument(
        "--data-root", default=project_root / "all_data", type=Path, help="Dataset root"
    )
    parser.add_argument(
        "--output-dir",
        default=project_root / "eval_outputs",
        type=Path,
        help="Directory to save prediction images",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="How many samples to visualize (rows of preds vs gt)",
    )
    return parser.parse_args()


def load_model(cfg: TrainConfig, model_path: Path, T: int):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    model = TemporalTransformerPredictor(
        image_size=cfg.image_size,
        feat_channels=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        T=T,
    ).to(cfg.device)

    state = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def visualize_predictions(
    model: TemporalTransformerPredictor,
    loader: DataLoader,
    cfg: TrainConfig,
    out_dir: Path,
    max_samples: int = 4,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for x0, y_seq, _ in loader:
        x0 = x0.to(cfg.device)
        y_seq = y_seq.to(cfg.device)
        pred_seq, _ = model(x0)

        for i in range(x0.size(0)):
            if saved >= max_samples:
                return

            top_row = torch.cat([x0[i].unsqueeze(0), pred_seq[i]], dim=0)
            bottom_row = torch.cat([x0[i].unsqueeze(0), y_seq[i]], dim=0)
            grid = torch.cat([top_row, bottom_row], dim=0)
            grid = make_grid(grid.cpu(), nrow=cfg.T + 1, padding=2)

            save_image(grid, out_dir / f"sample_{saved}_pred_vs_gt.png")
            saved += 1


@torch.no_grad()
def eval_one_img(model, loader, cfg: TrainConfig):
    model.eval()
    total_loss = 0.0
    total_vid = 0.0
    correct = 0
    n = 0

    for x0, _, _ in loader:
        print(x0.shape)
        x0 = x0.to(cfg.device)
        

        _, logit = model(x0)
        print(logit)
        break
        

def main():
    args = parse_args()
    cfg = TrainConfig()

    ds = TowerCollapseDataset(root_dir=str(args.data_root), image_size=cfg.image_size)
    cfg.T = 22

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = load_model(cfg, args.model_path, T=cfg.T)
    eval_one_img(model,loader,cfg)
    # val_loss, mae, acc = eval_one_epoch(model, loader, cfg)
    # print(f"Eval | loss={val_loss:.4f} | mae={mae:.6f} | acc={acc:.3f}")

    # visualize_predictions(model, loader, cfg, args.output_dir, max_samples=args.max_samples)
    # print(f"Saved prediction grids to {args.output_dir}")


if __name__ == "__main__":
    main()
