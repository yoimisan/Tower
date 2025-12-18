# model_a_video_pred.py
# PyTorch >= 2.0, torchvision >= 0.15

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

plt.switch_backend("Agg")

# -------------------------
# Utils
# -------------------------
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time-step embedding (like diffusion / transformer)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] integer or float timesteps
        return: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -------------------------
# Dataset (replace with yours)
# -------------------------
class TowerCollapseDataset(Dataset):
    """
    Expected per sample:
      - x0: initial frame, torch.FloatTensor [3, H, W] in [0,1]
      - y_seq: future frames, torch.FloatTensor [T, 3, H, W] in [0,1]
      - y_collapse: torch.FloatTensor scalar (0/1)
    """

    def __init__(self, root_dir: str, image_size: int = 128):
        """
        root_dir: 形如 'output' 的目录，内部为 Blender 渲染生成的数据：
          - output/cubes_4/0/...
          - output/cubes_4/1/...
          - output/cubes_5/0/...
          - ...
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.root_dir}")

        self.image_size = image_size

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        # 收集所有样本：(frame_paths_list, collapse_label)
        raw_samples: List[Tuple[List[Path], float]] = []

        # for cubes_dir in sorted(self.root_dir.glob("cubes_*")):
        #     if not cubes_dir.is_dir():
        #         continue
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            meta_path = scene_dir / "meta.json"
            if not meta_path.exists():
                continue

            # 读取标签：collapse_state -> 0/1
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            collapse_state = meta.get("collapse_state", "stable")
            y_c = 1.0 if collapse_state == "collapsed" else 0.0

            frame_paths = sorted(scene_dir.glob("frame_*.png"))
            # 至少需要 2 帧（1 帧做输入，剩余做预测目标）
            if len(frame_paths) < 2:
                continue

            raw_samples.append((frame_paths, y_c))

        if not raw_samples:
            raise RuntimeError(
                f"在 {self.root_dir} 下未找到任何有效样本（包含 meta.json 与 frame_*.png）"
            )

        # 为了保证所有样本长度一致，取所有场景中最短的帧数
        min_frames = min(len(frames) for frames, _ in raw_samples)
        if min_frames < 2:
            raise RuntimeError("所有样本帧数都小于 2，无法构建时间序列数据")

        # 使用统一长度：total_frames，模型的 T = total_frames - 1
        self.total_frames = min_frames
        self.T = self.total_frames - 1

        self.samples: List[Tuple[List[Path], float]] = []
        for frames, y_c in raw_samples:
            self.samples.append((frames[: self.total_frames], y_c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, y_c = self.samples[idx]

        frames = []
        for p in frame_paths:
            with Image.open(p) as img:
                img = img.convert("RGB")
                frames.append(self.tf(img))

        # [total_frames, 3, H, W]
        frames_tensor = torch.stack(frames, dim=0)

        # 第 0 帧作为条件输入 x0，后面的作为预测目标序列
        x0 = frames_tensor[0]  # [3, H, W]
        y_seq = frames_tensor[1:]  # [T, 3, H, W]，其中 T = total_frames - 1

        y_collapse = torch.tensor(float(y_c), dtype=torch.float32)
        return x0, y_seq, y_collapse


# -------------------------
# Model components
# -------------------------
class ResNetEncoder(nn.Module):
    """
    Encode a single frame into a feature map [B, C, h, w].
    Using ResNet18 trunk without avgpool/fc.
    """

    def __init__(self, out_channels=256):
        super().__init__()
        base = resnet18(weights=None)
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        # ResNet18 layer4 output is 512 channels
        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)  # [B, 512, h, w] (h,w ~ H/32)
        feat = self.proj(feat)  # [B, C, h, w]
        return feat


class ConvDecoder(nn.Module):
    """
    Decode feature map [B, C, h, w] back to RGB [B, 3, H, W].
    Simple upsampling decoder (lightweight).
    """

    def __init__(self, in_channels=256, out_size=128):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        # (Optional) enforce exact size
        if x.shape[-1] != self.out_size or x.shape[-2] != self.out_size:
            x = F.interpolate(
                x,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class TemporalTransformerPredictor(nn.Module):
    """
    Core idea:
      - Encode x0 into tokens (spatial tokens).
      - Create T "future query tokens" (one per future step).
      - Concatenate [future_tokens, spatial_tokens] -> Transformer Encoder.
      - Read out future_tokens -> project back to feature maps -> decode to frames.
      - Collapse head from pooled tokens.
    """

    def __init__(
        self,
        image_size=128,
        feat_channels=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        T=10,
    ):
        super().__init__()
        self.image_size = image_size
        self.feat_channels = feat_channels
        self.T = T

        # CNN encoder
        self.encoder = ResNetEncoder(out_channels=feat_channels)

        # Will flatten [B, C, h, w] -> [B, N, C]
        # h,w depends on ResNet stride (~H/32)
        self.token_ln = nn.LayerNorm(feat_channels)

        # Future query tokens (learned) + time embedding
        self.future_queries = nn.Parameter(torch.randn(T, feat_channels) * 0.02)
        self.time_emb = SinusoidalPosEmb(feat_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.SiLU(),
            nn.Linear(feat_channels, feat_channels),
        )

        # Transformer encoder over (T + N) tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Map future tokens back to feature maps.
        # We'll predict a global feature map per future step by "broadcasting" token to (h,w) via a small MLP,
        # then reshape to [B, C, h, w].
        self.future_to_map = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.GELU(),
            nn.Linear(feat_channels, feat_channels),
        )

        # Decoder
        self.decoder = ConvDecoder(in_channels=feat_channels, out_size=image_size)

        # Collapse head (binary)
        self.collapse_head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.GELU(),
            nn.Linear(feat_channels, 1),
        )

    def forward(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x0: [B, 3, H, W]
        returns:
          pred_seq: [B, T, 3, H, W]
          collapse_logit: [B] (raw logit)
        """
        B = x0.shape[0]
        feat = self.encoder(x0)  # [B, C, h, w]
        B, C, h, w = feat.shape
        N = h * w

        spatial_tokens = feat.flatten(2).transpose(1, 2)  # [B, N, C]
        spatial_tokens = self.token_ln(spatial_tokens)

        # Build future tokens with time embedding (t=1..T)
        t = torch.arange(1, self.T + 1, device=x0.device)  # [T]
        t_emb = self.time_mlp(self.time_emb(t))  # [T, C]
        future_tokens = self.future_queries + t_emb  # [T, C]
        future_tokens = future_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, T, C]

        tokens = torch.cat([future_tokens, spatial_tokens], dim=1)  # [B, T+N, C]
        tokens = self.transformer(tokens)  # [B, T+N, C]

        out_future = tokens[:, : self.T, :]  # [B, T, C]
        out_spatial = tokens[:, self.T :, :]  # [B, N, C]

        # Collapse logit: pool (either future or spatial; here use pooled spatial)
        pooled = out_spatial.mean(dim=1)  # [B, C]
        collapse_logit = self.collapse_head(pooled).squeeze(-1)  # [B]

        # For each future step token -> feature map -> decode
        # Simple way: token -> [B, C] -> expand to [B, C, h, w]
        maps = self.future_to_map(out_future)  # [B, T, C]
        maps = maps.view(B, self.T, C, 1, 1).expand(B, self.T, C, h, w)  # [B,T,C,h,w]

        # Decode each step
        pred_frames = []
        for i in range(self.T):
            pred_frames.append(self.decoder(maps[:, i]))  # [B, 3, H, W]
        pred_seq = torch.stack(pred_frames, dim=1)  # [B, T, 3, H, W]

        return pred_seq, collapse_logit


# -------------------------
# Train / Eval
# -------------------------
@dataclass
class TrainConfig:
    image_size: int = 128
    T: int = 10  # 将在构建数据集后，根据实际帧数自动覆盖
    batch_size: int = 8
    lr: float = 3e-4
    epochs: int = 10
    lambda_cls: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, opt, cfg: TrainConfig):
    model.train()
    total_loss = 0.0

    for x0, y_seq, y_c in loader:
        x0 = x0.to(cfg.device)
        y_seq = y_seq.to(cfg.device)
        y_c = y_c.to(cfg.device)

        pred_seq, logit = model(x0)

        # video prediction loss
        loss_vid = F.l1_loss(pred_seq, y_seq)

        # collapse classification loss
        loss_cls = F.binary_cross_entropy_with_logits(logit, y_c)

        loss = loss_vid + cfg.lambda_cls * loss_cls

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * x0.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, cfg: TrainConfig):
    model.eval()
    total_loss = 0.0
    total_vid = 0.0
    correct = 0
    n = 0

    for x0, y_seq, y_c in loader:
        x0 = x0.to(cfg.device)
        y_seq = y_seq.to(cfg.device)
        y_c = y_c.to(cfg.device)

        pred_seq, logit = model(x0)
        loss_vid = F.l1_loss(pred_seq, y_seq)
        loss_cls = F.binary_cross_entropy_with_logits(logit, y_c)
        loss = loss_vid + cfg.lambda_cls * loss_cls
        total_loss += loss.item() * x0.size(0)

        total_vid += F.l1_loss(pred_seq, y_seq, reduction="sum").item()

        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).float()
        correct += (pred == y_c).sum().item()
        n += x0.size(0)

    avg_loss = total_loss / len(loader.dataset)
    mae = total_vid / (n * cfg.T * 3 * cfg.image_size * cfg.image_size)
    acc = correct / n
    return avg_loss, mae, acc


def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_maes: List[float],
    val_accs: List[float],
    save_path: Path,
):
    epochs = list(range(1, len(train_losses) + 1))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_maes, label="val_mae")
    plt.plot(epochs, val_accs, label="val_acc")
    plt.xlabel("epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    cfg = TrainConfig()

    # 数据根目录：默认认为脚本在 src 下运行，output 在工程根目录
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "all_data"
    print(data_root)

    # 使用真实 Blender 渲染数据构建数据集
    ds = TowerCollapseDataset(root_dir=str(data_root), image_size=cfg.image_size)

    # 根据数据集的总帧数自动设置时间步数 T（未来帧数）
    cfg.T = ds.T

    # 按 8:2 划分训练 / 验证集
    n_total = len(ds)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_train = n_total - n_val

    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = TemporalTransformerPredictor(
        image_size=cfg.image_size,
        feat_channels=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        T=cfg.T,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)

    train_losses = []
    val_losses = []
    val_maes = []
    val_accs = []

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, cfg)
        val_loss, mae, acc = eval_one_epoch(model, val_loader, cfg)
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_maes.append(mae)
        val_accs.append(acc)
        print(
            f"Epoch {epoch:02d} | train_loss={loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mae={mae:.6f} | val_acc={acc:.3f}"
        )

    curve_path = project_root / "tf_training_curves.png"
    plot_curves(train_losses, val_losses, val_maes, val_accs, curve_path)
    print(f"Saved: {curve_path}")

    model_path = project_root / "model_a.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
