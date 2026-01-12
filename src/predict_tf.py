# model_a_video_pred.py
# PyTorch >= 2.0, torchvision >= 0.15

import math
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
from tqdm import tqdm
import torchvision

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

class TowerCollapseDataset(Dataset):
    """
    Preload version: 所有图片在 __init__ 中一次性读入内存
    每个样本返回：
      - x0: [3, H, W]
      - y_seq: [T, 3, H, W]
      - y_collapse: scalar (0/1)
    """

    def __init__(self, root_dir: str, image_size: int = 128):
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

        # ---------------------------------
        # Step 1: 收集所有样本路径和标签
        # ---------------------------------
        raw_samples: List[Tuple[List[Path], float]] = []

        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            meta_path = scene_dir / "meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            collapse_state = meta.get("collapse_state", "stable")
            y_c = 1.0 if collapse_state == "collapsed" else 0.0

            frame_paths = sorted(scene_dir.glob("frame_*.png"))
            if len(frame_paths) < 2:
                continue

            raw_samples.append((frame_paths, y_c))

        if not raw_samples:
            raise RuntimeError(f"在 {self.root_dir} 下未找到任何有效样本")

        # ---------------------------------
        # Step 2: 统一时间长度
        # ---------------------------------
        min_frames = 2 #min(len(frames) for frames, _ in raw_samples)
        if min_frames < 2:
            raise RuntimeError("所有样本帧数都小于 2，无法构建时间序列数据")

        self.total_frames = min_frames
        self.T = self.total_frames - 1

        # ---------------------------------
        # Step 3: 预加载所有图片
        # ---------------------------------
        self.data = []  # [(x0, y_seq, y_collapse), ...]

        print(f"[Dataset] Preloading {len(raw_samples)} samples into memory...")

        for frame_paths, y_c in tqdm(raw_samples):
            frame_paths = frame_paths[: self.total_frames]

            frames = []
            for p in frame_paths:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    frames.append(self.tf(img))

            frames_tensor = torch.stack(frames, dim=0)  # [total_frames, 3, H, W]

            x0 = frames_tensor[0]          # [3, H, W]
            y_seq = frames_tensor[1:]     # [T, 3, H, W]
            y_collapse = torch.tensor(float(y_c), dtype=torch.float32)

            self.data.append((x0, y_seq, y_collapse))

        print(f"[Dataset] Done. Total frames per sample = {self.total_frames}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


class TemporalDiscriminator(nn.Module):
    """Patch-based discriminator operating on concatenated temporal frames."""

    def __init__(self, in_channels: int, base_channels: int = 64, n_layers: int = 4):
        super().__init__()
        layers = []
        channels = base_channels
        layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, n_layers):
            in_ch = channels
            out_ch = min(base_channels * (2 ** i), 512)
            stride = 2 if i < n_layers - 1 else 1
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1))
            layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = out_ch

        self.features = nn.Sequential(*layers)
        self.final = nn.Conv2d(channels, 1, kernel_size=4, padding=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.features(frames)
        x = self.final(x)
        return x.mean(dim=(1, 2, 3))


# -------------------------
# Train / Eval
# -------------------------
@dataclass
class TrainConfig:
    image_size: int = 256
    T: int = 10  # 将在构建数据集后，根据实际帧数自动覆盖
    batch_size: int = 32
    lr: float = 1e-5
    epochs: int = 50
    lambda_cls: float = 0.5
    lambda_adv: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def _concat_temporal_frames(x0: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    frames = torch.cat([x0.unsqueeze(1), seq], dim=1)
    B, steps, C, H, W = frames.shape
    return frames.reshape(B, steps * C, H, W)

def compute_foreground_mask(img, thresh=5):
    """
    img: [B, T, 3, H, W] in [0,1]
    returns: [B, T, 1, H, W] binary mask
    """
    # 灰色背景: R≈G≈B
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    diff = (r - g).abs() + (r - b).abs() + (g - b).abs()
    mask = (diff > thresh).float()
    return mask.unsqueeze(2)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = torchvision.models.vgg11(weights="IMAGENET1K_V1").features

        # 取前 3 个 block（共 8 个 conv）
        self.blocks = nn.ModuleList([
            vgg[:4],    # conv1_1, relu, conv1_2, relu
            vgg[4:9],   # maxpool + conv2_1, relu, conv2_2, relu
            vgg[9:16],  # maxpool + conv3_1, relu, conv3_2, relu
        ])

        for b in self.blocks:
            b.eval()
            for p in b.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        """
        x, y: [B,3,H,W] in [0,1]
        """
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)

        return loss


def train_one_epoch_gan(
    model: TemporalTransformerPredictor,
    discriminator: TemporalDiscriminator,
    loader: DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    cfg: TrainConfig,
):
    model.train()
    discriminator.train()
    total_g = 0.0
    total_d = 0.0
    perceptual = VGGPerceptualLoss().to(cfg.device)


    for x0, y_seq, y_c in loader:
        x0 = x0.to(cfg.device)
        y_seq = y_seq.to(cfg.device)
        y_c = y_c.to(cfg.device)

        real_video = _concat_temporal_frames(x0, y_seq)

        # Update discriminator
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_seq_detached, _ = model(x0)
        fake_video = _concat_temporal_frames(x0, fake_seq_detached)
        real_score = discriminator(real_video)
        fake_score = discriminator(fake_video)
        loss_d_real = F.relu(1 - real_score).mean()
        loss_d_fake = F.relu(1 + fake_score).mean()
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.grad_clip)
        opt_d.step()

        # Update generator
        opt_g.zero_grad(set_to_none=True)
        pred_seq, logit = model(x0)
        fake_video_for_g = _concat_temporal_frames(x0, pred_seq)
        adv_score = discriminator(fake_video_for_g)
        
        loss_perc = 0.0
        for t in range(pred_seq.size(1)):
            loss_perc += perceptual(pred_seq[:,t], y_seq[:,t])
        loss_perc /= pred_seq.size(1)

        mask = compute_foreground_mask(y_seq)
        loss_l1 = (F.l1_loss(pred_seq, y_seq, reduction='none') * mask).mean()
        # import ipdb; ipdb.set_trace()

        # loss_l1 = F.l1_loss(pred_seq, y_seq)
        loss_adv = -adv_score.mean()
        loss_cls = F.binary_cross_entropy_with_logits(logit, y_c)
        
        # loss_g = loss_l1 + cfg.lambda_adv * loss_adv + cfg.lambda_cls * loss_cls
        loss_g = (
            1.0 * loss_l1 +
            0.2 * loss_perc +
            0.1 * loss_adv +
            0.5 * loss_cls
        )
        
        
        loss_g.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt_g.step()

        total_g += loss_g.item() * x0.size(0)
        total_d += loss_d.item() * x0.size(0)

    denom = len(loader.dataset)
    return total_g / denom, total_d / denom


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
    cfg.T = 1#ds.T

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

    discriminator = TemporalDiscriminator(in_channels=3 * (cfg.T + 1)).to(cfg.device)

    opt_g = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.lr, weight_decay=1e-2)

    train_losses = []
    val_losses = []
    val_maes = []
    val_accs = []

    for epoch in range(1, cfg.epochs + 1):
        train_g_loss, train_d_loss = train_one_epoch_gan(
            model, discriminator, train_loader, opt_g, opt_d, cfg
        )
        val_loss, mae, acc = eval_one_epoch(model, val_loader, cfg)
        train_losses.append(train_g_loss)
        val_losses.append(val_loss)
        val_maes.append(mae)
        val_accs.append(acc)
        print(
            f"Epoch {epoch:02d} | train_G={train_g_loss:.4f} | train_D={train_d_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_mae={mae:.6f} | val_acc={acc:.3f}"
        )

    curve_path = project_root / "tf_training_curves2.png"
    plot_curves(train_losses, val_losses, val_maes, val_accs, curve_path)
    print(f"Saved: {curve_path}")

    model_path = project_root / "model_b.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
