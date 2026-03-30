"""
DDIM Training Script — Brain MRI 2D Slice Diffusion Model
----------------------------------------------------------
Trains a 2D unconditional diffusion model on skull-stripped
T1 brain MRI slices using MONAI's DiffusionModelUNet + DDIMScheduler.

Usage:
    python train.py
    python train.py --epochs 100 --batch_size 16 --lr 1e-4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler

from dataset import build_datasets

# ── Config ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_timesteps",type=int,   default=1000)
    p.add_argument("--ddim_steps",   type=int,   default=50,
                   help="Inference steps (DDIM, fewer than training timesteps)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--save_dir",     type=str,   default="checkpoints")
    p.add_argument("--save_every",   type=int,   default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--val_every",    type=int,   default=5,
                   help="Run validation every N epochs")
    return p.parse_args()


# ── Model ────────────────────────────────────────────────────────────────────

def build_model() -> DiffusionModelUNet:
    """
    2D U-Net denoiser for 256×256 single-channel images.
    Architecture follows standard DDPM/DDIM practice for 256px images:
      - 4 resolution levels: 256 → 128 → 64 → 32
      - Attention at 32px and 64px levels
      - 128 base channels
    """
    return DiffusionModelUNet(
        spatial_dims      = 2,
        in_channels       = 1,
        out_channels      = 1,
        channels          = (128, 256, 256, 512),
        attention_levels  = (False, False, True, True),
        num_res_blocks    = (2, 2, 2, 2),
        num_head_channels = 32,
        norm_num_groups   = 32,
    )


# ── Training helpers ─────────────────────────────────────────────────────────

def training_step(
    model:     DiffusionModelUNet,
    scheduler: DDIMScheduler,
    batch:     dict,
    device:    torch.device,
) -> torch.Tensor:
    """
    Standard DDPM training step:
      1. Sample random timesteps
      2. Sample Gaussian noise
      3. Forward diffuse: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
      4. Predict noise with model
      5. MSE loss between predicted and actual noise
    """
    x0 = batch["slice"].to(device)                         # (B, 1, 256, 256)

    # Sample random timestep for each image in the batch
    t = torch.randint(
        0, scheduler.num_train_timesteps,
        (x0.shape[0],),
        device=device,
    ).long()

    # Sample noise and create noisy image
    noise  = torch.randn_like(x0)
    x_t    = scheduler.add_noise(
        original_samples = x0,
        noise            = noise,
        timesteps        = t,
    )

    # Predict noise
    noise_pred = model(x=x_t, timesteps=t)

    return F.mse_loss(noise_pred, noise)


@torch.no_grad()
def validation_step(
    model:     DiffusionModelUNet,
    scheduler: DDIMScheduler,
    loader:    DataLoader,
    device:    torch.device,
) -> float:
    """Compute mean MSE loss over the entire validation set."""
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        x0    = batch["slice"].to(device)
        t     = torch.randint(
            0, scheduler.num_train_timesteps,
            (x0.shape[0],), device=device,
        ).long()
        noise = torch.randn_like(x0)
        x_t   = scheduler.add_noise(x0, noise, t)
        pred  = model(x=x_t, timesteps=t)
        total_loss += F.mse_loss(pred, noise).item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def sample_images(
    model:      DiffusionModelUNet,
    scheduler:  DDIMScheduler,
    device:     torch.device,
    n_samples:  int = 4,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    Generate n_samples images using DDIM with ddim_steps inference steps.
    Returns tensor of shape (n_samples, 1, 256, 256) in [0, 1].
    """
    model.eval()
    scheduler.set_timesteps(num_inference_steps=ddim_steps)

    x = torch.randn(n_samples, 1, 256, 256, device=device)

    for t in scheduler.timesteps:
        t_batch    = t.unsqueeze(0).expand(n_samples).to(device)
        noise_pred = model(x=x, timesteps=t_batch)
        x, _       = scheduler.step(noise_pred, t, x)

    # Clamp to [0, 1] — model output is in normalised range
    return x.clamp(0.0, 1.0)


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    save_dir:  Path,
    epoch:     int,
    model:     DiffusionModelUNet,
    optimizer: torch.optim.Optimizer,
    val_loss:  float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"epoch_{epoch:04d}.pt"
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)
    print(f"  Saved checkpoint → {path}")


def load_latest_checkpoint(
    save_dir:  Path,
    model:     DiffusionModelUNet,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load most recent checkpoint if available. Returns starting epoch."""
    ckpts = sorted(save_dir.glob("epoch_*.pt"))
    if not ckpts:
        return 0
    ckpt = torch.load(ckpts[-1], weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from {ckpts[-1]}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"] + 1


# ── Main training loop ───────────────────────────────────────────────────────

def main() -> None:
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    print("\nBuilding datasets...")
    train_ds, val_ds, test_ds = build_datasets(seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
    )

    print(f"\nTrain slices : {len(train_ds):,}")
    print(f"Val slices   : {len(val_ds):,}")
    print(f"Test slices  : {len(test_ds):,}")

    # ── Model & scheduler ───────────────────────────────────────────────────
    model     = build_model().to(device)
    scheduler = DDIMScheduler(num_train_timesteps=args.num_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler    = torch.GradScaler(device.type, enabled=device.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Resume if checkpoint exists ─────────────────────────────────────────
    save_dir    = Path(args.save_dir)
    start_epoch = load_latest_checkpoint(save_dir, model, optimizer)

    # ── Training ────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs "
          f"(batch={args.batch_size}, lr={args.lr})\n")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = training_step(model, scheduler, batch, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        elapsed     = time.time() - t0

        print(f"Epoch [{epoch+1:>4}/{args.epochs}]  "
              f"train_loss={epoch_loss:.4f}  "
              f"time={elapsed:.1f}s", end="")

        # ── Validation ──────────────────────────────────────────────────────
        val_loss = float("nan")
        if (epoch + 1) % args.val_every == 0:
            val_loss = validation_step(model, scheduler, val_loader, device)
            flag     = " *" if val_loss < best_val_loss else ""
            print(f"  val_loss={val_loss:.4f}{flag}", end="")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(save_dir, epoch + 1, model, optimizer, val_loss)

        print()

        # ── Periodic checkpoint ─────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(save_dir, epoch + 1, model, optimizer, val_loss)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print("Generating sample images...")

    samples = sample_images(
        model, scheduler, device,
        n_samples=4, ddim_steps=args.ddim_steps,
    )
    print(f"Generated samples shape: {samples.shape}")
    print(f"Value range: [{samples.min():.3f}, {samples.max():.3f}]")


if __name__ == "__main__":
    main()
