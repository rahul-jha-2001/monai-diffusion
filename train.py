"""
DDIM Training Script — Brain MRI 2D Slice Diffusion Model
----------------------------------------------------------
Trains a 2D unconditional diffusion model on skull-stripped
T1 brain MRI slices using MONAI's DiffusionModelUNet + DDIMScheduler.

Logs are written to logs/train_<timestamp>.log at DEBUG level.
Console shows INFO and above only.

Usage:
    python3.12 train.py
    python3.12 train.py --epochs 100 --batch_size 16 --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler

from dataset import build_datasets


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    File handler  → DEBUG  (everything: timings, shapes, loss per step)
    Console handler → INFO  (epoch summaries, checkpoints, warnings)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"train_{ts}.log"

    fmt = logging.Formatter(
        fmt     = "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ── Timing context manager ────────────────────────────────────────────────────

class Timer:
    def __init__(self, label: str, logger: logging.Logger) -> None:
        self.label  = label
        self.logger = logger

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        self.logger.debug(f"[START] {self.label}")
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self.t0
        self.logger.debug(f"[END]   {self.label}  ->  {self.elapsed:.3f}s")


# ── Config ────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--num_timesteps", type=int,   default=1000)
    p.add_argument("--ddim_steps",    type=int,   default=50,
                   help="Inference steps used during sample generation at end of training")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--save_dir",      type=str,   default="checkpoints")
    p.add_argument("--save_every",    type=int,   default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--val_every",     type=int,   default=5,
                   help="Run validation every N epochs")
    p.add_argument("--log_dir",       type=str,   default="logs")
    p.add_argument("--log_every",     type=int,   default=50,
                   help="Log per-step debug info every N steps")
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(logger: logging.Logger) -> DiffusionModelUNet:
    logger.debug(
        "Building DiffusionModelUNet: spatial_dims=2  channels=(128,256,256,512)  "
        "attention_levels=(F,F,T,T)  num_res_blocks=(2,2,2,2)"
    )
    model = DiffusionModelUNet(
        spatial_dims      = 2,
        in_channels       = 1,
        out_channels      = 1,
        channels          = (128, 256, 256, 512),
        attention_levels  = (False, False, True, True),
        num_res_blocks    = (2, 2, 2, 2),
        num_head_channels = 32,
        norm_num_groups   = 32,
    )
    n_params     = sum(p.numel() for p in model.parameters())
    n_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built — total params: {n_params:,}  trainable: {n_trainable:,}")
    return model


# ── Training helpers ──────────────────────────────────────────────────────────

def training_step(
    model:     DiffusionModelUNet,
    scheduler: DDIMScheduler,
    batch:     dict,
    device:    torch.device,
    logger:    logging.Logger,
    step:      int,
    log_every: int,
) -> torch.Tensor:
    """
    Standard DDPM training step:
      1. Sample random timesteps
      2. Sample Gaussian noise
      3. Forward diffuse x_t
      4. Predict noise
      5. MSE loss
    """
    x0 = batch["slice"].to(device)   # (B, 1, 256, 256)

    t = torch.randint(
        0, scheduler.num_train_timesteps,
        (x0.shape[0],),
        device=device,
    ).long()

    noise = torch.randn_like(x0)
    x_t   = scheduler.add_noise(original_samples=x0, noise=noise, timesteps=t)

    noise_pred = model(x=x_t, timesteps=t)
    loss       = F.mse_loss(noise_pred, noise)

    if step % log_every == 0:
        logger.debug(
            f"  train_step={step:>6}  "
            f"t_range=[{t.min().item()}, {t.max().item()}]  "
            f"x0_range=[{x0.min():.3f}, {x0.max():.3f}]  "
            f"xt_range=[{x_t.min():.3f}, {x_t.max():.3f}]  "
            f"noise_pred_range=[{noise_pred.min():.3f}, {noise_pred.max():.3f}]  "
            f"loss={loss.item():.6f}"
        )

    return loss


@torch.no_grad()
def validation_step(
    model:     DiffusionModelUNet,
    scheduler: DDIMScheduler,
    loader:    DataLoader,
    device:    torch.device,
    logger:    logging.Logger,
) -> float:
    """Compute mean MSE loss over the entire validation set."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    t0 = time.perf_counter()

    logger.debug(f"Validation: {len(loader)} batches")

    for i, batch in enumerate(loader):
        x0    = batch["slice"].to(device)
        t     = torch.randint(
            0, scheduler.num_train_timesteps,
            (x0.shape[0],), device=device,
        ).long()
        noise = torch.randn_like(x0)
        x_t   = scheduler.add_noise(x0, noise, t)
        pred  = model(x=x_t, timesteps=t)
        loss  = F.mse_loss(pred, noise).item()
        total_loss += loss
        n_batches  += 1

        if i % 20 == 0:
            logger.debug(
                f"  val_batch={i:>4}/{len(loader)}  "
                f"batch_loss={loss:.6f}  "
                f"running_mean={total_loss/n_batches:.6f}"
            )

    val_loss = total_loss / max(n_batches, 1)
    elapsed  = time.perf_counter() - t0
    logger.debug(f"Validation complete: mean_loss={val_loss:.6f}  time={elapsed:.2f}s")
    return val_loss


@torch.no_grad()
def sample_images(
    model:      DiffusionModelUNet,
    scheduler:  DDIMScheduler,
    device:     torch.device,
    logger:     logging.Logger,
    n_samples:  int = 4,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    Generate n_samples images using DDIM.
    Returns (n_samples, 1, 256, 256) in [0, 1].
    """
    model.eval()
    logger.info(f"Generating {n_samples} samples  ddim_steps={ddim_steps}")

    with Timer("scheduler.set_timesteps", logger):
        scheduler.set_timesteps(num_inference_steps=ddim_steps)

    x = torch.randn(n_samples, 1, 256, 256, device=device)
    logger.debug(f"Initial noise: shape={tuple(x.shape)}")

    t0 = time.perf_counter()
    for i, t in enumerate(scheduler.timesteps):
        t_batch    = t.unsqueeze(0).expand(n_samples).to(device)
        noise_pred = model(x=x, timesteps=t_batch)
        x, _       = scheduler.step(noise_pred, t, x)
        if i % 10 == 0 or i == ddim_steps - 1:
            logger.debug(
                f"  sample_step [{i+1:>3}/{ddim_steps}]  "
                f"t={t.item():>4}  x_range=[{x.min():.3f}, {x.max():.3f}]"
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Sampling done: {elapsed:.2f}s  ({elapsed/ddim_steps*1000:.1f}ms/step)"
    )
    return x.clamp(0.0, 1.0)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    save_dir:  Path,
    epoch:     int,
    model:     DiffusionModelUNet,
    optimizer: torch.optim.Optimizer,
    val_loss:  float,
    logger:    logging.Logger,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"epoch_{epoch:04d}.pt"
    with Timer(f"torch.save checkpoint -> {path}", logger):
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss":  val_loss,
        }, path)
    logger.info(f"Checkpoint saved -> {path}  (val_loss={val_loss:.6f})")


def load_latest_checkpoint(
    save_dir:  Path,
    model:     DiffusionModelUNet,
    optimizer: torch.optim.Optimizer,
    logger:    logging.Logger,
) -> int:
    """Load most recent checkpoint if available. Returns starting epoch."""
    ckpts = sorted(save_dir.glob("epoch_*.pt"))
    logger.debug(f"Found {len(ckpts)} checkpoint(s) in {save_dir}")
    if not ckpts:
        logger.info("No existing checkpoints — starting from epoch 0")
        return 0
    latest = ckpts[-1]
    logger.info(f"Resuming from checkpoint: {latest}")
    with Timer(f"torch.load {latest}", logger):
        ckpt = torch.load(latest, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start = ckpt["epoch"] + 1
    logger.info(f"Resumed from epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}  -> starting epoch {start}")
    return start


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> None:
    args   = get_args()
    logger = setup_logging(args.log_dir)

    t_main = time.perf_counter()
    logger.info("=" * 70)
    logger.info("Training started")
    logger.info(f"Args: {vars(args)}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.debug(f"CUDA total memory: {total_mem:.1f}GB")

    # ── Seeds ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.debug(f"Seeds set: torch={args.seed}  numpy={args.seed}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Building datasets...")
    with Timer("build_datasets", logger):
        train_ds, val_ds, test_ds = build_datasets(seed=args.seed)

    logger.info(
        f"Dataset sizes — train: {len(train_ds):,}  "
        f"val: {len(val_ds):,}  test: {len(test_ds):,}"
    )

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
    logger.debug(
        f"DataLoaders created — "
        f"train_batches={len(train_loader)}  val_batches={len(val_loader)}  "
        f"batch_size={args.batch_size}  num_workers={args.num_workers}  "
        f"pin_memory={device.type == 'cuda'}"
    )

    # First batch shape sanity check
    sample_batch = next(iter(train_loader))
    logger.debug(
        f"Sample batch — "
        f"slice.shape={tuple(sample_batch['slice'].shape)}  "
        f"dtype={sample_batch['slice'].dtype}  "
        f"value_range=[{sample_batch['slice'].min():.3f}, {sample_batch['slice'].max():.3f}]"
    )

    # ── Model, scheduler, optimizer ───────────────────────────────────────────
    with Timer("build_model + .to(device)", logger):
        model = build_model(logger).to(device)

    scheduler = DDIMScheduler(num_train_timesteps=args.num_timesteps)
    logger.debug(f"DDIMScheduler: num_train_timesteps={args.num_timesteps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger.debug(f"Optimizer: AdamW  lr={args.lr}")

    scaler = torch.GradScaler(device.type, enabled=device.type == "cuda")
    logger.debug(f"GradScaler: enabled={device.type == 'cuda'}")

    if device.type == "cuda":
        logger.debug(
            f"CUDA memory after model init: "
            f"{torch.cuda.memory_allocated()/1e6:.1f}MB allocated  "
            f"{torch.cuda.memory_reserved()/1e6:.1f}MB reserved"
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    save_dir    = Path(args.save_dir)
    start_epoch = load_latest_checkpoint(save_dir, model, optimizer, logger)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(
        f"Starting training: epochs={args.epochs}  "
        f"start_epoch={start_epoch}  "
        f"batch_size={args.batch_size}  lr={args.lr}"
    )

    best_val_loss = float("inf")
    global_step   = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss   = 0.0
        t_epoch      = time.perf_counter()
        t_data_total = 0.0
        t_fwd_total  = 0.0
        t_bwd_total  = 0.0

        logger.debug(f"Epoch {epoch+1}/{args.epochs} — {len(train_loader)} batches")

        t_data_start = time.perf_counter()
        for step, batch in enumerate(train_loader):
            t_data_total += time.perf_counter() - t_data_start

            optimizer.zero_grad()

            # Forward
            t_fwd = time.perf_counter()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = training_step(
                    model, scheduler, batch, device, logger,
                    step=global_step, log_every=args.log_every,
                )
            t_fwd_total += time.perf_counter() - t_fwd

            # Backward
            t_bwd = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            t_bwd_total += time.perf_counter() - t_bwd

            epoch_loss  += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                logger.debug(
                    f"  step [{step+1:>5}/{len(train_loader)}]  "
                    f"loss={loss.item():.6f}  "
                    f"grad_norm={grad_norm:.4f}  "
                    f"data_ms={t_data_total/(step+1)*1000:.1f}  "
                    f"fwd_ms={t_fwd_total/(step+1)*1000:.1f}  "
                    f"bwd_ms={t_bwd_total/(step+1)*1000:.1f}"
                )

            t_data_start = time.perf_counter()

        epoch_loss /= len(train_loader)
        epoch_time  = time.perf_counter() - t_epoch

        logger.debug(
            f"Epoch {epoch+1} timing breakdown — "
            f"total={epoch_time:.1f}s  "
            f"data={t_data_total:.1f}s ({t_data_total/epoch_time*100:.1f}%)  "
            f"fwd={t_fwd_total:.1f}s ({t_fwd_total/epoch_time*100:.1f}%)  "
            f"bwd={t_bwd_total:.1f}s ({t_bwd_total/epoch_time*100:.1f}%)"
        )

        if device.type == "cuda":
            logger.debug(
                f"CUDA memory end of epoch {epoch+1}: "
                f"{torch.cuda.memory_allocated()/1e6:.1f}MB allocated  "
                f"{torch.cuda.max_memory_allocated()/1e6:.1f}MB peak"
            )

        # ── Validation ────────────────────────────────────────────────────────
        val_loss = float("nan")
        improved = False
        if (epoch + 1) % args.val_every == 0:
            logger.debug(f"Running validation at epoch {epoch+1}")
            with Timer(f"validation epoch {epoch+1}", logger):
                val_loss = validation_step(model, scheduler, val_loader, device, logger)
            improved = val_loss < best_val_loss
            flag     = " *" if improved else ""
            logger.info(
                f"Epoch [{epoch+1:>4}/{args.epochs}]  "
                f"train_loss={epoch_loss:.6f}  "
                f"val_loss={val_loss:.6f}{flag}  "
                f"time={epoch_time:.1f}s"
            )
            if improved:
                best_val_loss = val_loss
                save_checkpoint(save_dir, epoch + 1, model, optimizer, val_loss, logger)
        else:
            logger.info(
                f"Epoch [{epoch+1:>4}/{args.epochs}]  "
                f"train_loss={epoch_loss:.6f}  "
                f"time={epoch_time:.1f}s"
            )

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            logger.debug(f"Periodic checkpoint at epoch {epoch+1}")
            save_checkpoint(save_dir, epoch + 1, model, optimizer, val_loss, logger)

        model.train()

    # ── End of training ───────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_main
    logger.info(f"Training complete — best val loss: {best_val_loss:.6f}")
    logger.info(f"Total training time: {total_time:.1f}s  ({total_time/3600:.2f}h)")

    # ── Final sample generation ───────────────────────────────────────────────
    logger.info("Generating sample images from final model...")
    with Timer("final sample generation", logger):
        samples = sample_images(
            model, scheduler, device, logger,
            n_samples=4, ddim_steps=args.ddim_steps,
        )
    logger.info(
        f"Generated samples — shape={tuple(samples.shape)}  "
        f"value_range=[{samples.min():.3f}, {samples.max():.3f}]"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()