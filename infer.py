"""
Inference Script — Brain MRI 2D Slice Diffusion Model
------------------------------------------------------
Two modes:

  1. sample   — unconditional generation from pure noise
  2. sdedit   — SDEdit counterfactual: project a pathological slice
                onto the healthy manifold

Usage:
    # Unconditional generation
    python3.12 infer.py sample \\
        --ckpt checkpoints/epoch_0100.pt \\
        --n_samples 4 --ddim_steps 50 --out samples.png

    # SDEdit counterfactual
    python3.12 infer.py sdedit \\
        --ckpt checkpoints/epoch_0100.pt \\
        --nifti path/to/patient.nii.gz \\
        --slice_idx 60 \\
        --t_start 500 \\
        --ddim_steps 50 \\
        --out counterfactual.png

Logs are written to logs/infer_<timestamp>.log at DEBUG level.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Configure a logger that writes to:
      - logs/infer_<timestamp>.log  at DEBUG level (full detail)
      - stdout                      at INFO  level (progress only)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"infer_{ts}.log"

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

    logger = logging.getLogger("infer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ── Timing context manager ────────────────────────────────────────────────────

class Timer:
    """Context manager that logs elapsed time at DEBUG level."""
    def __init__(self, label: str, logger: logging.Logger) -> None:
        self.label  = label
        self.logger = logger

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        self.logger.debug(f"[START] {self.label}")
        return self

    def __exit__(self, *_) -> None:
        elapsed = time.perf_counter() - self.t0
        self.logger.debug(f"[END]   {self.label}  ->  {elapsed:.3f}s")
        self.elapsed = elapsed


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(logger: logging.Logger) -> DiffusionModelUNet:
    logger.debug("Building DiffusionModelUNet: spatial_dims=2, channels=(128,256,256,512)")
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
    n_params = sum(p.numel() for p in model.parameters())
    logger.debug(f"Model built  —  parameters: {n_params:,}")
    return model


def load_checkpoint(
    ckpt_path: str,
    device:    torch.device,
    logger:    logging.Logger,
) -> DiffusionModelUNet:
    logger.info(f"Loading checkpoint: {ckpt_path}")

    with Timer("torch.load checkpoint", logger):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    logger.debug(f"Checkpoint keys: {list(ckpt.keys())}")
    logger.info(f"Checkpoint  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

    with Timer("build_model + load_state_dict", logger):
        model = build_model(logger).to(device)
        model.load_state_dict(ckpt["model"])

    model.eval()
    logger.info("Model loaded and set to eval mode")
    return model


# ── Data helpers ──────────────────────────────────────────────────────────────

def _crop_to_brain(
    data:   np.ndarray,
    margin: int,
    logger: logging.Logger,
) -> np.ndarray:
    H, W, D = data.shape
    fills   = np.array([np.count_nonzero(data[..., z]) / (H * W) for z in range(D)])
    usable  = np.where(fills >= 0.01)[0]
    logger.debug(
        f"Brain crop: volume shape={data.shape}  "
        f"usable_slices={len(usable)}  "
        f"fill_range=[{fills.min():.3f}, {fills.max():.3f}]"
    )
    if len(usable) == 0:
        logger.warning("No usable slices found during brain crop — returning full volume")
        return data
    z_start = max(0, usable[0]  - margin)
    z_end   = min(D, usable[-1] + margin + 1)
    logger.debug(f"Brain crop: z_start={z_start}  z_end={z_end}  (margin={margin})")
    return data[..., z_start:z_end]


def _normalise_slice(slc: np.ndarray, logger: logging.Logger) -> np.ndarray:
    nonzero = slc[slc > 0]
    if len(nonzero) == 0:
        logger.warning("Normalise: slice has no non-zero voxels, returning zeros")
        return slc.astype(np.float32)
    lo = float(np.percentile(nonzero, 0.5))
    hi = float(np.percentile(nonzero, 99.5))
    logger.debug(
        f"Normalise: nonzero_voxels={len(nonzero)}  "
        f"p0.5={lo:.2f}  p99.5={hi:.2f}  "
        f"raw_range=[{slc.min():.2f}, {slc.max():.2f}]"
    )
    slc = np.clip(slc, lo, hi)
    slc = (slc - lo) / (hi - lo + 1e-8)
    logger.debug(f"Normalise: output range=[{slc.min():.4f}, {slc.max():.4f}]")
    return slc.astype(np.float32)


def load_slice(
    nifti_path: str,
    slice_idx:  int,
    logger:     logging.Logger,
    img_size:   int = 256,
) -> torch.Tensor:
    """
    Full preprocessing pipeline mirroring dataset.py:
      1. Load volume
      2. Crop to brain z-extent +/- 3 slices
      3. Resample in-plane to 1.0 mm via MONAI Spacing
      4. Pad/crop in-plane to img_size x img_size
      5. Normalise: 0.5-99.5th percentile -> [0, 1]

    Returns: (1, 1, img_size, img_size) float32 tensor in [0, 1].
    """
    from monai.transforms import Compose, EnsureChannelFirst, Spacing, ResizeWithPadOrCrop

    logger.info(f"Loading NIfTI: {nifti_path}")

    with Timer("nib.load + get_fdata", logger):
        nii      = nib.load(nifti_path)
        data     = nii.get_fdata(dtype=np.float32)
        pixdim_z = float(nii.header.get_zooms()[2])

    logger.debug(
        f"Volume loaded: shape={data.shape}  dtype={data.dtype}  "
        f"pixdim=({nii.header.get_zooms()[0]:.2f}, "
        f"{nii.header.get_zooms()[1]:.2f}, {pixdim_z:.2f})mm  "
        f"value_range=[{data.min():.2f}, {data.max():.2f}]"
    )

    with Timer("brain crop", logger):
        data = _crop_to_brain(data, margin=3, logger=logger)

    n_slices = data.shape[2]
    logger.info(f"After brain crop: {n_slices} slices remain")

    if not (0 <= slice_idx < n_slices):
        logger.error(
            f"slice_idx={slice_idx} out of range [0, {n_slices-1}] "
            f"(original had {nii.shape[2]} slices)"
        )
        raise ValueError(
            f"slice_idx {slice_idx} out of range [0, {n_slices - 1}] after brain crop"
        )

    logger.info(f"Building MONAI transform pipeline (target={img_size}px @ 1.0mm)")
    tfm = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Spacing(pixdim=(1.0, 1.0, pixdim_z), mode="bilinear"),
        ResizeWithPadOrCrop(
            spatial_size=(img_size, img_size, -1),
            mode="constant",
            constant_values=0,
        ),
    ])

    with Timer("MONAI Spacing + ResizeWithPadOrCrop", logger):
        vol = tfm(data).numpy()[0]   # (img_size, img_size, D')

    logger.debug(f"After MONAI transforms: vol.shape={vol.shape}  dtype={vol.dtype}")

    slc  = vol[..., slice_idx]
    fill = float(np.count_nonzero(slc)) / (img_size * img_size)
    logger.debug(
        f"Extracted slice {slice_idx}: "
        f"shape={slc.shape}  fill={fill:.3f}  "
        f"value_range=[{slc.min():.2f}, {slc.max():.2f}]"
    )

    if np.count_nonzero(slc) == 0:
        logger.error(f"Slice {slice_idx} is entirely zero after preprocessing")
        raise ValueError(f"Slice {slice_idx} is empty after preprocessing.")

    with Timer("normalise slice", logger):
        slc = _normalise_slice(slc, logger)

    tensor = torch.from_numpy(slc).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    logger.info(
        f"Slice ready: shape={tuple(tensor.shape)}  dtype={tensor.dtype}  "
        f"value_range=[{tensor.min():.4f}, {tensor.max():.4f}]"
    )
    return tensor


# ── Inference modes ───────────────────────────────────────────────────────────

@torch.no_grad()
def unconditional_sample(
    model:      DiffusionModelUNet,
    scheduler:  DDIMScheduler,
    device:     torch.device,
    logger:     logging.Logger,
    n_samples:  int = 4,
    ddim_steps: int = 50,
    img_size:   int = 256,
) -> torch.Tensor:
    """Generate brain MRI slices from pure Gaussian noise. Returns [0, 1]."""
    logger.info(f"Unconditional sampling: n_samples={n_samples}  ddim_steps={ddim_steps}")

    with Timer("scheduler.set_timesteps", logger):
        scheduler.set_timesteps(num_inference_steps=ddim_steps)

    logger.debug(
        f"Timesteps ({len(scheduler.timesteps)}): "
        f"{scheduler.timesteps[:5].tolist()} ... {scheduler.timesteps[-5:].tolist()}"
    )

    x = torch.randn(n_samples, 1, img_size, img_size, device=device)
    logger.debug(f"Initial noise: shape={tuple(x.shape)}  device={x.device}")

    t_total = time.perf_counter()
    for i, t in enumerate(scheduler.timesteps):
        t0_step    = time.perf_counter()
        t_batch    = t.unsqueeze(0).expand(n_samples).to(device)
        noise_pred = model(x=x, timesteps=t_batch)
        x, _       = scheduler.step(noise_pred, t, x)
        step_ms    = (time.perf_counter() - t0_step) * 1000

        if i % 10 == 0 or i == ddim_steps - 1:
            logger.debug(
                f"  step [{i+1:>3}/{ddim_steps}]  t={t.item():>4}  "
                f"x_range=[{x.min():.3f}, {x.max():.3f}]  "
                f"step_time={step_ms:.1f}ms"
            )

    total_s = time.perf_counter() - t_total
    logger.info(
        f"Sampling complete: {total_s:.2f}s total  "
        f"({total_s/ddim_steps*1000:.1f}ms/step avg)"
    )
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def sdedit(
    model:      DiffusionModelUNet,
    scheduler:  DDIMScheduler,
    image:      torch.Tensor,
    device:     torch.device,
    logger:     logging.Logger,
    t_start:    int = 500,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    SDEdit counterfactual inference.
    Partially corrupts the input to t_start, then denoises with the
    healthy-trained model to project onto the healthy manifold.
    Returns (1, 1, H, W) in [0, 1].
    """
    logger.info(f"SDEdit: t_start={t_start}  ddim_steps={ddim_steps}")
    logger.debug(
        f"Input image: shape={tuple(image.shape)}  dtype={image.dtype}  "
        f"value_range=[{image.min():.4f}, {image.max():.4f}]"
    )

    with Timer("scheduler.set_timesteps", logger):
        scheduler.set_timesteps(num_inference_steps=ddim_steps)

    timesteps = scheduler.timesteps
    logger.debug(
        f"DDIM timesteps ({len(timesteps)}): "
        f"{timesteps[:5].tolist()} ... {timesteps[-5:].tolist()}"
    )

    t_idx     = (timesteps - t_start).abs().argmin().item()
    t_noising = timesteps[t_idx]
    n_denoise = len(timesteps) - t_idx
    logger.info(
        f"t_start={t_start} -> nearest DDIM step={t_noising.item()}  "
        f"(idx={t_idx}, will denoise {n_denoise}/{len(timesteps)} steps)"
    )

    image = image.to(device)
    noise    = torch.randn_like(image)
    t_tensor = torch.tensor([t_noising], device=device).long()

    with Timer("scheduler.add_noise (forward diffusion)", logger):
        x = scheduler.add_noise(image, noise, t_tensor)

    logger.debug(
        f"After add_noise: x_range=[{x.min():.3f}, {x.max():.3f}]  "
        f"noise_range=[{noise.min():.3f}, {noise.max():.3f}]"
    )

    logger.info(f"Reverse diffusion: {n_denoise} steps from t={t_noising.item()} -> 0")
    t_total = time.perf_counter()

    for i, t in enumerate(timesteps[t_idx:]):
        t0_step    = time.perf_counter()
        t_batch    = t.unsqueeze(0).to(device)
        noise_pred = model(x=x, timesteps=t_batch)
        x, _       = scheduler.step(noise_pred, t, x)
        step_ms    = (time.perf_counter() - t0_step) * 1000

        if i % 10 == 0 or i == n_denoise - 1:
            logger.debug(
                f"  step [{i+1:>3}/{n_denoise}]  t={t.item():>4}  "
                f"x_range=[{x.min():.3f}, {x.max():.3f}]  "
                f"step_time={step_ms:.1f}ms"
            )

    total_s = time.perf_counter() - t_total
    logger.info(
        f"SDEdit denoising complete: {total_s:.2f}s total  "
        f"({total_s/n_denoise*1000:.1f}ms/step avg)"
    )
    return x.clamp(0.0, 1.0)


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_grid(
    samples:  torch.Tensor,
    out_path: str,
    logger:   logging.Logger,
    titles:   list[str] | None = None,
) -> None:
    """Save a row of grayscale images. samples: (N, 1, H, W) in [0, 1]."""
    logger.debug(f"Saving sample grid: n={samples.shape[0]}  path={out_path}")
    imgs = samples.cpu().float().numpy()
    n    = imgs.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=9)
    plt.tight_layout()
    with Timer(f"plt.savefig {out_path}", logger):
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sample grid -> {out_path}")


def save_sdedit_comparison(
    original:       torch.Tensor,
    counterfactual: torch.Tensor,
    out_path:       str,
    logger:         logging.Logger,
) -> None:
    """Side-by-side: input | counterfactual | difference map. All in [0, 1]."""
    logger.debug(f"Saving SDEdit comparison: path={out_path}")
    orig = original.cpu().float().numpy()[0, 0]
    cf   = counterfactual.cpu().float().numpy()[0, 0]
    diff = np.abs(orig - cf)
    logger.debug(
        f"Comparison stats — "
        f"orig=[{orig.min():.4f},{orig.max():.4f}]  "
        f"cf=[{cf.min():.4f},{cf.max():.4f}]  "
        f"diff_mean={diff.mean():.4f}  diff_max={diff.max():.4f}"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input (pathological)")
    axes[1].imshow(cf,   cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Counterfactual (healthy)")
    im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
    axes[2].set_title("Difference map")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    with Timer(f"plt.savefig {out_path}", logger):
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved SDEdit comparison -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference for brain MRI diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["sample", "sdedit"],
                   help="'sample' = unconditional generation; "
                        "'sdedit' = counterfactual from a real scan")
    p.add_argument("--ckpt",       required=True, help="Path to .pt checkpoint")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--out",        default=None,
                   help="Output image path (default: samples.png or counterfactual.png)")
    p.add_argument("--log_dir",    default="logs", help="Directory for log files")

    # sample-only
    p.add_argument("--n_samples",  type=int, default=4,
                   help="[sample] Number of images to generate")

    # sdedit-only
    p.add_argument("--nifti",      default=None,
                   help="[sdedit] Path to input NIfTI (.nii / .nii.gz)")
    p.add_argument("--slice_idx",  type=int, default=None,
                   help="[sdedit] Axial slice index into brain-cropped volume "
                        "(default: middle slice)")
    p.add_argument("--t_start",    type=int, default=500,
                   help="[sdedit] Noise level for partial diffusion (0-1000)")

    return p.parse_args()


def main() -> None:
    args   = get_args()
    logger = setup_logging(args.log_dir)

    t_main = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Inference started  mode={args.mode}")
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.debug(
            f"CUDA memory: {torch.cuda.memory_allocated()/1e6:.1f}MB allocated  "
            f"{torch.cuda.memory_reserved()/1e6:.1f}MB reserved"
        )

    model     = load_checkpoint(args.ckpt, device, logger)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    logger.debug("DDIMScheduler initialised  num_train_timesteps=1000")

    # ── Unconditional sampling ───────────────────────────────────────────────
    if args.mode == "sample":
        out = args.out or "samples.png"
        samples = unconditional_sample(
            model, scheduler, device, logger,
            n_samples  = args.n_samples,
            ddim_steps = args.ddim_steps,
        )
        save_grid(samples, out, logger)

    # ── SDEdit counterfactual ────────────────────────────────────────────────
    elif args.mode == "sdedit":
        if args.nifti is None:
            logger.error("--nifti is required for sdedit mode")
            raise ValueError("--nifti is required for sdedit mode")

        nifti_path = args.nifti
        vol        = nib.load(nifti_path)
        n_slices   = vol.shape[2]
        logger.debug(f"NIfTI header shape (pre-crop): {vol.shape}")

        slice_idx = args.slice_idx
        if slice_idx is None:
            slice_idx = int(n_slices * 0.5)
            logger.info(
                f"No --slice_idx given, defaulting to middle: "
                f"{slice_idx}/{n_slices} (pre-crop)"
            )

        out = args.out or "counterfactual.png"

        with Timer("load_slice total", logger):
            image = load_slice(nifti_path, slice_idx, logger)

        if device.type == "cuda":
            logger.debug(
                f"CUDA memory after load_slice: "
                f"{torch.cuda.memory_allocated()/1e6:.1f}MB"
            )

        with Timer("sdedit total", logger):
            cf = sdedit(
                model, scheduler, image, device, logger,
                t_start    = args.t_start,
                ddim_steps = args.ddim_steps,
            )

        if device.type == "cuda":
            logger.debug(
                f"CUDA memory after sdedit: "
                f"{torch.cuda.memory_allocated()/1e6:.1f}MB"
            )

        save_sdedit_comparison(image, cf, out, logger)

    total_s = time.perf_counter() - t_main
    logger.info(f"Inference complete  total_time={total_s:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()