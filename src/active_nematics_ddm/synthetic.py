"""Small deterministic movies for smoke tests and tutorials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, shift


def translating_texture(
    n_frames: int = 192,
    shape: tuple[int, int] = (128, 128),
    velocity_px_per_frame: tuple[float, float] = (0.18, 0.35),
    noise_std: float = 0.04,
    seed: int = 7,
) -> np.ndarray:
    """Generate a periodic, translating random texture with camera noise."""

    if n_frames < 2 or min(shape) < 8:
        raise ValueError("synthetic movie is too small")
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(size=shape), sigma=2.5, mode="wrap")
    texture -= texture.min()
    texture /= max(float(texture.max()), np.finfo(float).eps)
    frames = np.empty((n_frames, *shape), dtype=np.float32)
    for index in range(n_frames):
        dy = index * velocity_px_per_frame[0]
        dx = index * velocity_px_per_frame[1]
        frame = shift(texture, (dy, dx), mode="wrap", order=3, prefilter=True)
        frame += rng.normal(scale=noise_std, size=shape)
        frames[index] = frame.astype(np.float32)
    return frames


def write_demo(path: str | Path, **kwargs: object) -> Path:
    """Write a synthetic lossless TIFF stack and return its path."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, translating_texture(**kwargs), photometric="minisblack")
    return path
