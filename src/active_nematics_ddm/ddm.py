"""Memory-conscious radial differential dynamic microscopy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import blackmanharris


@dataclass(frozen=True)
class DDMResult:
    """Radially averaged image structure function and diagnostic uncertainty."""

    q_um_inv: np.ndarray
    q_edges_um_inv: np.ndarray
    lag_frames: np.ndarray
    lag_s: np.ndarray
    structure_function: np.ndarray
    origin_sem: np.ndarray
    pair_count: np.ndarray
    modes_per_bin: np.ndarray
    image_shape: tuple[int, int]
    pixel_size_um: float
    frame_interval_s: float
    window: str


def _window(shape: tuple[int, int], name: str) -> np.ndarray:
    if name == "none":
        return np.ones(shape, dtype=np.float32)
    if name == "hann":
        wy, wx = np.hanning(shape[0]), np.hanning(shape[1])
    elif name == "blackman-harris":
        wy, wx = blackmanharris(shape[0], sym=True), blackmanharris(shape[1], sym=True)
    else:
        raise ValueError(f"unknown window: {name}")
    result = np.outer(wy, wx)
    rms = np.sqrt(np.mean(result**2))
    if rms == 0:
        raise ValueError("window is zero for this image shape")
    return np.asarray(result / rms, dtype=np.float32)


def _fft_frames(
    frames: np.ndarray,
    *,
    window: str,
    subtract_frame_mean: bool,
    batch_size: int = 32,
) -> np.ndarray:
    n_frames, height, width = frames.shape
    transform = np.empty((n_frames, height, width // 2 + 1), dtype=np.complex64)
    taper = _window((height, width), window)
    for start in range(0, n_frames, batch_size):
        stop = min(n_frames, start + batch_size)
        work = np.array(frames[start:stop], dtype=np.float32, copy=True)
        if subtract_frame_mean:
            work -= work.mean(axis=(-2, -1), keepdims=True)
        work *= taper
        transform[start:stop] = np.fft.rfft2(work, axes=(-2, -1), norm="ortho")
    return transform


def _radial_geometry(
    shape: tuple[int, int],
    pixel_size_um: float,
    q_bins: int,
    q_min_um_inv: float | None,
    q_max_um_inv: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    qy = 2 * np.pi * np.fft.fftfreq(height, d=pixel_size_um)
    qx = 2 * np.pi * np.fft.rfftfreq(width, d=pixel_size_um)
    q_grid = np.hypot(qy[:, None], qx[None, :])
    # rfft stores only qx >= 0. Interior columns represent both ±qx modes;
    # weighting them by two reproduces an average over the complete FFT plane.
    mode_weights = np.full(q_grid.shape, 2.0, dtype=np.float64)
    mode_weights[:, 0] = 1.0
    if width % 2 == 0:
        mode_weights[:, -1] = 1.0
    q_min = 0.0 if q_min_um_inv is None else float(q_min_um_inv)
    # The inscribed Fourier circle has comparable angular support at every q.
    q_nyquist = min(float(np.max(np.abs(qy))), float(np.max(qx)))
    q_max = q_nyquist if q_max_um_inv is None else min(float(q_max_um_inv), q_nyquist)
    if not 0 <= q_min < q_max:
        raise ValueError(f"invalid q range [{q_min}, {q_max}] rad/um")

    edges = np.linspace(q_min, q_max, q_bins + 1, dtype=np.float64)
    bins = np.searchsorted(edges, q_grid.ravel(), side="right") - 1
    bins[np.isclose(q_grid.ravel(), edges[-1])] = q_bins - 1
    valid = (bins >= 0) & (bins < q_bins)
    bin_map = np.where(valid, bins, -1).astype(np.int32)
    valid_weights = mode_weights.ravel()[valid]
    counts = np.bincount(bins[valid], weights=valid_weights, minlength=q_bins)
    q_sum = np.bincount(
        bins[valid],
        weights=q_grid.ravel()[valid] * valid_weights,
        minlength=q_bins,
    )
    centers = np.divide(q_sum, counts, out=(edges[:-1] + edges[1:]) / 2, where=counts > 0)
    return edges, centers, bin_map, counts.astype(np.int64), mode_weights


def _origins(n_frames: int, lag: int, maximum: int) -> np.ndarray:
    available = n_frames - lag
    if available <= maximum:
        return np.arange(available, dtype=np.int64)
    # Deterministic coverage of the entire recording, including both ends.
    return np.unique(np.rint(np.linspace(0, available - 1, maximum)).astype(np.int64))


def _one_lag(
    transform: np.ndarray,
    lag: int,
    bin_map: np.ndarray,
    counts: np.ndarray,
    mode_weights: np.ndarray,
    max_pairs: int,
    pair_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    starts = _origins(transform.shape[0], lag, max_pairs)
    n_bins = len(counts)
    profiles = np.full((len(starts), n_bins), np.nan, dtype=np.float64)
    flat_bins = bin_map.ravel()
    valid = flat_bins >= 0
    valid_bins = flat_bins[valid]
    valid_weights = mode_weights.ravel()[valid]

    for batch_start in range(0, len(starts), pair_batch_size):
        batch_origins = starts[batch_start : batch_start + pair_batch_size]
        delta = transform[batch_origins + lag] - transform[batch_origins]
        power = delta.real.astype(np.float64) ** 2 + delta.imag.astype(np.float64) ** 2
        for local_index, values in enumerate(power.reshape(len(batch_origins), -1)):
            sums = np.bincount(
                valid_bins,
                weights=values[valid] * valid_weights,
                minlength=n_bins,
            )
            profiles[batch_start + local_index] = np.divide(
                sums,
                counts,
                out=np.full(n_bins, np.nan),
                where=counts > 0,
            )

    occupied = counts > 0
    mean = np.full(n_bins, np.nan)
    mean[occupied] = np.mean(profiles[:, occupied], axis=0)
    if len(starts) > 1:
        sem = np.full(n_bins, np.nan)
        sem[occupied] = np.std(profiles[:, occupied], axis=0, ddof=1) / np.sqrt(len(starts))
    else:
        sem = np.full(n_bins, np.nan)
    return mean, sem, len(starts)


def compute_radial_ddm(
    frames: np.ndarray,
    *,
    pixel_size_um: float,
    frame_interval_s: float,
    lags: np.ndarray,
    q_bins: int = 64,
    q_min_um_inv: float | None = None,
    q_max_um_inv: float | None = None,
    window: str = "blackman-harris",
    subtract_frame_mean: bool = True,
    max_pairs_per_lag: int = 512,
    pair_batch_size: int = 16,
    workers: int = 1,
) -> DDMResult:
    """Compute radially averaged ``D(q, lag)`` from a frame stack.

    FFTs are evaluated once per frame. Lags share this read-only cache and can
    run concurrently in threads without copying the video. ``origin_sem`` is a
    diagnostic SEM across sampled time origins, not a final confidence interval:
    origins overlap and are correlated, so publication-level inference should
    use temporal block bootstrap and independent-video replication.
    """

    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] < 2 or min(frames.shape[1:]) < 4:
        raise ValueError("frames must have shape (time>=2, y>=4, x>=4)")
    if not np.issubdtype(frames.dtype, np.number) or not np.all(np.isfinite(frames)):
        raise ValueError("frames must be finite numeric values")
    if pixel_size_um <= 0 or frame_interval_s <= 0:
        raise ValueError("pixel_size_um and frame_interval_s must be positive")
    lags = np.unique(np.asarray(lags, dtype=np.int64))
    if len(lags) == 0 or lags[0] < 1 or lags[-1] >= frames.shape[0]:
        raise ValueError("lags must lie between 1 and n_frames - 1")
    if max_pairs_per_lag < 2 or pair_batch_size < 1 or workers < 1:
        raise ValueError("invalid pair or worker settings")

    edges, centers, bin_map, counts, mode_weights = _radial_geometry(
        frames.shape[1:], pixel_size_um, q_bins, q_min_um_inv, q_max_um_inv
    )
    transform = _fft_frames(
        frames,
        window=window,
        subtract_frame_mean=subtract_frame_mean,
    )

    def task(lag: np.int64) -> tuple[np.ndarray, np.ndarray, int]:
        return _one_lag(
            transform,
            int(lag),
            bin_map,
            counts,
            mode_weights,
            max_pairs_per_lag,
            pair_batch_size,
        )

    worker_count = min(int(workers), len(lags))
    if worker_count == 1:
        rows = list(map(task, lags))
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ddm-lag") as pool:
            rows = list(pool.map(task, lags))

    structure = np.stack([row[0] for row in rows])
    sem = np.stack([row[1] for row in rows])
    pairs = np.asarray([row[2] for row in rows], dtype=np.int64)
    return DDMResult(
        q_um_inv=centers,
        q_edges_um_inv=edges,
        lag_frames=lags,
        lag_s=lags * frame_interval_s,
        structure_function=structure,
        origin_sem=sem,
        pair_count=pairs,
        modes_per_bin=counts,
        image_shape=(frames.shape[1], frames.shape[2]),
        pixel_size_um=float(pixel_size_um),
        frame_interval_s=float(frame_interval_s),
        window=window,
    )
