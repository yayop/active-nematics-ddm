"""Conservative, auditable preprocessing for DDM image stacks."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class PhotobleachDiagnostics:
    """Compact fit metadata plus the per-frame audit trace."""

    method: str
    applied: bool
    frame_index: np.ndarray
    time_s: np.ndarray
    raw_mean: np.ndarray
    background_subtracted_mean: np.ndarray
    fitted_background_subtracted_mean: np.ndarray
    fitted_raw_mean: np.ndarray
    relative_illumination: np.ndarray
    gain: np.ndarray
    corrected_mean: np.ndarray
    background_level: float | None
    rate_s_inv: float | None
    tau_s: float | None
    r_squared_log_mean: float | None
    drop_fraction: float | None
    maximum_gain: float | None
    min_r_squared_gate: float
    min_drop_fraction_gate: float
    max_gain_gate: float

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe diagnostics without duplicating the trace arrays."""

        return {
            "method": self.method,
            "applied": self.applied,
            "model": (
                "mean(I_t)-background = amplitude*exp(-rate*t)"
                if self.method == "global-exponential"
                else None
            ),
            "fit_domain": (
                "log(mean(I_t)-background)" if self.method == "global-exponential" else None
            ),
            "fit_loss": "soft_l1" if self.method == "global-exponential" else None,
            "r_squared_definition": (
                "ordinary unweighted R^2 in log-mean space"
                if self.method == "global-exponential"
                else None
            ),
            "gain_reference": (
                "first selected frame" if self.method == "global-exponential" else None
            ),
            "trace_file": "photobleach_trace.csv",
            "selected_frame_count": int(len(self.frame_index)),
            "background_level": self.background_level,
            "rate_s_inv": self.rate_s_inv,
            "tau_s": self.tau_s,
            "r_squared_log_mean": self.r_squared_log_mean,
            "drop_fraction": self.drop_fraction,
            "raw_signal_drop_fraction": (
                float(
                    1.0 - self.background_subtracted_mean[-1] / self.background_subtracted_mean[0]
                )
                if self.applied
                else None
            ),
            "maximum_gain": self.maximum_gain,
            "gates": {
                "min_r_squared": self.min_r_squared_gate,
                "min_drop_fraction": self.min_drop_fraction_gate,
                "max_gain": self.max_gain_gate,
            },
            "raw_mean_first": float(self.raw_mean[0]),
            "raw_mean_last": float(self.raw_mean[-1]),
            "corrected_mean_first": float(self.corrected_mean[0]),
            "corrected_mean_last": float(self.corrected_mean[-1]),
        }


def _empty_trace(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=np.float64)


def _fit_robust_exponential(signal: np.ndarray, time_s: np.ndarray) -> tuple[float, float, float]:
    """Fit ``log(signal) = intercept - total_log_drop * normalized_time``."""

    log_signal = np.log(signal)
    normalized_time = time_s / time_s[-1]
    design = np.column_stack((np.ones(len(signal)), -normalized_time))
    initial, *_ = np.linalg.lstsq(design, log_signal, rcond=None)
    initial_intercept = float(initial[0])
    initial_drop = max(float(initial[1]), 0.0)
    initial_residual = log_signal - (initial_intercept - initial_drop * normalized_time)
    median_residual = float(np.median(initial_residual))
    mad = float(np.median(np.abs(initial_residual - median_residual)))
    robust_scale = max(1.4826 * mad, 1e-6)

    fit = least_squares(
        lambda parameters: log_signal - (parameters[0] - parameters[1] * normalized_time),
        x0=np.asarray([initial_intercept, initial_drop], dtype=np.float64),
        bounds=(np.asarray([-np.inf, 0.0]), np.asarray([np.inf, np.inf])),
        loss="soft_l1",
        f_scale=robust_scale,
        max_nfev=1000,
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"global-exponential photobleach fit failed: {fit.message}")

    intercept, total_log_drop = (float(value) for value in fit.x)
    fitted_log = intercept - total_log_drop * normalized_time
    residual_sum = float(np.sum(np.square(log_signal - fitted_log)))
    total_sum = float(np.sum(np.square(log_signal - np.mean(log_signal))))
    roundoff = np.finfo(np.float64).eps * max(1.0, float(np.sum(np.square(log_signal))))
    if total_sum <= roundoff:
        r_squared = 1.0 if residual_sum <= roundoff else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum
    return intercept, total_log_drop, r_squared


def apply_photobleach_correction(
    frames: np.ndarray,
    *,
    frame_interval_s: float,
    method: str = "none",
    background_level: float | None = None,
    min_r_squared: float = 0.90,
    min_drop_fraction: float = 0.02,
    max_gain: float = 2.0,
) -> PhotobleachDiagnostics:
    """Optionally correct a global exponential decay, mutating ``frames`` in place.

    The correction is accepted only after all quality gates pass, so a rejected
    fit never partially modifies the input stack.
    """

    if method not in {"none", "global-exponential"}:
        raise ValueError(f"unsupported photobleach method: {method!r}")
    gate_values = {
        "min_r_squared": min_r_squared,
        "min_drop_fraction": min_drop_fraction,
        "max_gain": max_gain,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in gate_values.values()
    ):
        raise ValueError("photobleach quality gates must be finite numbers")
    if not 0 <= min_r_squared <= 1:
        raise ValueError("min_r_squared must lie between zero and one")
    if not 0 <= min_drop_fraction < 1:
        raise ValueError("min_drop_fraction must lie in [0, 1)")
    if max_gain < 1:
        raise ValueError("max_gain must be at least one")
    if 1.0 / (1.0 - min_drop_fraction) > max_gain:
        raise ValueError("max_gain cannot accommodate min_drop_fraction")
    if background_level is not None and (
        isinstance(background_level, bool)
        or not isinstance(background_level, (int, float))
        or not math.isfinite(float(background_level))
    ):
        raise ValueError("background_level must be a finite number or omitted")
    if method == "none" and background_level is not None:
        raise ValueError("background_level is only valid for global-exponential correction")
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("photobleach preprocessing requires a (time, y, x) stack")
    if not np.issubdtype(frames.dtype, np.floating):
        raise ValueError("photobleach preprocessing requires floating-point frames")
    if not frames.flags.writeable:
        raise ValueError("photobleach preprocessing requires a writeable frame array")
    if (
        isinstance(frame_interval_s, bool)
        or not isinstance(frame_interval_s, (int, float))
        or not math.isfinite(float(frame_interval_s))
        or frame_interval_s <= 0
    ):
        raise ValueError("frame_interval_s must be finite and positive")

    frame_index = np.arange(frames.shape[0], dtype=np.int64)
    time_s = frame_index.astype(np.float64) * float(frame_interval_s)
    raw_mean = np.mean(frames, axis=(1, 2), dtype=np.float64)
    if not np.all(np.isfinite(raw_mean)):
        raise ValueError("frame means must be finite for photobleach preprocessing")

    blank = _empty_trace(len(frames))
    if method == "none":
        return PhotobleachDiagnostics(
            method=method,
            applied=False,
            frame_index=frame_index,
            time_s=time_s,
            raw_mean=raw_mean,
            background_subtracted_mean=blank.copy(),
            fitted_background_subtracted_mean=blank.copy(),
            fitted_raw_mean=blank.copy(),
            relative_illumination=blank.copy(),
            gain=np.ones(len(frames), dtype=np.float64),
            corrected_mean=raw_mean.copy(),
            background_level=None,
            rate_s_inv=None,
            tau_s=None,
            r_squared_log_mean=None,
            drop_fraction=None,
            maximum_gain=1.0,
            min_r_squared_gate=float(min_r_squared),
            min_drop_fraction_gate=float(min_drop_fraction),
            max_gain_gate=float(max_gain),
        )
    if len(frames) < 5:
        raise ValueError("global-exponential photobleach correction requires at least 5 frames")
    if background_level is None:
        raise ValueError("a finite background_level is required for global-exponential correction")

    background = float(background_level)
    signal = raw_mean - background
    if np.any(signal <= 0):
        minimum = float(np.min(signal))
        raise ValueError(
            "mean(I)-background must be positive in every selected frame; "
            f"minimum was {minimum:.6g}"
        )

    intercept, total_log_drop, r_squared = _fit_robust_exponential(signal, time_s)
    drop_fraction = float(-np.expm1(-total_log_drop))
    maximum_gain = (
        float(np.exp(total_log_drop))
        if total_log_drop <= math.log(np.finfo(np.float64).max)
        else math.inf
    )
    failures: list[str] = []
    if r_squared < min_r_squared:
        failures.append(f"R^2={r_squared:.6g} < {min_r_squared:.6g}")
    if drop_fraction < min_drop_fraction:
        failures.append(f"drop={drop_fraction:.6g} < {min_drop_fraction:.6g}")
    if maximum_gain > max_gain:
        failures.append(f"gain={maximum_gain:.6g} > {max_gain:.6g}")
    if failures:
        raise ValueError("photobleach correction rejected by quality gates: " + "; ".join(failures))

    normalized_time = time_s / time_s[-1]
    fitted_signal = np.exp(intercept - total_log_drop * normalized_time)
    relative_illumination = np.exp(-total_log_drop * normalized_time)
    gain = np.exp(total_log_drop * normalized_time)
    if not all(
        np.all(np.isfinite(values)) for values in (fitted_signal, relative_illumination, gain)
    ):
        raise ValueError("photobleach fit produced non-finite model values")

    dtype_limit = float(np.finfo(frames.dtype).max)
    for index, frame_gain in enumerate(gain):
        offset = background * (1.0 - float(frame_gain))
        scaled_min = float(np.min(frames[index])) * float(frame_gain)
        scaled_max = float(np.max(frames[index])) * float(frame_gain)
        frame_min = scaled_min + offset
        frame_max = scaled_max + offset
        if (
            not math.isfinite(offset)
            or abs(offset) > dtype_limit
            or max(abs(scaled_min), abs(scaled_max)) > dtype_limit
            or not math.isfinite(frame_min)
            or not math.isfinite(frame_max)
            or max(abs(frame_min), abs(frame_max)) > dtype_limit
        ):
            raise ValueError("photobleach correction would exceed the frame dtype range")

    for index, frame_gain in enumerate(gain):
        frames[index] *= frame_gain
        frames[index] += background * (1.0 - float(frame_gain))
    if not np.all(np.isfinite(frames)):
        raise ValueError("photobleach correction produced non-finite pixels")
    corrected_mean = np.mean(frames, axis=(1, 2), dtype=np.float64)
    rate_s_inv = total_log_drop / time_s[-1]

    return PhotobleachDiagnostics(
        method=method,
        applied=True,
        frame_index=frame_index,
        time_s=time_s,
        raw_mean=raw_mean,
        background_subtracted_mean=signal,
        fitted_background_subtracted_mean=fitted_signal,
        fitted_raw_mean=background + fitted_signal,
        relative_illumination=relative_illumination,
        gain=gain,
        corrected_mean=corrected_mean,
        background_level=background,
        rate_s_inv=float(rate_s_inv),
        tau_s=float(1.0 / rate_s_inv) if rate_s_inv > 0 else None,
        r_squared_log_mean=float(r_squared),
        drop_fraction=drop_fraction,
        maximum_gain=maximum_gain,
        min_r_squared_gate=float(min_r_squared),
        min_drop_fraction_gate=float(min_drop_fraction),
        max_gain_gate=float(max_gain),
    )


def save_photobleach_trace(
    path: Path,
    diagnostics: PhotobleachDiagnostics,
    *,
    source_frame_start: int = 0,
    source_frame_step: int = 1,
) -> None:
    """Write one audit row per selected frame for every pipeline run."""

    fields = [
        "selected_frame_index",
        "source_frame_index",
        "time_s",
        "raw_mean",
        "background_subtracted_mean",
        "fitted_background_subtracted_mean",
        "fitted_raw_mean",
        "relative_illumination",
        "gain",
        "corrected_mean",
    ]

    def value_or_blank(value: float) -> float | str:
        return float(value) if np.isfinite(value) else ""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(diagnostics.frame_index)):
            writer.writerow(
                {
                    "selected_frame_index": int(diagnostics.frame_index[index]),
                    "source_frame_index": (
                        source_frame_start + int(diagnostics.frame_index[index]) * source_frame_step
                    ),
                    "time_s": float(diagnostics.time_s[index]),
                    "raw_mean": float(diagnostics.raw_mean[index]),
                    "background_subtracted_mean": value_or_blank(
                        diagnostics.background_subtracted_mean[index]
                    ),
                    "fitted_background_subtracted_mean": value_or_blank(
                        diagnostics.fitted_background_subtracted_mean[index]
                    ),
                    "fitted_raw_mean": value_or_blank(diagnostics.fitted_raw_mean[index]),
                    "relative_illumination": value_or_blank(
                        diagnostics.relative_illumination[index]
                    ),
                    "gain": float(diagnostics.gain[index]),
                    "corrected_mean": float(diagnostics.corrected_mean[index]),
                }
            )
