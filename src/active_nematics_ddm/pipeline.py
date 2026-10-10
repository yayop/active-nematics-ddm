"""End-to-end DDM run with provenance and publication-oriented outputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import InputConfig, RunConfig, validate_config
from .ddm import DDMResult, compute_radial_ddm
from .fit import RelaxationFits, fit_relaxation
from .io import directory_images, read_frames
from .lags import logarithmic_lags
from .preprocess import (
    PhotobleachDiagnostics,
    apply_photobleach_correction,
    save_photobleach_trace,
)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "sha256": _sha256(path),
    }


def _selected_directory_images(config: InputConfig) -> list[Path]:
    files = directory_images(config.path)
    stop = config.frame_stop if config.frame_stop is not None else len(files)
    selected = files[config.frame_start : stop : config.frame_step]
    if config.max_frames is not None:
        selected = selected[: config.max_frames]
    return selected


def _input_identity(config: InputConfig) -> dict[str, Any]:
    """Content-hash exactly the file or directory frames selected for a run."""

    path = config.path
    if path.is_file():
        identity: dict[str, Any] = {
            "kind": "file",
            **_file_identity(path),
        }
    else:
        files = _selected_directory_images(config)
        digest = hashlib.sha256()
        total_size = 0
        latest_mtime = 0.0
        for item in files:
            stat = item.stat()
            total_size += stat.st_size
            latest_mtime = max(latest_mtime, stat.st_mtime)
            file_digest = _sha256(item)
            digest.update(f"{item.name}\0{stat.st_size}\0{file_digest}\n".encode())
        identity = {
            "kind": "image_directory",
            "file_count": len(files),
            "size_bytes": total_size,
            "latest_modified_utc": datetime.fromtimestamp(latest_mtime, UTC).isoformat()
            if files
            else None,
            "sha256": digest.hexdigest(),
            "hash_note": "hash covers ordered selected names, sizes, and each image's SHA-256",
            "first_selected_file": files[0].name if files else None,
            "last_selected_file": files[-1].name if files else None,
        }
    if config.flatfield_path is not None:
        identity["flatfield"] = {
            "path": str(config.flatfield_path),
            "dark_level": config.flatfield_dark_level,
            **_file_identity(config.flatfield_path),
        }
    return identity


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].casefold()))


def _git_commit(repo: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _provenance(
    config: RunConfig,
    frames: np.ndarray,
    result: DDMResult,
    input_identity: dict[str, Any],
    photobleach: PhotobleachDiagnostics,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "input": {
            "path": str(config.input.path),
            **input_identity,
            "selected_frame_count": int(frames.shape[0]),
            "selected_frame_shape": list(frames.shape[1:]),
            "selected_dtype": str(frames.dtype),
        },
        "calibration": {
            "source_pixel_size_um": config.input.pixel_size_um,
            "effective_pixel_size_um": result.pixel_size_um,
            "source_frame_interval_s": config.input.frame_interval_s,
            "effective_frame_interval_s": result.frame_interval_s,
            "flatfield_applied": config.input.flatfield_path is not None,
        },
        "photobleaching": photobleach.summary(),
        "software": {
            "repository_commit": _git_commit(repo),
            "python": sys.version,
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "imageio": _version("imageio"),
            "imageio_ffmpeg": _version("imageio-ffmpeg"),
            "ffmpeg": imageio_ffmpeg.get_ffmpeg_version(),
            "nd2": _version("nd2"),
            "tifffile": _version("tifffile"),
            "matplotlib": _version("matplotlib"),
            "threadpoolctl": _version("threadpoolctl"),
            "active_nematics_ddm": _version("active-nematics-ddm") or "source tree",
            "packages": _package_versions(),
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "uncertainty_note": (
            "origin_sem is diagnostic only because time-origin pairs overlap; "
            "use temporal block bootstrap and independent videos for inference"
        ),
    }


def _save_npz(path: Path, result: DDMResult, fits: RelaxationFits | None) -> None:
    payload: dict[str, Any] = {
        "q_um_inv": result.q_um_inv,
        "q_edges_um_inv": result.q_edges_um_inv,
        "lag_frames": result.lag_frames,
        "lag_s": result.lag_s,
        "structure_function": result.structure_function,
        "origin_sem": result.origin_sem,
        "pair_count": result.pair_count,
        "modes_per_bin": result.modes_per_bin,
        "image_shape": np.asarray(result.image_shape),
        "pixel_size_um": result.pixel_size_um,
        "frame_interval_s": result.frame_interval_s,
        "window": result.window,
    }
    if fits is not None:
        payload.update(
            {
                "fit_baseline": fits.baseline,
                "fit_amplitude": fits.amplitude,
                "fit_tau_s": fits.tau_s,
                "fit_beta": fits.beta,
                "fit_tau63_s": fits.tau63_s,
                "fit_r_squared": fits.r_squared,
                "fit_signal_to_sem": fits.signal_to_sem,
                "fit_success": fits.success,
                "fit_accepted": fits.accepted,
            }
        )
    np.savez_compressed(path, **payload)


def _save_fits(path: Path, fits: RelaxationFits) -> None:
    fields = [
        "q_um_inv",
        "baseline",
        "amplitude",
        "tau_s",
        "gamma_beta",
        "tau63_s",
        "r_squared",
        "signal_to_sem",
        "success",
        "accepted",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, q_value in enumerate(fits.q_um_inv):
            writer.writerow(
                {
                    "q_um_inv": q_value,
                    "baseline": fits.baseline[index],
                    "amplitude": fits.amplitude[index],
                    "tau_s": fits.tau_s[index],
                    "gamma_beta": fits.beta[index],
                    "tau63_s": fits.tau63_s[index],
                    "r_squared": fits.r_squared[index],
                    "signal_to_sem": fits.signal_to_sem[index],
                    "success": bool(fits.success[index]),
                    "accepted": bool(fits.accepted[index]),
                    "message": fits.message[index],
                }
            )


def _plot_overview(path: Path, result: DDMResult, fits: RelaxationFits | None) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    positive_mask = np.isfinite(result.structure_function) & (result.structure_function > 0)
    positive = result.structure_function[positive_mask]
    floor = np.percentile(positive, 1) if len(positive) else 1e-12
    image = axes[0, 0].pcolormesh(
        result.lag_s,
        result.q_um_inv,
        np.log10(np.maximum(result.structure_function.T, floor)),
        shading="auto",
        cmap="viridis",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set(xlabel="lag Δt [s]", ylabel="q [rad µm⁻¹]", title="log₁₀ D(q, Δt)")
    figure.colorbar(image, ax=axes[0, 0], label="log₁₀ image structure function")

    if fits is not None and np.any(fits.accepted):
        valid_q = np.flatnonzero(fits.accepted)
    else:
        valid_q = np.flatnonzero(result.modes_per_bin > 4)
    if len(valid_q):
        positions = np.linspace(0, len(valid_q) - 1, min(6, len(valid_q)))
        choices = valid_q[np.unique(np.rint(positions).astype(int))]
        for index in choices:
            axes[0, 1].plot(
                result.lag_s,
                result.structure_function[:, index],
                marker="o",
                markersize=2.5,
                label=f"q={result.q_um_inv[index]:.3g}",
            )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(xlabel="lag Δt [s]", ylabel="D(q, Δt)", title="Selected q bins")
    axes[0, 1].legend(fontsize=7, frameon=False)

    if fits is not None:
        fit_mask = (
            fits.accepted
            & np.isfinite(fits.tau_s)
            & (fits.tau_s > 0)
            & (fits.q_um_inv > 0)
        )
        axes[1, 0].loglog(fits.q_um_inv[fit_mask], fits.tau_s[fit_mask], "o", label="model τ")
        empirical = (
            fits.accepted
            & np.isfinite(fits.tau63_s)
            & (fits.tau63_s > 0)
            & (fits.q_um_inv > 0)
        )
        axes[1, 0].loglog(
            fits.q_um_inv[empirical], fits.tau63_s[empirical], ".", label="empirical τ₆₃"
        )
        axes[1, 0].legend(frameon=False)
        axes[1, 1].plot(fits.q_um_inv[fit_mask], fits.beta[fit_mask], "o")
        axes[1, 1].axhline(1, color="0.5", linestyle="--", linewidth=1)
        axes[1, 1].axhline(2, color="0.5", linestyle=":", linewidth=1)
    axes[1, 0].set(xlabel="q [rad µm⁻¹]", ylabel="decorrelation time [s]", title="Relaxation scale")
    axes[1, 1].set(xlabel="q [rad µm⁻¹]", ylabel="γ", title="Stretched/compressed exponent")
    axes[1, 1].set_ylim(bottom=0)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def is_same_or_ancestor(parent: Path, child: Path) -> bool:
    """Return whether ``parent`` equals or contains ``child`` across Windows drives."""

    parent_text = os.path.normcase(str(parent.resolve()))
    child_text = os.path.normcase(str(child.resolve()))
    try:
        return os.path.commonpath((parent_text, child_text)) == parent_text
    except ValueError:  # distinct Windows drives
        return False


def validate_output_location(config: RunConfig) -> None:
    """Reject output locations that could move or contaminate protected data."""

    target = config.output.directory.resolve()
    input_path = config.input.path.resolve()
    source = config.source.resolve()
    repo = Path(__file__).resolve().parents[2]
    home = Path.home().resolve()

    if target.parent == target or target == Path(target.anchor):
        raise ValueError(f"output directory cannot be a filesystem root: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"output path exists but is not a directory: {target}")
    if is_same_or_ancestor(target, home):
        raise ValueError(f"output directory cannot be home or one of its ancestors: {target}")
    if is_same_or_ancestor(target, repo):
        raise ValueError(f"output directory cannot contain the repository: {target}")
    if is_same_or_ancestor(target, source):
        raise ValueError(f"output directory cannot contain the source config: {target}")
    if input_path.is_dir():
        overlaps_input = is_same_or_ancestor(target, input_path) or is_same_or_ancestor(
            input_path, target
        )
    else:
        overlaps_input = is_same_or_ancestor(target, input_path)
    if overlaps_input:
        raise ValueError(f"output directory must not overlap the input data: {target}")
    if config.input.flatfield_path is not None and is_same_or_ancestor(
        target, config.input.flatfield_path.resolve()
    ):
        raise ValueError(f"output directory cannot contain the flat-field input: {target}")


def _prepare_output(target: Path, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.mkdir()
        return
    if not overwrite:
        raise FileExistsError(f"output exists (set output.overwrite=true to archive it): {target}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(f"{target.name}.backup-{stamp}")
    if backup.exists():
        raise FileExistsError(f"refusing to replace existing backup: {backup}")
    target.replace(backup)
    target.mkdir()


def run_pipeline(config: RunConfig, *, workers: int | None = None) -> Path:
    """Run the isotropic MVP and return the results directory."""

    if workers is not None:
        config = replace(config, analysis=replace(config.analysis, workers=workers))
    validate_config(config)
    validate_output_location(config)
    _prepare_output(config.output.directory, config.output.overwrite)
    shutil.copy2(config.source, config.output.directory / "source_config.toml")
    (config.output.directory / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    input_identity = _input_identity(config.input)
    frames = read_frames(
        config.input.path,
        frame_start=config.input.frame_start,
        frame_stop=config.input.frame_stop,
        frame_step=config.input.frame_step,
        max_frames=config.input.max_frames,
        channel=config.input.channel,
        roi=config.input.roi,
        spatial_bin=config.input.spatial_bin,
        flatfield_path=config.input.flatfield_path,
        flatfield_dark_level=config.input.flatfield_dark_level,
    )
    effective_pixel = config.input.pixel_size_um * config.input.spatial_bin
    effective_interval = config.input.frame_interval_s * config.input.frame_step
    photobleach = apply_photobleach_correction(
        frames,
        frame_interval_s=effective_interval,
        method=config.analysis.photobleach_method,
        background_level=config.analysis.photobleach_background_level,
        min_r_squared=config.analysis.photobleach_min_r_squared,
        min_drop_fraction=config.analysis.photobleach_min_drop_fraction,
        max_gain=config.analysis.photobleach_max_gain,
    )
    save_photobleach_trace(
        config.output.directory / "photobleach_trace.csv",
        photobleach,
        source_frame_start=config.input.frame_start,
        source_frame_step=config.input.frame_step,
    )
    lag_max = config.analysis.lag_max_frames
    lags = logarithmic_lags(
        len(frames),
        minimum=config.analysis.lag_min_frames,
        maximum=lag_max,
        per_decade=config.analysis.lags_per_decade,
    )
    result = compute_radial_ddm(
        frames,
        pixel_size_um=effective_pixel,
        frame_interval_s=effective_interval,
        lags=lags,
        q_bins=config.analysis.q_bins,
        q_min_um_inv=config.analysis.q_min_um_inv,
        q_max_um_inv=config.analysis.q_max_um_inv,
        window=config.analysis.window,
        subtract_frame_mean=config.analysis.subtract_frame_mean,
        max_pairs_per_lag=config.analysis.max_pairs_per_lag,
        pair_batch_size=config.analysis.pair_batch_size,
        workers=config.analysis.workers,
    )
    fits = None
    if config.fit.enabled:
        fits = fit_relaxation(
            result,
            q_min_um_inv=config.fit.q_min_um_inv,
            q_max_um_inv=config.fit.q_max_um_inv,
            beta_bounds=(config.fit.beta_min, config.fit.beta_max),
            min_points=config.fit.min_points,
            min_r_squared=config.fit.min_r_squared,
            min_signal_to_sem=config.fit.min_signal_to_sem,
            min_plateau_fraction=config.fit.min_plateau_fraction,
            max_initial_fraction=config.fit.max_initial_fraction,
        )
        _save_fits(config.output.directory / "fits.csv", fits)
    _save_npz(config.output.directory / "ddm_radial.npz", result, fits)
    _plot_overview(config.output.directory / "overview.png", result, fits)
    provenance = _provenance(config, frames, result, input_identity, photobleach)
    (config.output.directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return config.output.directory
