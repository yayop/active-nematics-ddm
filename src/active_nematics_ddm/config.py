"""Typed TOML configuration and validation."""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputConfig:
    path: Path
    pixel_size_um: float
    frame_interval_s: float
    frame_start: int = 0
    frame_stop: int | None = None
    frame_step: int = 1
    max_frames: int | None = None
    channel: int | None = None
    roi: tuple[int, int, int, int] | None = None
    spatial_bin: int = 1
    flatfield_path: Path | None = None
    flatfield_dark_level: float = 0.0


@dataclass(frozen=True)
class AnalysisConfig:
    lag_min_frames: int = 1
    lag_max_frames: int | None = None
    lags_per_decade: int = 12
    max_pairs_per_lag: int = 512
    q_bins: int = 64
    q_min_um_inv: float | None = None
    q_max_um_inv: float | None = None
    window: str = "blackman-harris"
    subtract_frame_mean: bool = True
    photobleach_method: str = "none"
    photobleach_background_level: float | None = None
    photobleach_min_r_squared: float = 0.90
    photobleach_min_drop_fraction: float = 0.02
    photobleach_max_gain: float = 2.0
    workers: int = 8
    pair_batch_size: int = 16


@dataclass(frozen=True)
class FitConfig:
    enabled: bool = True
    q_min_um_inv: float | None = None
    q_max_um_inv: float | None = None
    beta_min: float = 0.25
    beta_max: float = 3.0
    min_points: int = 8
    min_r_squared: float = 0.8
    min_signal_to_sem: float = 5.0
    min_plateau_fraction: float = 0.8
    max_initial_fraction: float = 0.25


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    overwrite: bool = False


@dataclass(frozen=True)
class RunConfig:
    input: InputConfig
    analysis: AnalysisConfig
    fit: FitConfig
    output: OutputConfig
    source: Path

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["input"]["path"] = str(self.input.path)
        if self.input.flatfield_path is not None:
            result["input"]["flatfield_path"] = str(self.input.flatfield_path)
        result["output"]["directory"] = str(self.output.directory)
        result["source"] = str(self.source)
        return result


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or omitted")
    return value


def _integer(value: Any, name: str) -> int:
    result = _optional_int(value, name)
    if result is None:
        raise ValueError(f"{name} is required")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} setting(s): {', '.join(unknown)}")


def _roi(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(type(v) is int for v in value)
    ):
        raise ValueError("input.roi must be [y0, y1, x0, x1]")
    y0, y1, x0, x1 = value
    if min(value) < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError("input.roi must have non-negative, increasing bounds")
    return y0, y1, x0, x1


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run configuration.

    Relative input and output paths are resolved from the TOML file location,
    so the same config behaves identically from any working directory.
    """

    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    _reject_unknown(raw, {"input", "analysis", "fit", "output"}, "top-level")
    missing = [section for section in ("input", "output") if section not in raw]
    if missing:
        raise ValueError(f"missing TOML section(s): {', '.join(missing)}")

    base = source.parent
    inp = raw["input"]
    out = raw["output"]
    if not isinstance(inp, dict) or not isinstance(out, dict):
        raise ValueError("input and output must be TOML tables")
    _reject_unknown(inp, {field.name for field in dataclasses.fields(InputConfig)}, "input")
    if "path" not in inp or "pixel_size_um" not in inp or "frame_interval_s" not in inp:
        raise ValueError("input.path, input.pixel_size_um and input.frame_interval_s are required")
    if not isinstance(inp["path"], str) or not inp["path"].strip():
        raise ValueError("input.path must be a non-empty string")
    input_path = Path(inp["path"]).expanduser()
    if not input_path.is_absolute():
        input_path = (base / input_path).resolve()
    flatfield_value = inp.get("flatfield_path")
    if flatfield_value is not None and (
        not isinstance(flatfield_value, str) or not flatfield_value.strip()
    ):
        raise ValueError("input.flatfield_path must be a non-empty string or omitted")
    flatfield_path = None
    if flatfield_value is not None:
        flatfield_path = Path(flatfield_value).expanduser()
        if not flatfield_path.is_absolute():
            flatfield_path = (base / flatfield_path).resolve()

    input_config = InputConfig(
        path=input_path,
        pixel_size_um=_number(inp["pixel_size_um"], "input.pixel_size_um"),
        frame_interval_s=_number(inp["frame_interval_s"], "input.frame_interval_s"),
        frame_start=_integer(inp.get("frame_start", 0), "input.frame_start"),
        frame_stop=_optional_int(inp.get("frame_stop"), "input.frame_stop"),
        frame_step=_integer(inp.get("frame_step", 1), "input.frame_step"),
        max_frames=_optional_int(inp.get("max_frames"), "input.max_frames"),
        channel=_optional_int(inp.get("channel"), "input.channel"),
        roi=_roi(inp.get("roi")),
        spatial_bin=_integer(inp.get("spatial_bin", 1), "input.spatial_bin"),
        flatfield_path=flatfield_path,
        flatfield_dark_level=_number(
            inp.get("flatfield_dark_level", 0.0), "input.flatfield_dark_level"
        ),
    )

    analysis_raw = raw.get("analysis", {})
    fit_raw = raw.get("fit", {})
    if not isinstance(analysis_raw, dict) or not isinstance(fit_raw, dict):
        raise ValueError("analysis and fit must be TOML tables")
    _reject_unknown(
        analysis_raw,
        {field.name for field in dataclasses.fields(AnalysisConfig)},
        "analysis",
    )
    _reject_unknown(fit_raw, {field.name for field in dataclasses.fields(FitConfig)}, "fit")
    analysis_config = AnalysisConfig(**analysis_raw)
    fit_config = FitConfig(**fit_raw)

    _reject_unknown(out, {field.name for field in dataclasses.fields(OutputConfig)}, "output")
    if "directory" not in out:
        raise ValueError("output.directory is required")
    if not isinstance(out["directory"], str) or not out["directory"].strip():
        raise ValueError("output.directory must be a non-empty string")
    output_path = Path(out["directory"]).expanduser()
    if not output_path.is_absolute():
        output_path = (base / output_path).resolve()
    output_config = OutputConfig(
        directory=output_path,
        overwrite=_boolean(out.get("overwrite", False), "output.overwrite"),
    )

    config = RunConfig(input_config, analysis_config, fit_config, output_config, source)
    validate_config(config)
    return config


def validate_config(config: RunConfig, *, require_input: bool = True) -> None:
    """Raise ``ValueError`` for unsafe or physically invalid settings."""

    inp, analysis, fit = config.input, config.analysis, config.fit
    if require_input and not inp.path.exists():
        raise ValueError(f"input path does not exist: {inp.path}")
    if require_input and not (inp.path.is_file() or inp.path.is_dir()):
        raise ValueError(f"input path is neither a file nor directory: {inp.path}")
    if require_input and inp.flatfield_path is not None and not inp.flatfield_path.is_file():
        raise ValueError(f"flat-field path is not a file: {inp.flatfield_path}")
    numeric_input = (inp.pixel_size_um, inp.frame_interval_s)
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in numeric_input
    ):
        raise ValueError("pixel_size_um and frame_interval_s must be numbers")
    if not all(math.isfinite(float(value)) for value in numeric_input):
        raise ValueError("pixel_size_um and frame_interval_s must be finite")
    if inp.pixel_size_um <= 0 or inp.frame_interval_s <= 0:
        raise ValueError("pixel_size_um and frame_interval_s must be positive")
    if not math.isfinite(inp.flatfield_dark_level):
        raise ValueError("flatfield_dark_level must be finite")
    if inp.flatfield_path is None and inp.flatfield_dark_level != 0:
        raise ValueError("input.flatfield_dark_level requires input.flatfield_path")
    integer_input = (inp.frame_start, inp.frame_step, inp.spatial_bin)
    if not all(type(value) is int for value in integer_input):
        raise ValueError("frame_start, frame_step, and spatial_bin must be integers")
    if inp.frame_start < 0 or inp.frame_step < 1 or inp.spatial_bin < 1:
        raise ValueError("frame_start >= 0, frame_step >= 1, and spatial_bin >= 1 are required")
    optional_input_ints = (inp.frame_stop, inp.max_frames, inp.channel)
    if any(value is not None and type(value) is not int for value in optional_input_ints):
        raise ValueError("frame_stop, max_frames, and channel must be integers or omitted")
    if inp.frame_stop is not None and inp.frame_stop <= inp.frame_start:
        raise ValueError("frame_stop must exceed frame_start")
    if inp.max_frames is not None and inp.max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if inp.channel is not None and inp.channel < 0:
        raise ValueError("input.channel must be non-negative")
    integer_analysis = (
        analysis.lag_min_frames,
        analysis.lags_per_decade,
        analysis.max_pairs_per_lag,
        analysis.q_bins,
        analysis.workers,
        analysis.pair_batch_size,
        fit.min_points,
    )
    if not all(type(value) is int for value in integer_analysis):
        raise ValueError("analysis integer settings and fit.min_points must be integers")
    if analysis.lag_max_frames is not None and type(analysis.lag_max_frames) is not int:
        raise ValueError("analysis.lag_max_frames must be an integer or omitted")
    if (
        type(analysis.subtract_frame_mean) is not bool
        or type(fit.enabled) is not bool
        or type(config.output.overwrite) is not bool
    ):
        raise ValueError("boolean settings must be true or false")
    if analysis.lag_min_frames < 1:
        raise ValueError("analysis.lag_min_frames must be at least 1")
    if analysis.lag_max_frames is not None and analysis.lag_max_frames < analysis.lag_min_frames:
        raise ValueError("analysis.lag_max_frames must not be below lag_min_frames")
    if analysis.lags_per_decade < 1 or analysis.max_pairs_per_lag < 2:
        raise ValueError("lags_per_decade >= 1 and max_pairs_per_lag >= 2 are required")
    if analysis.q_bins < 4 or analysis.workers < 1 or analysis.pair_batch_size < 1:
        raise ValueError("q_bins >= 4, workers >= 1, and pair_batch_size >= 1 are required")
    if not isinstance(analysis.window, str) or analysis.window not in {
        "blackman-harris",
        "hann",
        "none",
    }:
        raise ValueError("analysis.window must be 'blackman-harris', 'hann', or 'none'")
    if not isinstance(analysis.photobleach_method, str) or analysis.photobleach_method not in {
        "none",
        "global-exponential",
    }:
        raise ValueError("analysis.photobleach_method must be 'none' or 'global-exponential'")
    finite_values = (
        fit.beta_min,
        fit.beta_max,
        fit.min_r_squared,
        fit.min_signal_to_sem,
        fit.min_plateau_fraction,
        fit.max_initial_fraction,
        analysis.q_min_um_inv,
        analysis.q_max_um_inv,
        fit.q_min_um_inv,
        fit.q_max_um_inv,
        analysis.photobleach_background_level,
        analysis.photobleach_min_r_squared,
        analysis.photobleach_min_drop_fraction,
        analysis.photobleach_max_gain,
    )
    if any(
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        )
        for value in finite_values
    ):
        raise ValueError("all q ranges and analysis/fit thresholds must be finite numbers")
    if (
        analysis.photobleach_method == "global-exponential"
        and analysis.photobleach_background_level is None
    ):
        raise ValueError(
            "analysis.photobleach_background_level is required when "
            "photobleach_method='global-exponential'"
        )
    if analysis.photobleach_method == "none" and analysis.photobleach_background_level is not None:
        raise ValueError(
            "analysis.photobleach_background_level is only valid when "
            "photobleach_method='global-exponential'"
        )
    if not 0 <= analysis.photobleach_min_r_squared <= 1:
        raise ValueError("analysis.photobleach_min_r_squared must lie between zero and one")
    if not 0 <= analysis.photobleach_min_drop_fraction < 1:
        raise ValueError("analysis.photobleach_min_drop_fraction must lie in [0, 1)")
    if analysis.photobleach_max_gain < 1:
        raise ValueError("analysis.photobleach_max_gain must be at least one")
    minimum_required_gain = 1.0 / (1.0 - analysis.photobleach_min_drop_fraction)
    if minimum_required_gain > analysis.photobleach_max_gain:
        raise ValueError(
            "photobleach gates are incompatible: max_gain cannot accommodate min_drop_fraction"
        )
    if not (0 < fit.beta_min < fit.beta_max):
        raise ValueError("fit beta bounds are invalid")
    if fit.min_points < 4:
        raise ValueError("fit.min_points must be at least 4")
    if not 0 <= fit.min_r_squared <= 1 or fit.min_signal_to_sem < 0:
        raise ValueError("fit quality thresholds are invalid")
    if not 0 < fit.min_plateau_fraction < 1 or not 0 < fit.max_initial_fraction < 1:
        raise ValueError("fit relaxation-fraction thresholds must lie between zero and one")
    for name, lower, upper in (
        ("analysis", analysis.q_min_um_inv, analysis.q_max_um_inv),
        ("fit", fit.q_min_um_inv, fit.q_max_um_inv),
    ):
        if lower is not None and lower < 0:
            raise ValueError(f"{name}.q_min_um_inv must be non-negative")
        if upper is not None and upper <= 0:
            raise ValueError(f"{name}.q_max_um_inv must be positive")
        if lower is not None and upper is not None and upper <= lower:
            raise ValueError(f"{name} q range must have max > min")
