"""Microscopy frame loading with explicit, reproducible selection."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import nd2
import numpy as np
import tifffile

TIFF_SUFFIXES = {".tif", ".tiff"}
IMAGE_SUFFIXES = TIFF_SUFFIXES | {".png", ".jpg", ".jpeg", ".bmp", ".npy"}


def _validate_nd2_layout(movie: nd2.ND2File, path: Path) -> int:
    """Return the time length for an unambiguous single-channel TYX ND2."""

    sizes = dict(movie.sizes)
    unexpected = sorted(set(sizes) - {"T", "Y", "X"})
    if unexpected or not {"T", "Y", "X"}.issubset(sizes):
        axes = ", ".join(f"{axis}={length}" for axis, length in sizes.items())
        raise ValueError(
            f"ambiguous ND2 axes ({axes}) for {path}; export/select one position, "
            "Z plane, and channel as a TYX movie"
        )
    frame_count = int(sizes["T"])
    if frame_count < 2:
        raise ValueError(f"ND2 input must contain at least two time points: {path}")
    return frame_count


def _selection_indices(
    frame_count: int,
    *,
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
    max_frames: int | None,
) -> list[int]:
    stop = frame_count if frame_stop is None else min(frame_stop, frame_count)
    indices = list(range(frame_start, stop, frame_step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def directory_images(path: str | Path) -> list[Path]:
    """List supported top-level images in deterministic natural order."""

    path = Path(path).expanduser().resolve()
    images = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )
    if not images:
        raise ValueError(f"no supported images found directly inside directory: {path}")
    return images


def _tiff_frames(path: Path, *, single_image: bool) -> Iterator[np.ndarray]:
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError(
                f"ambiguous TIFF with {len(tif.series)} series: {path}; export one TYX series"
            )
        series = tif.series[0]
        axes = series.axes.upper()
        single_axes = {"YX", "YXS"}
        stack_axes = {"TYX", "TYXS", "QYX", "QYXS", "IYX", "IYXS"}
        if axes in single_axes:
            yield series.asarray()
            return
        if single_image:
            raise ValueError(
                f"directory item {path.name} has TIFF axes {axes}; expected one YX/YXS frame"
            )
        if axes not in stack_axes:
            raise ValueError(
                f"ambiguous TIFF axes {axes} for {path}; export a single channel/Z as TYX"
            )
        expected_frames = int(series.shape[0])
        pages = list(series.pages)
        if len(pages) != expected_frames:
            raise ValueError(
                f"TIFF axes {axes} use a non-page time layout in {path}; export as page-wise TYX"
            )
        for page in pages:
            yield page.asarray()


def _read_directory_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return next(_tiff_frames(path, single_image=True))
    if suffix == ".npy":
        image = np.load(path, mmap_mode="r")
    else:
        image = iio.imread(path)
    if image.ndim >= 3 and image.shape[0] == 1 and image.shape[-1] not in (3, 4):
        image = image[0]
    if image.ndim not in (2, 3):
        raise ValueError(f"directory item must be one 2D/color frame, got {path}: {image.shape}")
    return image


def _read_reference_image(path: Path, channel: int | None) -> np.ndarray:
    if path.is_dir():
        raise ValueError(f"flat-field path must be one image, not a directory: {path}")
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        image = next(_tiff_frames(path, single_image=True))
    elif suffix == ".npy":
        image = np.load(path, mmap_mode="r")
    else:
        image = iio.imread(path)
    return _select_channel(np.asarray(image), channel)


def _flatfield_gain(
    path: Path | None,
    *,
    channel: int | None,
    dark_level: float,
) -> np.ndarray | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    reference = np.asarray(_read_reference_image(path, channel), dtype=np.float32)
    denominator = reference - np.float32(dark_level)
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError(
            "flat-field must be finite and strictly above flatfield_dark_level everywhere"
        )
    normalization = float(np.median(denominator))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("flat-field normalization is not finite and positive")
    return np.asarray(normalization / denominator, dtype=np.float32)


def _iter_raw_frames(path: Path) -> Iterator[np.ndarray]:
    if path.is_dir():
        for image_path in directory_images(path):
            yield _read_directory_image(image_path)
        return
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        if array.ndim < 3:
            raise ValueError("a .npy input must have shape (time, y, x[, channel])")
        yield from array
        return
    if suffix == ".nd2":
        with nd2.ND2File(path) as movie:
            frame_count = _validate_nd2_layout(movie, path)
            for index in range(frame_count):
                yield movie.read_frame(index)
        return
    if suffix in TIFF_SUFFIXES:
        yield from _tiff_frames(path, single_image=False)
        return
    yield from iio.imiter(path)


def _select_channel(frame: np.ndarray, channel: int | None) -> np.ndarray:
    frame = np.asarray(frame)
    frame = np.squeeze(frame)
    if frame.ndim == 2:
        if channel not in (None, 0):
            raise ValueError(f"channel {channel} requested for a grayscale frame")
        return frame
    if frame.ndim != 3:
        raise ValueError(f"expected a 2D or color frame, got shape {frame.shape}")

    if frame.shape[-1] not in (2, 3, 4):
        raise ValueError(
            f"ambiguous frame shape {frame.shape}; export a time/y/x stack or select one channel"
        )
    if channel is not None:
        if not 0 <= channel < frame.shape[-1]:
            raise ValueError(f"channel {channel} is outside frame shape {frame.shape}")
        return frame[..., channel]
    if frame.shape[-1] < 3:
        return frame[..., 0]
    rgb = frame[..., :3].astype(np.float32, copy=False)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _crop_and_bin(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    spatial_bin: int,
) -> np.ndarray:
    if roi is not None:
        y0, y1, x0, x1 = roi
        if y1 > frame.shape[0] or x1 > frame.shape[1]:
            raise ValueError(f"ROI {roi} lies outside frame shape {frame.shape}")
        frame = frame[y0:y1, x0:x1]
    if spatial_bin > 1:
        height = frame.shape[0] // spatial_bin * spatial_bin
        width = frame.shape[1] // spatial_bin * spatial_bin
        if height == 0 or width == 0:
            raise ValueError("spatial_bin is larger than the selected frame")
        frame = frame[:height, :width]
        frame = frame.reshape(
            height // spatial_bin,
            spatial_bin,
            width // spatial_bin,
            spatial_bin,
        ).mean(axis=(1, 3))
    return np.asarray(frame, dtype=np.float32)


def _prepare_frame(
    raw: np.ndarray,
    *,
    channel: int | None,
    roi: tuple[int, int, int, int] | None,
    spatial_bin: int,
    flatfield_gain: np.ndarray | None,
    flatfield_dark_level: float,
) -> np.ndarray:
    frame = _select_channel(raw, channel)
    if flatfield_gain is not None:
        if frame.shape != flatfield_gain.shape:
            raise ValueError(
                f"flat-field shape {flatfield_gain.shape} does not match frame shape {frame.shape}"
            )
        frame = (np.asarray(frame, dtype=np.float32) - flatfield_dark_level) * flatfield_gain
    return _crop_and_bin(frame, roi, spatial_bin)


def read_frames(
    path: str | Path,
    *,
    frame_start: int = 0,
    frame_stop: int | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
    channel: int | None = None,
    roi: tuple[int, int, int, int] | None = None,
    spatial_bin: int = 1,
    flatfield_path: str | Path | None = None,
    flatfield_dark_level: float = 0.0,
) -> np.ndarray:
    """Read selected frames into a contiguous ``float32`` array.

    Decoding is sequential, which avoids repeated random seeks in compressed
    video. For datasets that do not fit RAM, crop/bin or split into temporal
    windows; a disk-backed FFT cache is planned for the next milestone.
    """

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if frame_start < 0 or frame_step < 1 or spatial_bin < 1:
        raise ValueError("invalid frame selection or spatial bin")
    if not np.isfinite(flatfield_dark_level):
        raise ValueError("flatfield_dark_level must be finite")
    gain = _flatfield_gain(
        None if flatfield_path is None else Path(flatfield_path),
        channel=channel,
        dark_level=float(flatfield_dark_level),
    )

    if path.suffix.lower() == ".nd2":
        with nd2.ND2File(path) as movie:
            frame_count = _validate_nd2_layout(movie, path)
            indices = _selection_indices(
                frame_count,
                frame_start=frame_start,
                frame_stop=frame_stop,
                frame_step=frame_step,
                max_frames=max_frames,
            )
            if len(indices) < 2:
                raise ValueError(f"selection returned only {len(indices)} frame(s)")
            if channel not in (None, 0):
                raise ValueError(f"channel {channel} requested for a single-channel ND2")

            first = _prepare_frame(
                np.asarray(movie.read_frame(indices[0])),
                channel=channel,
                roi=roi,
                spatial_bin=spatial_bin,
                flatfield_gain=gain,
                flatfield_dark_level=flatfield_dark_level,
            )
            selected_nd2 = np.empty((len(indices), *first.shape), dtype=np.float32)
            selected_nd2[0] = first
            for destination, source_index in enumerate(indices[1:], start=1):
                frame = _prepare_frame(
                    np.asarray(movie.read_frame(source_index)),
                    channel=channel,
                    roi=roi,
                    spatial_bin=spatial_bin,
                    flatfield_gain=gain,
                    flatfield_dark_level=flatfield_dark_level,
                )
                if frame.shape != first.shape:
                    raise ValueError(
                        f"frame {source_index} has shape {frame.shape}; expected {first.shape}"
                    )
                selected_nd2[destination] = frame
        return np.ascontiguousarray(selected_nd2)

    selected: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for index, raw in enumerate(_iter_raw_frames(path)):
        if index < frame_start:
            continue
        if frame_stop is not None and index >= frame_stop:
            break
        if (index - frame_start) % frame_step:
            continue
        frame = _prepare_frame(
            raw,
            channel=channel,
            roi=roi,
            spatial_bin=spatial_bin,
            flatfield_gain=gain,
            flatfield_dark_level=flatfield_dark_level,
        )
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(f"frame {index} has shape {frame.shape}; expected {shape}")
        selected.append(frame)
        if max_frames is not None and len(selected) >= max_frames:
            break

    if len(selected) < 2:
        raise ValueError(f"selection returned only {len(selected)} frame(s)")
    return np.ascontiguousarray(np.stack(selected), dtype=np.float32)


def inspect_input(path: str | Path) -> dict[str, Any]:
    """Return JSON-safe, non-destructive metadata and first-frame information."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".nd2":
        with nd2.ND2File(path) as movie:
            frame_count = _validate_nd2_layout(movie, path)
            first = movie.read_frame(0)
            sizes = dict(movie.sizes)
            voxel = movie.voxel_size()
            events = movie.events()
            timestamps = np.asarray(
                [record.get("Time [s]", np.nan) for record in events], dtype=np.float64
            )
            finite_times = timestamps[np.isfinite(timestamps)]
            intervals = np.diff(finite_times)
            result: dict[str, Any] = {
                "path": str(path),
                "kind": "file",
                "suffix": ".nd2",
                "size_bytes": path.stat().st_size,
                "first_frame_shape": list(np.asarray(first).shape),
                "first_frame_dtype": str(np.asarray(first).dtype),
                "axes": list(sizes),
                "array_shape": list(movie.shape),
                "frame_count": frame_count,
                "pixel_size_um_xy": [float(voxel.y), float(voxel.x)],
            }
            if len(finite_times):
                result["timestamp_start_s"] = float(finite_times[0])
                result["timestamp_end_s"] = float(finite_times[-1])
            if len(intervals):
                result["frame_interval_median_s"] = float(np.median(intervals))
                result["frame_interval_min_s"] = float(np.min(intervals))
                result["frame_interval_max_s"] = float(np.max(intervals))
            return result

    first = next(_iter_raw_frames(path))
    result: dict[str, Any] = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "first_frame_shape": list(np.asarray(first).shape),
        "first_frame_dtype": str(np.asarray(first).dtype),
    }
    if path.is_dir():
        images = directory_images(path)
        result["frame_count"] = len(images)
        result["size_bytes"] = sum(item.stat().st_size for item in images)
        result["extensions"] = sorted({item.suffix.lower() for item in images})
        result["first_file"] = images[0].name
        result["last_file"] = images[-1].name
    elif path.suffix.lower() == ".npy":
        result["suffix"] = path.suffix.lower()
        result["size_bytes"] = path.stat().st_size
        array = np.load(path, mmap_mode="r")
        result["array_shape"] = list(array.shape)
        result["frame_count"] = int(array.shape[0])
    elif path.suffix.lower() in TIFF_SUFFIXES:
        result["suffix"] = path.suffix.lower()
        result["size_bytes"] = path.stat().st_size
        with tifffile.TiffFile(path) as tif:
            result["page_count"] = len(tif.pages)
            result["series_shapes"] = [list(series.shape) for series in tif.series]
    else:
        result["suffix"] = path.suffix.lower()
        result["size_bytes"] = path.stat().st_size
        try:
            metadata = iio.immeta(path)
            for source, target in (
                ("fps", "container_fps"),
                ("duration", "duration_s"),
                ("nframes", "container_frame_count"),
            ):
                value = metadata.get(source)
                if isinstance(value, (int, float, str, bool)):
                    result[target] = value
        except Exception as error:  # metadata support varies by codec/plugin
            result["metadata_warning"] = str(error)
    return result


def format_inspection(path: str | Path) -> str:
    return json.dumps(inspect_input(path), indent=2, sort_keys=True)
