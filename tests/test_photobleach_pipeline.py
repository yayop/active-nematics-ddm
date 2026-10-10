import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from active_nematics_ddm.config import load_config
from active_nematics_ddm.pipeline import run_pipeline


def _write_pipeline_config(
    path: Path,
    *,
    output: str,
    correction: bool,
) -> Path:
    photobleach = ""
    if correction:
        photobleach = """photobleach_method = "global-exponential"
photobleach_background_level = 100.0
photobleach_min_r_squared = 0.999
photobleach_min_drop_fraction = 0.05
photobleach_max_gain = 10.0
"""
    path.write_text(
        f'''[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.1
frame_step = 2

[analysis]
lag_min_frames = 1
lag_max_frames = 8
lags_per_decade = 6
max_pairs_per_lag = 32
q_bins = 8
window = "none"
subtract_frame_mean = true
workers = 1
pair_batch_size = 4
{photobleach}
[fit]
enabled = false

[output]
directory = "{output}"
''',
        encoding="utf-8",
    )
    return path


def test_pipeline_writes_trace_and_correction_suppresses_static_bleach_artifact(
    tmp_path: Path,
):
    rng = np.random.default_rng(31415)
    signal_image = 20.0 + rng.normal(scale=4.0, size=(16, 18))
    source_time_s = np.arange(40) * 0.1
    frames = 100.0 + np.exp(-0.4 * source_time_s)[:, None, None] * signal_image
    frames = frames.astype(np.float32)
    movie = tmp_path / "movie.npy"
    np.save(movie, frames)

    raw_config = load_config(
        _write_pipeline_config(tmp_path / "raw.toml", output="raw-output", correction=False)
    )
    corrected_config = load_config(
        _write_pipeline_config(
            tmp_path / "corrected.toml",
            output="corrected-output",
            correction=True,
        )
    )
    raw_output = run_pipeline(raw_config)
    corrected_output = run_pipeline(corrected_config)

    with np.load(raw_output / "ddm_radial.npz") as payload:
        raw_structure = payload["structure_function"].copy()
    with np.load(corrected_output / "ddm_radial.npz") as payload:
        corrected_structure = payload["structure_function"].copy()
    finite = np.isfinite(raw_structure) & np.isfinite(corrected_structure)
    assert np.max(corrected_structure[finite]) < np.max(raw_structure[finite]) * 1e-7

    with (raw_output / "photobleach_trace.csv").open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    with (corrected_output / "photobleach_trace.csv").open(newline="", encoding="utf-8") as handle:
        corrected_rows = list(csv.DictReader(handle))
    assert len(raw_rows) == len(corrected_rows) == 20
    assert raw_rows[0]["gain"] == "1.0"
    assert raw_rows[0]["fitted_raw_mean"] == ""
    assert corrected_rows[-1]["source_frame_index"] == "38"
    assert float(corrected_rows[-1]["time_s"]) == pytest.approx(3.8)

    raw_provenance = json.loads((raw_output / "provenance.json").read_text(encoding="utf-8"))
    assert raw_provenance["photobleaching"]["method"] == "none"
    assert raw_provenance["photobleaching"]["applied"] is False
    provenance = json.loads((corrected_output / "provenance.json").read_text(encoding="utf-8"))
    bleaching = provenance["photobleaching"]
    assert bleaching["applied"] is True
    assert bleaching["rate_s_inv"] == pytest.approx(0.4, rel=2e-6)
    assert bleaching["r_squared_log_mean"] > 0.999999
    assert bleaching["trace_file"] == "photobleach_trace.csv"
    np.testing.assert_array_equal(np.load(movie), frames)


def test_pipeline_applies_and_hashes_flatfield(tmp_path: Path):
    flatfield = np.arange(1, 17, dtype=np.float32).reshape(4, 4) + 10.0
    levels = np.arange(1, 9, dtype=np.float32)
    frames = 10.0 + levels[:, None, None] * (flatfield - 10.0)
    np.save(tmp_path / "movie.npy", frames)
    tifffile.imwrite(tmp_path / "flat.tif", flatfield)
    config_path = tmp_path / "flatfield.toml"
    config_path.write_text(
        '''[input]
path = "movie.npy"
flatfield_path = "flat.tif"
flatfield_dark_level = 10.0
pixel_size_um = 0.5
frame_interval_s = 0.2

[analysis]
lag_max_frames = 3
q_bins = 4
window = "none"
workers = 1

[fit]
enabled = false

[output]
directory = "flatfield-output"
''',
        encoding="utf-8",
    )

    output = run_pipeline(load_config(config_path))
    with np.load(output / "ddm_radial.npz") as payload:
        structure = payload["structure_function"]
    finite = np.isfinite(structure)
    assert np.all(structure[finite] == 0)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["calibration"]["flatfield_applied"] is True
    assert provenance["input"]["flatfield"]["dark_level"] == 10.0
    assert len(provenance["input"]["flatfield"]["sha256"]) == 64
