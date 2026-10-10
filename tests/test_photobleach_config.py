from pathlib import Path

import numpy as np
import pytest

from active_nematics_ddm.config import load_config


def _write_config(tmp_path: Path, analysis: str = "") -> Path:
    np.save(tmp_path / "movie.npy", np.zeros((8, 8, 8), dtype=np.uint8))
    path = tmp_path / "run.toml"
    path.write_text(
        f"""[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.2

[analysis]
{analysis}

[output]
directory = "out"
""",
        encoding="utf-8",
    )
    return path


def test_legacy_config_defaults_to_no_photobleach_correction(tmp_path: Path):
    config = load_config(_write_config(tmp_path))
    assert config.analysis.photobleach_method == "none"
    assert config.analysis.photobleach_background_level is None


@pytest.mark.parametrize(
    ("analysis", "message"),
    [
        (
            'photobleach_method = "global-exponential"',
            "photobleach_background_level is required",
        ),
        ('photobleach_method = "linear"', "photobleach_method must be"),
        ("photobleach_background_level = 10.0", "only valid when"),
        (
            'photobleach_method = "global-exponential"\nphotobleach_background_level = inf',
            "must be finite",
        ),
        (
            'photobleach_method = "global-exponential"\nphotobleach_background_level = true',
            "must be finite",
        ),
        ("photobleach_min_r_squared = nan", "must be finite"),
        ("photobleach_min_r_squared = true", "must be finite"),
        ("photobleach_min_r_squared = 1.1", "must lie between"),
        ("photobleach_min_drop_fraction = 1.0", r"must lie in \[0, 1\)"),
        ("photobleach_max_gain = 0.9", "must be at least one"),
        (
            "photobleach_min_drop_fraction = 0.5\nphotobleach_max_gain = 1.5",
            "gates are incompatible",
        ),
    ],
)
def test_photobleach_config_rejects_unsafe_settings(tmp_path: Path, analysis: str, message: str):
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, analysis))


def test_global_exponential_config_requires_and_accepts_finite_background(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            """photobleach_method = "global-exponential"
photobleach_background_level = -5.0
photobleach_min_r_squared = 0.8
photobleach_min_drop_fraction = 0.01
photobleach_max_gain = 4.0""",
        )
    )
    assert config.analysis.photobleach_background_level == -5.0
