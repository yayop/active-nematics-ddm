from pathlib import Path

import numpy as np
import pytest
import tifffile

from active_nematics_ddm.config import load_config
from active_nematics_ddm.io import inspect_input, read_frames


def test_npy_selection_crop_and_bin(tmp_path: Path):
    source = np.arange(10 * 8 * 12, dtype=np.uint16).reshape(10, 8, 12)
    path = tmp_path / "movie.npy"
    np.save(path, source)
    frames = read_frames(
        path,
        frame_start=1,
        frame_stop=9,
        frame_step=2,
        roi=(0, 8, 2, 10),
        spatial_bin=2,
    )
    assert frames.shape == (4, 4, 4)
    assert frames.dtype == np.float32
    metadata = inspect_input(path)
    assert metadata["frame_count"] == 10


def test_config_resolves_relative_paths(tmp_path: Path):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.2

[output]
directory = "out"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.input.path == movie.resolve()
    assert config.output.directory == (tmp_path / "out").resolve()


def test_directory_sequence_uses_natural_sort(tmp_path: Path):
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    for name, value in (("frame10.tif", 10), ("frame2.tif", 2), ("frame1.tif", 1)):
        tifffile.imwrite(sequence / name, np.full((6, 8), value, dtype=np.uint8))
    frames = read_frames(sequence)
    np.testing.assert_array_equal(frames[:, 0, 0], [1, 2, 10])
    metadata = inspect_input(sequence)
    assert metadata["frame_count"] == 3
    assert metadata["first_file"] == "frame1.tif"


def test_multichannel_ome_tiff_is_rejected_instead_of_mixing_time_and_channel(tmp_path: Path):
    path = tmp_path / "multichannel.ome.tif"
    data = np.zeros((3, 2, 8, 10), dtype=np.uint8)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "TCYX"})
    with pytest.raises(ValueError, match="ambiguous TIFF axes TCYX"):
        read_frames(path)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ('overwrite = "false"', "output.overwrite must be true or false"),
        ("overwrite = false", None),
    ],
)
def test_config_parses_booleans_strictly(tmp_path: Path, setting: str, expected: str | None):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "strict.toml"
    config_path.write_text(
        f'''[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.2

[output]
directory = "out"
{setting}
''',
        encoding="utf-8",
    )
    if expected is None:
        assert load_config(config_path).output.overwrite is False
    else:
        with pytest.raises(ValueError, match=expected):
            load_config(config_path)


@pytest.mark.parametrize("bad_value", ["1.9", '"2"'])
def test_config_rejects_non_integer_frame_step(tmp_path: Path, bad_value: str):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "bad-step.toml"
    config_path.write_text(
        f'''[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.2
frame_step = {bad_value}

[output]
directory = "out"
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input.frame_step must be an integer"):
        load_config(config_path)


def test_config_rejects_nonfinite_calibration(tmp_path: Path):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "nonfinite.toml"
    config_path.write_text(
        '''[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = inf

[output]
directory = "out"
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input.frame_interval_s must be a finite number"):
        load_config(config_path)
