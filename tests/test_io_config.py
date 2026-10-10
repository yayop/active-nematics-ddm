from pathlib import Path

import numpy as np
import pytest
import tifffile

from active_nematics_ddm import io
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


def test_flatfield_is_applied_in_float_before_spatial_binning(tmp_path: Path):
    dark = 10.0
    flat = np.arange(1, 17, dtype=np.float32).reshape(4, 4) + dark
    levels = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    movie = dark + levels[:, None, None] * (flat - dark)
    movie_path = tmp_path / "movie.npy"
    flat_path = tmp_path / "flat.tif"
    np.save(movie_path, movie)
    tifffile.imwrite(flat_path, flat)

    frames = read_frames(
        movie_path,
        spatial_bin=2,
        flatfield_path=flat_path,
        flatfield_dark_level=dark,
    )
    expected = levels * np.median(flat - dark)
    np.testing.assert_allclose(frames[:, 0, 0], expected)
    np.testing.assert_allclose(frames.std(axis=(-2, -1)), 0, atol=1e-6)


def test_flatfield_rejects_pixels_at_or_below_dark_level(tmp_path: Path):
    movie_path = tmp_path / "movie.npy"
    flat_path = tmp_path / "flat.tif"
    np.save(movie_path, np.ones((3, 4, 4), dtype=np.float32))
    tifffile.imwrite(flat_path, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="strictly above"):
        read_frames(movie_path, flatfield_path=flat_path, flatfield_dark_level=0)


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


def test_config_resolves_and_validates_flatfield(tmp_path: Path):
    movie = tmp_path / "movie.npy"
    flatfield = tmp_path / "flat.tif"
    np.save(movie, np.ones((3, 8, 8), dtype=np.uint8))
    tifffile.imwrite(flatfield, np.full((8, 8), 20, dtype=np.uint16))
    config_path = tmp_path / "flatfield.toml"
    config_path.write_text(
        '''[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.2
flatfield_path = "flat.tif"
flatfield_dark_level = 5.0

[output]
directory = "out"
''',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.input.flatfield_path == flatfield.resolve()
    assert config.input.flatfield_dark_level == 5.0


def test_config_rejects_dark_level_without_flatfield(tmp_path: Path):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.ones((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "orphan-dark.toml"
    config_path.write_text(
        '''[input]
path = "movie.npy"
pixel_size_um = 0.4
frame_interval_s = 0.2
flatfield_dark_level = 5.0

[output]
directory = "out"
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires input.flatfield_path"):
        load_config(config_path)


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


def test_nd2_reads_only_selected_frames_and_reports_calibration(tmp_path: Path, monkeypatch):
    path = tmp_path / "movie.nd2"
    path.write_bytes(b"fake-nd2")

    class Voxel:
        x = 0.65
        y = 0.65

    class FakeND2:
        requested: list[int] = []

        def __init__(self, source):
            assert Path(source) == path
            self.sizes = {"T": 10, "Y": 8, "X": 12}
            self.shape = (10, 8, 12)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read_frame(self, index):
            self.requested.append(index)
            return np.full((8, 12), index, dtype=np.uint16)

        def voxel_size(self):
            return Voxel()

        def events(self):
            return [{"Time [s]": index * 0.5} for index in range(10)]

    monkeypatch.setattr(io.nd2, "ND2File", FakeND2)
    frames = read_frames(path, frame_start=1, frame_stop=9, frame_step=3, spatial_bin=2)
    np.testing.assert_array_equal(frames[:, 0, 0], [1, 4, 7])
    assert FakeND2.requested == [1, 4, 7]

    metadata = inspect_input(path)
    assert metadata["frame_count"] == 10
    assert metadata["pixel_size_um_xy"] == [0.65, 0.65]
    assert metadata["frame_interval_median_s"] == pytest.approx(0.5)


def test_nd2_rejects_ambiguous_non_time_loops(tmp_path: Path, monkeypatch):
    path = tmp_path / "positions.nd2"
    path.write_bytes(b"fake-nd2")

    class FakeND2:
        sizes = {"T": 3, "P": 2, "Y": 8, "X": 12}

        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(io.nd2, "ND2File", FakeND2)
    with pytest.raises(ValueError, match="ambiguous ND2 axes"):
        read_frames(path)


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
