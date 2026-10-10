from pathlib import Path

import numpy as np
import pytest

from active_nematics_ddm.cli import _validate_batch_configs
from active_nematics_ddm.config import load_config
from active_nematics_ddm.pipeline import run_pipeline, validate_output_location


def test_output_cannot_overlap_input_directory(tmp_path: Path):
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    np.save(sequence / "frame1.npy", np.zeros((8, 8), dtype=np.uint8))
    np.save(sequence / "frame2.npy", np.ones((8, 8), dtype=np.uint8))
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(
        '''[input]
path = "sequence"
pixel_size_um = 0.5
frame_interval_s = 0.2

[output]
directory = "sequence"
overwrite = true
''',
        encoding="utf-8",
    )
    config = load_config(config_path)
    with pytest.raises(ValueError, match="must not overlap the input data"):
        validate_output_location(config)
    assert (sequence / "frame1.npy").is_file()


def test_output_cannot_contain_source_config(tmp_path: Path):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(
        '''[input]
path = "movie.npy"
pixel_size_um = 0.5
frame_interval_s = 0.2

[output]
directory = "."
overwrite = true
''',
        encoding="utf-8",
    )
    config = load_config(config_path)
    with pytest.raises(ValueError, match="cannot contain the source config"):
        validate_output_location(config)
    assert config_path.is_file()


def test_batch_rejects_duplicate_output_directories(tmp_path: Path):
    configs = []
    for index in range(2):
        movie = tmp_path / f"movie{index}.npy"
        np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
        config_path = tmp_path / f"run{index}.toml"
        config_path.write_text(
            f'''[input]
path = "{movie.name}"
pixel_size_um = 0.5
frame_interval_s = 0.2

[output]
directory = "shared-output"
''',
            encoding="utf-8",
        )
        configs.append(load_config(config_path))
    with pytest.raises(ValueError, match="batch output directories overlap"):
        _validate_batch_configs(configs)


def test_invalid_worker_override_is_rejected_before_output_creation(tmp_path: Path):
    movie = tmp_path / "movie.npy"
    np.save(movie, np.zeros((3, 8, 8), dtype=np.uint8))
    config_path = tmp_path / "workers.toml"
    config_path.write_text(
        '''[input]
path = "movie.npy"
pixel_size_um = 0.5
frame_interval_s = 0.2

[output]
directory = "new-output"
''',
        encoding="utf-8",
    )
    config = load_config(config_path)
    with pytest.raises(ValueError, match="workers"):
        run_pipeline(config, workers=0)
    assert not config.output.directory.exists()
