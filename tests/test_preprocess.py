from pathlib import Path

import numpy as np
import pytest

from active_nematics_ddm.preprocess import (
    apply_photobleach_correction,
    save_photobleach_trace,
)


def _exponential_stack(
    *,
    frame_count: int = 41,
    frame_interval_s: float = 0.25,
    background: float = 100.0,
    rate_s_inv: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1729)
    signal_image = 20.0 + rng.normal(scale=3.0, size=(12, 14))
    time_s = np.arange(frame_count) * frame_interval_s
    illumination = np.exp(-rate_s_inv * time_s)
    frames = background + illumination[:, None, None] * signal_image
    return frames.astype(np.float32), signal_image


def test_global_exponential_recovers_rate_and_corrects_in_place():
    background = 100.0
    expected_rate = 0.08
    frames, signal_image = _exponential_stack(
        background=background,
        rate_s_inv=expected_rate,
    )

    diagnostics = apply_photobleach_correction(
        frames,
        frame_interval_s=0.25,
        method="global-exponential",
        background_level=background,
        min_r_squared=0.999,
        min_drop_fraction=0.02,
        max_gain=3.0,
    )

    expected = np.broadcast_to(background + signal_image, frames.shape)
    np.testing.assert_allclose(frames, expected, rtol=2e-6, atol=2e-5)
    assert diagnostics.applied
    assert diagnostics.rate_s_inv == pytest.approx(expected_rate, rel=2e-6)
    assert diagnostics.r_squared_log_mean > 0.999999
    assert diagnostics.gain[0] == pytest.approx(1.0)
    assert diagnostics.maximum_gain == pytest.approx(np.exp(expected_rate * 10.0))


def test_robust_fit_limits_effect_of_single_mean_outlier():
    expected_rate = 0.08
    frames, _ = _exponential_stack(rate_s_inv=expected_rate)
    frames[20] = 100.0 + 1.8 * (frames[20] - 100.0)

    diagnostics = apply_photobleach_correction(
        frames,
        frame_interval_s=0.25,
        method="global-exponential",
        background_level=100.0,
        min_r_squared=0.0,
        min_drop_fraction=0.02,
        max_gain=3.0,
    )

    assert diagnostics.rate_s_inv == pytest.approx(expected_rate, rel=0.02)


def test_none_is_byte_identical_and_trace_is_still_written(tmp_path: Path):
    frames, _ = _exponential_stack()
    original = frames.tobytes()
    diagnostics = apply_photobleach_correction(
        frames,
        frame_interval_s=0.5,
        method="none",
    )

    assert frames.tobytes() == original
    assert not diagnostics.applied
    np.testing.assert_array_equal(diagnostics.corrected_mean, diagnostics.raw_mean)
    trace = tmp_path / "photobleach_trace.csv"
    save_photobleach_trace(
        trace,
        diagnostics,
        source_frame_start=3,
        source_frame_step=2,
    )
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(frames) + 1
    assert lines[0].startswith("selected_frame_index,source_frame_index,time_s")
    assert lines[1].startswith("0,3,0.0")
    assert lines[2].startswith("1,5,0.5")


@pytest.mark.parametrize(
    ("modify", "options", "message"),
    [
        (
            lambda frames: frames.__setitem__(slice(None), frames[0]),
            {"min_drop_fraction": 0.02},
            "drop=",
        ),
        (
            lambda frames: None,
            {"max_gain": 1.1},
            "gain=",
        ),
        (
            lambda frames: frames.__setitem__(
                slice(None),
                100.0
                + (frames - 100.0) * (1.0 + 0.25 * np.sin(np.arange(len(frames)))[:, None, None]),
            ),
            {"min_r_squared": 0.9999, "min_drop_fraction": 0.0},
            "R\\^2=",
        ),
    ],
)
def test_quality_gate_failure_never_mutates_frames(modify, options, message):
    frames, _ = _exponential_stack()
    modify(frames)
    before = frames.copy()
    settings = {
        "frame_interval_s": 0.25,
        "method": "global-exponential",
        "background_level": 100.0,
        "min_r_squared": 0.0,
        "min_drop_fraction": 0.0,
        "max_gain": 3.0,
        **options,
    }

    with pytest.raises(ValueError, match=message):
        apply_photobleach_correction(frames, **settings)

    np.testing.assert_array_equal(frames, before)


def test_background_must_leave_a_positive_mean_and_failure_does_not_mutate():
    frames, _ = _exponential_stack()
    before = frames.copy()
    with pytest.raises(ValueError, match=r"mean\(I\)-background must be positive"):
        apply_photobleach_correction(
            frames,
            frame_interval_s=0.25,
            method="global-exponential",
            background_level=200.0,
        )
    np.testing.assert_array_equal(frames, before)
