import numpy as np
import pytest

from active_nematics_ddm.ddm import compute_radial_ddm
from active_nematics_ddm.fit import fit_relaxation
from active_nematics_ddm.lags import logarithmic_lags


def test_static_movie_has_zero_structure_function():
    rng = np.random.default_rng(1)
    frame = rng.normal(size=(32, 40)).astype(np.float32)
    frames = np.repeat(frame[None, ...], 16, axis=0)
    result = compute_radial_ddm(
        frames,
        pixel_size_um=0.5,
        frame_interval_s=0.2,
        lags=np.array([1, 2, 5]),
        q_bins=12,
        workers=2,
    )
    finite = np.isfinite(result.structure_function)
    assert np.all(result.structure_function[finite] == 0)
    np.testing.assert_array_equal(result.pair_count, [15, 14, 11])


def test_parallel_and_serial_results_match():
    rng = np.random.default_rng(2)
    frames = rng.normal(size=(24, 24, 28)).astype(np.float32)
    options = dict(
        pixel_size_um=0.4,
        frame_interval_s=0.1,
        lags=np.array([1, 3, 7, 12]),
        q_bins=10,
        max_pairs_per_lag=9,
        pair_batch_size=3,
    )
    serial = compute_radial_ddm(frames, workers=1, **options)
    parallel = compute_radial_ddm(frames, workers=3, **options)
    np.testing.assert_allclose(serial.structure_function, parallel.structure_function)
    np.testing.assert_allclose(serial.origin_sem, parallel.origin_sem)
    np.testing.assert_array_equal(serial.pair_count, parallel.pair_count)


def test_q_and_time_calibration():
    frames = np.zeros((8, 16, 20), dtype=np.float32)
    frames[:, 4:8, 5:9] = np.arange(8)[:, None, None]
    result = compute_radial_ddm(
        frames,
        pixel_size_um=0.25,
        frame_interval_s=0.4,
        lags=np.array([1, 3]),
        q_bins=8,
    )
    np.testing.assert_allclose(result.lag_s, [0.4, 1.2])
    assert result.q_edges_um_inv[-1] <= np.pi / 0.25 + 1e-12


def test_rfft_radial_average_matches_full_fft_plane():
    rng = np.random.default_rng(9)
    frames = rng.normal(size=(7, 15, 20)).astype(np.float32)
    result = compute_radial_ddm(
        frames,
        pixel_size_um=0.6,
        frame_interval_s=0.2,
        lags=np.array([2]),
        q_bins=9,
        window="none",
        subtract_frame_mean=False,
        workers=1,
    )

    transform = np.fft.fft2(frames, axes=(-2, -1), norm="ortho")
    power = np.mean(np.abs(transform[2:] - transform[:-2]) ** 2, axis=0)
    qy = 2 * np.pi * np.fft.fftfreq(frames.shape[1], d=0.6)
    qx = 2 * np.pi * np.fft.fftfreq(frames.shape[2], d=0.6)
    q_grid = np.hypot(qy[:, None], qx[None, :])
    bins = np.searchsorted(result.q_edges_um_inv, q_grid.ravel(), side="right") - 1
    bins[np.isclose(q_grid.ravel(), result.q_edges_um_inv[-1])] = len(result.q_um_inv) - 1
    valid = (bins >= 0) & (bins < len(result.q_um_inv))
    sums = np.bincount(bins[valid], weights=power.ravel()[valid], minlength=len(result.q_um_inv))
    counts = np.bincount(bins[valid], minlength=len(result.q_um_inv))
    expected = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    np.testing.assert_allclose(result.structure_function[0], expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(result.modes_per_bin, counts)


def test_ou_fourier_modes_recover_diffusive_q_squared_scaling():
    rng = np.random.default_rng(123)
    n_frames, height, width = 2_500, 32, 32
    diffusion_px2_per_frame = 0.13
    mode_numbers = np.array([2, 4])
    q_truth = 2 * np.pi * mode_numbers / width
    coefficients = np.empty((len(q_truth), n_frames))
    for mode_index, q_value in enumerate(q_truth):
        rho = np.exp(-diffusion_px2_per_frame * q_value**2)
        coefficients[mode_index, 0] = rng.normal()
        innovation = np.sqrt(1 - rho**2)
        for frame_index in range(1, n_frames):
            coefficients[mode_index, frame_index] = (
                rho * coefficients[mode_index, frame_index - 1] + innovation * rng.normal()
            )

    x = np.arange(width)
    patterns = np.array(
        [
            np.repeat(np.cos(2 * np.pi * mode * x / width)[None, :], height, axis=0)
            for mode in mode_numbers
        ]
    )
    frames = np.einsum("mt,myx->tyx", coefficients, patterns).astype(np.float32)
    frames += rng.normal(scale=0.08, size=frames.shape).astype(np.float32)
    lags = logarithmic_lags(n_frames, maximum=200, per_decade=12)
    result = compute_radial_ddm(
        frames,
        pixel_size_um=1.0,
        frame_interval_s=1.0,
        lags=lags,
        q_bins=32,
        window="none",
        max_pairs_per_lag=900,
        workers=2,
    )
    fits = fit_relaxation(result)
    indices = np.array([np.argmin(np.abs(result.q_um_inv - q)) for q in q_truth])
    expected_tau = 1.0 / (diffusion_px2_per_frame * q_truth**2)
    assert fits.accepted[indices].all()
    np.testing.assert_allclose(fits.tau_s[indices], expected_tau, rtol=0.25)
    measured_rate_ratio = fits.tau_s[indices[0]] / fits.tau_s[indices[1]]
    assert measured_rate_ratio == pytest.approx(4.0, rel=0.35)
