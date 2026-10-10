import numpy as np

from active_nematics_ddm.ddm import DDMResult
from active_nematics_ddm.fit import empirical_tau63, fit_relaxation, stretched_structure


def test_empirical_tau63_on_ideal_exponential():
    lag = np.geomspace(0.01, 100, 200)
    values = stretched_structure(lag, 0.2, 3.0, 2.5, 1.0)
    assert abs(empirical_tau63(lag, values) - 2.5) / 2.5 < 0.06


def test_fit_recovers_synthetic_curve():
    lag = np.geomspace(0.05, 30, 60)
    q = np.array([0.5, 1.0])
    truth = [(0.1, 2.0, 1.8, 1.4), (0.2, 1.5, 0.7, 0.9)]
    structure = np.column_stack([stretched_structure(lag, *parameters) for parameters in truth])
    result = DDMResult(
        q_um_inv=q,
        q_edges_um_inv=np.array([0.25, 0.75, 1.25]),
        lag_frames=np.arange(1, len(lag) + 1),
        lag_s=lag,
        structure_function=structure,
        origin_sem=np.ones_like(structure) * 0.01,
        pair_count=np.ones(len(lag), dtype=int) * 100,
        modes_per_bin=np.array([10, 20]),
        image_shape=(32, 32),
        pixel_size_um=0.5,
        frame_interval_s=0.1,
        window="none",
    )
    fits = fit_relaxation(result)
    assert fits.success.all()
    np.testing.assert_allclose(fits.tau_s, [1.8, 0.7], rtol=1e-3)
    np.testing.assert_allclose(fits.beta, [1.4, 0.9], rtol=1e-3)


def test_fit_is_not_accepted_without_sem():
    lag = np.geomspace(0.05, 30, 60)
    structure = stretched_structure(lag, 0.1, 2.0, 1.2, 1.1)[:, None]
    result = DDMResult(
        q_um_inv=np.array([0.7]),
        q_edges_um_inv=np.array([0.5, 0.9]),
        lag_frames=np.arange(1, len(lag) + 1),
        lag_s=lag,
        structure_function=structure,
        origin_sem=np.full_like(structure, np.nan),
        pair_count=np.ones(len(lag), dtype=int) * 20,
        modes_per_bin=np.array([12]),
        image_shape=(32, 32),
        pixel_size_um=0.5,
        frame_interval_s=0.1,
        window="none",
    )
    fits = fit_relaxation(result)
    assert fits.success[0]
    assert not fits.accepted[0]
    assert "SEM unavailable" in fits.message[0]
