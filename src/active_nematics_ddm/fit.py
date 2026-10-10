"""Model-light relaxation metrics and empirical DDM fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .ddm import DDMResult


@dataclass(frozen=True)
class RelaxationFits:
    q_um_inv: np.ndarray
    baseline: np.ndarray
    amplitude: np.ndarray
    tau_s: np.ndarray
    beta: np.ndarray
    tau63_s: np.ndarray
    r_squared: np.ndarray
    signal_to_sem: np.ndarray
    success: np.ndarray
    accepted: np.ndarray
    message: tuple[str, ...]


def stretched_structure(
    lag_s: np.ndarray,
    baseline: float,
    amplitude: float,
    tau_s: float,
    beta: float,
) -> np.ndarray:
    """``B + A * (1 - exp(-(lag/tau)**beta))``."""

    lag_s = np.asarray(lag_s, dtype=np.float64)
    return baseline + amplitude * (1.0 - np.exp(-np.power(lag_s / tau_s, beta)))


def empirical_tau63(lag_s: np.ndarray, values: np.ndarray) -> float:
    """Estimate the 63.2% rise time without assuming a stretched exponential."""

    lag_s = np.asarray(lag_s, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(lag_s) & np.isfinite(values) & (lag_s > 0)
    lag_s, values = lag_s[valid], values[valid]
    if len(values) < 6:
        return np.nan
    n_edge = max(2, min(4, len(values) // 4))
    low = float(np.median(values[:n_edge]))
    high = float(np.median(values[-n_edge:]))
    amplitude = high - low
    if not np.isfinite(amplitude) or amplitude <= 0:
        return np.nan
    fraction = np.maximum.accumulate((values - low) / amplitude)
    target = 1.0 - np.exp(-1.0)
    crossing = np.flatnonzero(fraction >= target)
    if len(crossing) == 0:
        return np.nan
    upper = int(crossing[0])
    if upper == 0:
        return float(lag_s[0])
    lower = upper - 1
    f0, f1 = fraction[lower], fraction[upper]
    if f1 <= f0:
        return float(lag_s[upper])
    weight = (target - f0) / (f1 - f0)
    return float(np.exp(np.log(lag_s[lower]) * (1 - weight) + np.log(lag_s[upper]) * weight))


def fit_relaxation(
    result: DDMResult,
    *,
    q_min_um_inv: float | None = None,
    q_max_um_inv: float | None = None,
    beta_bounds: tuple[float, float] = (0.25, 3.0),
    min_points: int = 8,
    min_r_squared: float = 0.8,
    min_signal_to_sem: float = 5.0,
    min_plateau_fraction: float = 0.8,
    max_initial_fraction: float = 0.25,
) -> RelaxationFits:
    """Fit one empirical stretched/compressed relaxation curve per q bin."""

    n_q = len(result.q_um_inv)
    arrays = [np.full(n_q, np.nan, dtype=float) for _ in range(7)]
    baseline, amplitude, tau, beta, tau63, r_squared, signal_to_sem = arrays
    success = np.zeros(n_q, dtype=bool)
    accepted = np.zeros(n_q, dtype=bool)
    messages: list[str] = []

    for index, q_value in enumerate(result.q_um_inv):
        if q_min_um_inv is not None and q_value < q_min_um_inv:
            messages.append("below q fit range")
            continue
        if q_max_um_inv is not None and q_value > q_max_um_inv:
            messages.append("above q fit range")
            continue
        y = result.structure_function[:, index]
        mask = np.isfinite(result.lag_s) & np.isfinite(y) & (result.lag_s > 0)
        t_fit, y_fit = result.lag_s[mask], y[mask]
        tau63[index] = empirical_tau63(t_fit, y_fit)
        if len(y_fit) < min_points or np.ptp(y_fit) <= 0:
            messages.append("insufficient dynamic range")
            continue

        edge = max(2, min(4, len(y_fit) // 4))
        b0 = max(0.0, float(np.median(y_fit[:edge])))
        a0 = max(float(np.median(y_fit[-edge:]) - b0), float(np.ptp(y_fit)), np.finfo(float).eps)
        tau0 = tau63[index] if np.isfinite(tau63[index]) else float(np.median(t_fit))
        p0 = (b0, a0, tau0, 1.0)
        lower = (0.0, 0.0, max(np.min(t_fit) * 1e-3, 1e-12), beta_bounds[0])
        upper = (np.inf, np.inf, np.max(t_fit) * 1e3, beta_bounds[1])
        try:
            parameters, _ = curve_fit(
                stretched_structure,
                t_fit,
                y_fit,
                p0=p0,
                bounds=(lower, upper),
                maxfev=20_000,
            )
            prediction = stretched_structure(t_fit, *parameters)
            residual_sum = float(np.sum((y_fit - prediction) ** 2))
            total_sum = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
            baseline[index], amplitude[index], tau[index], beta[index] = parameters
            r_squared[index] = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
            success[index] = True
            sem = result.origin_sem[:, index][mask]
            positive_sem = sem[np.isfinite(sem) & (sem > 0)]
            if len(positive_sem):
                signal_to_sem[index] = parameters[1] / np.median(positive_sem)
            early_fraction = 1.0 - np.exp(-((t_fit[0] / parameters[2]) ** parameters[3]))
            late_fraction = 1.0 - np.exp(-((t_fit[-1] / parameters[2]) ** parameters[3]))
            flags: list[str] = []
            if not np.isfinite(r_squared[index]) or r_squared[index] < min_r_squared:
                flags.append("low R2")
            if min_signal_to_sem > 0:
                if not np.isfinite(signal_to_sem[index]):
                    flags.append("SEM unavailable")
                elif signal_to_sem[index] < min_signal_to_sem:
                    flags.append("low signal/SEM")
            if early_fraction > max_initial_fraction:
                flags.append("baseline not resolved")
            if late_fraction < min_plateau_fraction:
                flags.append("plateau not resolved")
            beta_margin = 1e-3 * (beta_bounds[1] - beta_bounds[0])
            if parameters[3] <= beta_bounds[0] + beta_margin:
                flags.append("beta at lower bound")
            if parameters[3] >= beta_bounds[1] - beta_margin:
                flags.append("beta at upper bound")
            accepted[index] = not flags
            messages.append("ok" if not flags else "; ".join(flags))
        except (RuntimeError, ValueError, FloatingPointError) as error:
            messages.append(f"fit failed: {error}")

    return RelaxationFits(
        q_um_inv=result.q_um_inv.copy(),
        baseline=baseline,
        amplitude=amplitude,
        tau_s=tau,
        beta=beta,
        tau63_s=tau63,
        r_squared=r_squared,
        signal_to_sem=signal_to_sem,
        success=success,
        accepted=accepted,
        message=tuple(messages),
    )
