"""Lag-time selection."""

from __future__ import annotations

import numpy as np


def logarithmic_lags(
    n_frames: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
    per_decade: int = 12,
) -> np.ndarray:
    """Return unique integer lags with approximately logarithmic spacing."""

    if n_frames < 2:
        raise ValueError("at least two frames are required")
    if minimum < 1 or per_decade < 1:
        raise ValueError("minimum and per_decade must be positive")
    maximum = min(maximum if maximum is not None else max(1, n_frames // 4), n_frames - 1)
    if maximum < minimum:
        raise ValueError("maximum lag is smaller than minimum lag")
    count = max(2, int(np.ceil(np.log10(maximum / minimum) * per_decade)) + 1)
    lags = np.unique(np.rint(np.geomspace(minimum, maximum, count)).astype(np.int64))
    return lags[(lags >= minimum) & (lags <= maximum)]
