import numpy as np
import pytest

from active_nematics_ddm.lags import logarithmic_lags


def test_logarithmic_lags_are_unique_sorted_and_bounded():
    lags = logarithmic_lags(1_000, minimum=1, maximum=200, per_decade=10)
    assert lags[0] == 1
    assert lags[-1] == 200
    assert np.all(np.diff(lags) > 0)
    assert len(lags) < 200


def test_lags_reject_impossible_range():
    with pytest.raises(ValueError):
        logarithmic_lags(5, minimum=5)
