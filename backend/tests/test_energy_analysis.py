import numpy as np
import pandas as pd
import pytest

from src.pipeline.energy_analysis import (
    classify_energy_state,
    cliffs_delta,
    effect_size_eta_squared_from_kruskal,
    get_clustered_only,
    holm_adjust,
)


def test_classify_energy_state():
    assert classify_energy_state(0.10) == "near_stable"
    assert classify_energy_state(0.30) == "moderately_unstable"
    assert classify_energy_state(0.60) == "highly_unstable"
    assert classify_energy_state(np.nan) == "unknown"


def test_get_clustered_only():
    df = pd.DataFrame({"cluster": [1, 2, -1, np.nan, 3]})
    result = get_clustered_only(df)
    assert len(result) == 3
    assert -1 not in result["cluster"].values
    assert not result["cluster"].isna().any()


def test_effect_size_eta_squared_from_kruskal():
    assert effect_size_eta_squared_from_kruskal(10.0, 100, 3) > 0.0
    assert np.isnan(effect_size_eta_squared_from_kruskal(10.0, 2, 3))


def test_cliffs_delta():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert cliffs_delta(x, y) == -1.0

    x = np.array([6, 5, 4])
    y = np.array([3, 2, 1])
    assert cliffs_delta(x, y) == 1.0

    assert np.isnan(cliffs_delta(np.array([]), np.array([1, 2])))


def test_holm_adjust():
    p_values = [0.01, 0.04, 0.03, 0.005]
    adjusted = holm_adjust(p_values)
    
    assert len(adjusted) == 4
    assert adjusted[3] == 0.02  # 0.005 * 4
    assert adjusted[0] == 0.03  # max(0.02, 0.01 * 3)
