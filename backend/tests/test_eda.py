import numpy as np
import pandas as pd

from src.pipeline.eda import iqr_outlier_summary, summarize_series


def test_summarize_series():
    s = pd.Series([1, 2, 3, 4, 5, np.nan])
    summary = summarize_series(s, "test_metric")

    assert summary["metric"] == "test_metric"
    assert summary["count"] == 5
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["mean"] == 3.0
    assert summary["median"] == 3.0


def test_summarize_series_empty():
    s = pd.Series([np.nan, np.nan])
    summary = summarize_series(s, "empty_metric")

    assert summary["metric"] == "empty_metric"
    assert summary["count"] == 0
    assert np.isnan(summary["min"])


def test_iqr_outlier_summary():
    s = pd.Series([10, 12, 11, 14, 13, 100, -50])
    summary = iqr_outlier_summary(s, "outlier_metric")

    assert summary["metric"] == "outlier_metric"
    assert summary["outlier_count"] == 2
    assert summary["outlier_fraction"] == 2 / 7
    assert summary["q1"] <= summary["q3"]
