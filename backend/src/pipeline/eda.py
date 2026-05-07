import numpy as np
import pandas as pd


def summarize_series(s: pd.Series, name: str) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return {
            "metric": name,
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }

    return {
        "metric": name,
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }


def iqr_outlier_summary(series: pd.Series, metric_name: str) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = s[(s < lower_bound) | (s > upper_bound)]

    return {
        "metric": metric_name,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(outliers.shape[0]),
        "outlier_fraction": float(outliers.shape[0] / s.shape[0]) if s.shape[0] > 0 else np.nan,
    }
