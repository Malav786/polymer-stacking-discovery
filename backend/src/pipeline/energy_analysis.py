import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STABLE_ENERGY = -936.6398
NEAR_STABLE_MAX = 0.15
MODERATELY_UNSTABLE_MAX = 0.50


def classify_energy_state(delta_e: float) -> str:
    if pd.isna(delta_e):
        return "unknown"
    if delta_e <= NEAR_STABLE_MAX:
        return "near_stable"
    if delta_e <= MODERATELY_UNSTABLE_MAX:
        return "moderately_unstable"
    return "highly_unstable"


def get_clustered_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["cluster"].notna() & (df["cluster"] != -1)].copy()


def effect_size_eta_squared_from_kruskal(h_stat: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return max((h_stat - k + 1) / (n - k), 0.0)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return (gt - lt) / (len(x) * len(y))


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = min((m - rank) * p[idx], 1.0)

    sorted_adj = adjusted[order].copy()
    for i in range(1, m):
        sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])

    adjusted[order] = np.clip(sorted_adj, 0, 1)
    return adjusted.tolist()


def build_phase_g_master_table(phase_f_primary_csv: Path | str) -> pd.DataFrame:
    df = pd.read_csv(phase_f_primary_csv).copy()

    df = df.rename(
        columns={
            "cluster_label": "cluster",
            "cluster_confidence": "cluster_confidence",
        }
    )

    if "energy" in df.columns:
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    elif "energy_master" in df.columns:
        df["energy"] = pd.to_numeric(df["energy_master"], errors="coerce")
    else:
        raise ValueError("No usable energy column found.")

    if "delta_energy" in df.columns:
        df["delta_energy"] = pd.to_numeric(df["delta_energy"], errors="coerce")
    elif "delta_energy_master" in df.columns:
        df["delta_energy"] = pd.to_numeric(df["delta_energy_master"], errors="coerce")
    else:
        df["delta_energy"] = df["energy"] - STABLE_ENERGY

    df["stable_energy"] = STABLE_ENERGY
    df["energy_state"] = df["delta_energy"].apply(classify_energy_state)

    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")

    if "cluster_confidence" in df.columns:
        df["cluster_confidence"] = pd.to_numeric(df["cluster_confidence"], errors="coerce")
    else:
        df["cluster_confidence"] = np.nan

    df["is_clustered"] = df["cluster"].notna()
    df["is_noise"] = df["cluster"].fillna(-999).astype("Int64") == -1

    return df


def build_cluster_representatives(df: pd.DataFrame, top_clusters: list[int]) -> pd.DataFrame:
    rows = []

    for cluster_id in top_clusters:
        df_c = df[df["cluster"].astype(int) == int(cluster_id)].copy()
        if df_c.empty:
            continue

        mean_delta = df_c["delta_energy"].mean()
        df_c["distance_to_cluster_mean_delta"] = (df_c["delta_energy"] - mean_delta).abs()

        rep_lowest = df_c.nsmallest(1, "delta_energy").iloc[0]
        rep_typical = df_c.nsmallest(1, "distance_to_cluster_mean_delta").iloc[0]
        rep_highest = df_c.nlargest(1, "delta_energy").iloc[0]

        rows.extend(
            [
                {
                    "cluster": cluster_id,
                    "representative_type": "lowest_energy",
                    "structure_id": rep_lowest["structure_id"],
                    "relative_cif_path": rep_lowest.get("relative_cif_path"),
                    "energy": rep_lowest["energy"],
                    "delta_energy": rep_lowest["delta_energy"],
                    "lower_rotation": rep_lowest.get("lower_rotation"),
                    "displacement": rep_lowest.get("displacement"),
                    "upper_rotation": rep_lowest.get("upper_rotation"),
                },
                {
                    "cluster": cluster_id,
                    "representative_type": "most_typical_energy",
                    "structure_id": rep_typical["structure_id"],
                    "relative_cif_path": rep_typical.get("relative_cif_path"),
                    "energy": rep_typical["energy"],
                    "delta_energy": rep_typical["delta_energy"],
                    "lower_rotation": rep_typical.get("lower_rotation"),
                    "displacement": rep_typical.get("displacement"),
                    "upper_rotation": rep_typical.get("upper_rotation"),
                },
                {
                    "cluster": cluster_id,
                    "representative_type": "highest_energy",
                    "structure_id": rep_highest["structure_id"],
                    "relative_cif_path": rep_highest.get("relative_cif_path"),
                    "energy": rep_highest["energy"],
                    "delta_energy": rep_highest["delta_energy"],
                    "lower_rotation": rep_highest.get("lower_rotation"),
                    "displacement": rep_highest.get("displacement"),
                    "upper_rotation": rep_highest.get("upper_rotation"),
                },
            ]
        )

    return pd.DataFrame(rows)
