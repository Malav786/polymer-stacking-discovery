from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import umap

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


NON_EMBEDDING_COLUMNS = {
    "structure_id",
    "relative_cif_path",
    "energy",
    "delta_energy",
    "lower_rotation",
    "displacement",
    "upper_rotation",
    "cluster_label",
    "cluster_confidence",
    "motif_name",
}


def pick_embedding_columns(df: pd.DataFrame, prefix_candidates: tuple[str, ...]) -> list[str]:
    candidates = []

    for col in df.columns:
        col_lower = col.lower()
        if col in NON_EMBEDDING_COLUMNS:
            continue
        if any(col_lower.startswith(prefix) for prefix in prefix_candidates):
            candidates.append(col)

    if candidates:
        return candidates

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in NON_EMBEDDING_COLUMNS]

    if not numeric_cols:
        raise ValueError("No embedding columns detected in embeddings CSV.")

    return numeric_cols


def build_embedding_matrix(df: pd.DataFrame, embedding_cols: list[str]) -> np.ndarray:
    X = df[embedding_cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("Embedding matrix contains NaN values. Clean embeddings first.")
    return X


def zscore_matrix(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    unique = set(labels)
    if -1 in unique:
        unique.remove(-1)
    if len(unique) < 2:
        return None

    mask = labels != -1
    if mask.sum() < 3:
        return None

    return float(silhouette_score(X[mask], labels[mask]))


def assign_motif_names(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    names = []
    for _, row in df.iterrows():
        disp = row.get("displacement_mean", np.nan)
        lower_std = row.get("lower_rotation_std", np.nan)
        upper_std = row.get("upper_rotation_std", np.nan)
        inter = row.get("interlayer_dist_mean_mean", np.nan)
        dE = row.get("delta_energy_mean", np.nan)

        if pd.notna(inter) and inter < df["interlayer_dist_mean_mean"].median():
            compactness = "compact"
        else:
            compactness = "loose"

        if pd.notna(disp) and disp < df["displacement_mean"].median():
            alignment = "aligned"
        else:
            alignment = "shifted"

        rotation_spread = np.nanmean([lower_std, upper_std])
        if pd.notna(rotation_spread) and rotation_spread > np.nanmedian(
            np.nanmean([df["lower_rotation_std"], df["upper_rotation_std"]], axis=0)
        ):
            twist = "twisted"
        else:
            twist = "ordered"

        if pd.notna(dE) and dE <= df["delta_energy_mean"].quantile(0.25):
            stability = "stable"
        elif pd.notna(dE) and dE >= df["delta_energy_mean"].quantile(0.75):
            stability = "unstable"
        else:
            stability = "moderate"

        names.append(f"{alignment} {compactness} {twist} stacking ({stability})")

    df["motif_name"] = names
    return df


def run_dimensionality_reduction(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    random_state: int = 42,
    umap_n_neighbors: int = 25,
    umap_min_dist: float = 0.15,
    umap_metric: str = "euclidean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reducers: dict[str, Any] = {}

    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_coords = pca_2d.fit_transform(X_scaled)
    reducers["pca_2d"] = pca_2d

    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    umap_coords = umap_2d.fit_transform(X_scaled)
    reducers["umap_2d"] = umap_2d

    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    tsne_coords = tsne_2d.fit_transform(X_scaled)
    reducers["tsne_2d"] = tsne_2d

    df_2d = df.copy()
    df_2d["pca_1"] = pca_coords[:, 0]
    df_2d["pca_2"] = pca_coords[:, 1]
    df_2d["umap_1"] = umap_coords[:, 0]
    df_2d["umap_2"] = umap_coords[:, 1]
    df_2d["tsne_1"] = tsne_coords[:, 0]
    df_2d["tsne_2"] = tsne_coords[:, 1]

    return df_2d, reducers


def get_clustering_input(X_scaled: np.ndarray, pca_components: int = 20, random_state: int = 42) -> np.ndarray:
    n_components = min(pca_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X_scaled)


def cluster_hdbscan(X_cluster: np.ndarray, min_cluster_size: int = 35, min_samples: int | None = 10) -> pd.DataFrame:
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed.")

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = model.fit_predict(X_cluster)

    if hasattr(model, "probabilities_"):
        confidence = model.probabilities_
    else:
        confidence = np.full(shape=len(labels), fill_value=np.nan)

    out = pd.DataFrame(
        {
            "cluster_method": "hdbscan",
            "cluster_label": labels.astype(int),
            "cluster_confidence": confidence.astype(float),
        }
    )
    return out


def cluster_kmeans(X_cluster: np.ndarray, k: int = 6, random_state: int = 42) -> pd.DataFrame:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(X_cluster)

    centers = model.cluster_centers_
    dists = np.linalg.norm(X_cluster - centers[labels], axis=1)
    confidence = 1.0 / (1.0 + dists)

    return pd.DataFrame(
        {
            "cluster_method": "kmeans",
            "cluster_label": labels.astype(int),
            "cluster_confidence": confidence.astype(float),
        }
    )


def cluster_gmm(X_cluster: np.ndarray, k: int = 6, random_state: int = 42) -> pd.DataFrame:
    model = GaussianMixture(n_components=k, random_state=random_state)
    model.fit(X_cluster)
    labels = model.predict(X_cluster)
    probs = model.predict_proba(X_cluster).max(axis=1)

    return pd.DataFrame(
        {
            "cluster_method": "gmm",
            "cluster_label": labels.astype(int),
            "cluster_confidence": probs.astype(float),
        }
    )


def cluster_hierarchical(X_cluster: np.ndarray, k: int = 6) -> pd.DataFrame:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X_cluster)

    return pd.DataFrame(
        {
            "cluster_method": "hierarchical",
            "cluster_label": labels.astype(int),
            "cluster_confidence": np.nan,
        }
    )


def summarize_clusters(df_clustered: pd.DataFrame, low_energy_quantile: float = 0.10) -> pd.DataFrame:
    df = df_clustered.copy()

    valid = df[df["cluster_label"] != -1].copy()
    if valid.empty:
        raise ValueError("No non-noise clusters available for motif summary.")

    low_energy_threshold = valid["delta_energy"].quantile(low_energy_quantile)

    def proportion_low_energy(x: pd.Series) -> float:
        return float((x <= low_energy_threshold).mean())

    agg = {
        "structure_id": "count",
        "energy": ["mean", "std", "min", "max"],
        "delta_energy": ["mean", "std", "min", "max", proportion_low_energy],
        "lower_rotation": ["mean", "std"],
        "displacement": ["mean", "std"],
        "upper_rotation": ["mean", "std"],
        "cluster_confidence": ["mean"],
    }

    optional_feature_cols = [
        "centroid_separation",
        "com_separation",
        "interlayer_dist_mean",
        "interlayer_dist_min",
        "interlayer_dist_max",
        "lower_nn_mean",
        "upper_nn_mean",
    ]
    for col in optional_feature_cols:
        if col in valid.columns:
            agg[col] = ["mean", "std"]

    summary = valid.groupby("cluster_label").agg(agg)
    summary.columns = [
        "_".join([str(part) for part in col if str(part)]).replace("<lambda_0>", "low_energy_proportion")
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index().rename(columns={"structure_id_count": "cluster_size"})
    summary = summary.sort_values("cluster_size", ascending=False).reset_index(drop=True)

    summary = assign_motif_names(summary)

    return summary


def select_cluster_representatives(
    df_clustered: pd.DataFrame,
    embedding_cols: list[str],
) -> pd.DataFrame:
    rows = []

    valid = df_clustered[df_clustered["cluster_label"] != -1].copy()

    for cluster_id, group in valid.groupby("cluster_label"):
        group = group.copy().reset_index(drop=True)
        X_group = group[embedding_cols].to_numpy(dtype=float)

        centroid = X_group.mean(axis=0)
        dists = np.linalg.norm(X_group - centroid, axis=1)

        idx_centroid_nearest = int(np.argmin(dists))
        idx_most_unusual = int(np.argmax(dists))
        idx_lowest_energy = int(group["delta_energy"].idxmin() - group.index.min())

        if "cluster_confidence" in group.columns and group["cluster_confidence"].notna().any():
            idx_typical = int(group["cluster_confidence"].fillna(-1).to_numpy().argmax())
        else:
            idx_typical = idx_centroid_nearest

        selected = {
            "cluster_label": cluster_id,
            "centroid_nearest_structure_id": group.iloc[idx_centroid_nearest]["structure_id"],
            "centroid_nearest_relative_cif_path": group.iloc[idx_centroid_nearest].get("relative_cif_path"),
            "lowest_energy_structure_id": group.iloc[idx_lowest_energy]["structure_id"],
            "lowest_energy_relative_cif_path": group.iloc[idx_lowest_energy].get("relative_cif_path"),
            "most_typical_structure_id": group.iloc[idx_typical]["structure_id"],
            "most_typical_relative_cif_path": group.iloc[idx_typical].get("relative_cif_path"),
            "most_unusual_structure_id": group.iloc[idx_most_unusual]["structure_id"],
            "most_unusual_relative_cif_path": group.iloc[idx_most_unusual].get("relative_cif_path"),
            "cluster_size": len(group),
        }

        rows.append(selected)

    reps = pd.DataFrame(rows).sort_values("cluster_label").reset_index(drop=True)

    return reps
