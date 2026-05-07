import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def evaluate_kmeans_silhouette(X: np.ndarray, n_clusters: int = 6, random_state: int = 42) -> tuple[float, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return float(score), labels


def cluster_energy_stats(
    labels: np.ndarray, delta_energy: np.ndarray, representation_name: str | None = None
) -> tuple[pd.DataFrame, float]:
    temp = pd.DataFrame(
        {
            "cluster": labels,
            "delta_energy": delta_energy,
        }
    )

    summary = (
        temp.groupby("cluster")
        .agg(
            count=("delta_energy", "count"),
            mean_delta_energy=("delta_energy", "mean"),
            std_delta_energy=("delta_energy", "std"),
        )
        .reset_index()
    )

    if representation_name is not None:
        summary["representation"] = representation_name

    weighted_std = np.average(summary["std_delta_energy"].fillna(0), weights=summary["count"])
    return summary, float(weighted_std)


def mean_neighbor_delta_gap(X: np.ndarray, delta_energy: np.ndarray, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)

    _, indices = nn.kneighbors(X)

    gaps = []
    for i in range(len(X)):
        nbr_idx = indices[i, 1:]  # skip self
        gap = np.mean(np.abs(delta_energy[i] - delta_energy[nbr_idx]))
        gaps.append(gap)

    return float(np.mean(gaps))


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, delta_energy: np.ndarray) -> tuple[float, float]:
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    energy_valid = delta_energy[mask]

    if len(np.unique(labels_valid)) < 2:
        return np.nan, np.nan

    sil = silhouette_score(X_valid, labels_valid)

    temp = pd.DataFrame(
        {
            "cluster": labels_valid,
            "delta_energy": energy_valid,
        }
    )

    summary = (
        temp.groupby("cluster")
        .agg(
            count=("delta_energy", "count"),
            std_energy=("delta_energy", "std"),
        )
        .reset_index()
    )

    weighted_std = np.average(summary["std_energy"].fillna(0), weights=summary["count"])

    return float(sil), float(weighted_std)
