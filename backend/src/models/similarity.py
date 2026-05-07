import numpy as np
import pandas as pd


def build_neighbor_table(
    matrix: np.ndarray,
    ids: list[str],
    metric_name: str,
    top_k: int = 10,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    rows = []

    for i, query_id in enumerate(ids):
        scores = matrix[i]

        if higher_is_better:
            neighbor_idx = np.argsort(scores)[::-1][:top_k]
        else:
            neighbor_idx = np.argsort(scores)[:top_k]

        for rank, j in enumerate(neighbor_idx, start=1):
            rows.append(
                {
                    "query_structure_id": query_id,
                    "neighbor_rank": rank,
                    "neighbor_structure_id": ids[j],
                    "metric": metric_name,
                    "score": float(scores[j]),
                }
            )

    return pd.DataFrame(rows)


def get_structure_metadata(structure_id: str, source_df: pd.DataFrame) -> dict:
    row = source_df.loc[source_df["structure_id"] == structure_id]
    if row.empty:
        raise ValueError(f"structure_id not found: {structure_id}")
    return row.iloc[0].to_dict()


def get_top_neighbors(query_structure_id: str, metric_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    result = (
        metric_df.loc[metric_df["query_structure_id"] == query_structure_id]
        .sort_values("neighbor_rank")
        .head(k)
        .reset_index(drop=True)
    )
    return result


def compare_query_and_neighbors(
    query_structure_id: str,
    metric_df: pd.DataFrame,
    retrieval_source: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    query_meta = get_structure_metadata(query_structure_id, retrieval_source)
    neighbors = get_top_neighbors(query_structure_id, metric_df, k=k).copy()

    neighbors["query_energy"] = query_meta.get("energy")
    neighbors["query_delta_energy"] = query_meta.get("delta_energy")
    neighbors["query_cluster_label"] = query_meta.get("cluster_label")
    neighbors["query_lower_rotation"] = query_meta.get("lower_rotation")
    neighbors["query_displacement"] = query_meta.get("displacement")
    neighbors["query_upper_rotation"] = query_meta.get("upper_rotation")

    neighbors["same_cluster"] = neighbors["cluster_label"] == query_meta.get("cluster_label")
    neighbors["energy_difference"] = neighbors["energy"] - query_meta.get("energy", 0)
    neighbors["delta_energy_difference"] = neighbors["delta_energy"] - query_meta.get("delta_energy", 0)
    neighbors["lower_rotation_difference"] = neighbors["lower_rotation"] - query_meta.get("lower_rotation", 0)
    neighbors["displacement_difference"] = neighbors["displacement"] - query_meta.get("displacement", 0)
    neighbors["upper_rotation_difference"] = neighbors["upper_rotation"] - query_meta.get("upper_rotation", 0)

    return neighbors


def get_structures_in_motif(cluster_label: int, motif_df: pd.DataFrame) -> pd.DataFrame:
    return (
        motif_df[motif_df["cluster_label"] == cluster_label]
        .sort_values(["delta_energy", "energy"], ascending=[True, True])
        .reset_index(drop=True)
    )


def get_lowest_energy_structures_in_motif(
    cluster_label: int, top_n: int = 10, ranking_df: pd.DataFrame = None
) -> pd.DataFrame:
    if ranking_df is None:
        raise ValueError("ranking_df must be provided")
    return (
        ranking_df[ranking_df["cluster_label"] == cluster_label]
        .sort_values("rank_within_cluster")
        .head(top_n)
        .reset_index(drop=True)
    )


def get_motif_representatives(
    cluster_label: int, representatives_df: pd.DataFrame
) -> pd.DataFrame:
    return (
        representatives_df[representatives_df["cluster_label"] == cluster_label]
        .sort_values("representative_type")
        .reset_index(drop=True)
    )


def build_query_validation_summary(
    query_ids: list[str],
    metric_df: pd.DataFrame,
    retrieval_source: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    rows = []

    for qid in query_ids:
        query_meta = get_structure_metadata(qid, retrieval_source)
        neighbors = compare_query_and_neighbors(qid, metric_df, retrieval_source, k=k).copy()

        row = {
            "query_structure_id": qid,
            "query_cluster_label": query_meta.get("cluster_label"),
            "query_energy": query_meta.get("energy"),
            "query_delta_energy": query_meta.get("delta_energy"),
            "query_lower_rotation": query_meta.get("lower_rotation"),
            "query_displacement": query_meta.get("displacement"),
            "query_upper_rotation": query_meta.get("upper_rotation"),
            "top_k": k,
            "same_cluster_count": int(neighbors["same_cluster"].sum()) if "same_cluster" in neighbors else 0,
            "same_cluster_fraction": float(neighbors["same_cluster"].mean()) if "same_cluster" in neighbors else 0.0,
            "mean_neighbor_score": float(neighbors["score"].mean()),
            "min_neighbor_score": float(neighbors["score"].min()),
            "max_neighbor_score": float(neighbors["score"].max()),
            "mean_neighbor_energy": float(neighbors["energy"].mean()) if "energy" in neighbors else np.nan,
            "mean_neighbor_delta_energy": float(neighbors["delta_energy"].mean()) if "delta_energy" in neighbors else np.nan,
            "mean_abs_energy_difference": float(neighbors["energy_difference"].abs().mean()) if "energy_difference" in neighbors else np.nan,
            "mean_abs_delta_energy_difference": float(neighbors["delta_energy_difference"].abs().mean()) if "delta_energy_difference" in neighbors else np.nan,
            "mean_abs_lower_rotation_difference": float(neighbors["lower_rotation_difference"].abs().mean()) if "lower_rotation_difference" in neighbors else np.nan,
            "mean_abs_displacement_difference": float(neighbors["displacement_difference"].abs().mean()) if "displacement_difference" in neighbors else np.nan,
            "mean_abs_upper_rotation_difference": float(neighbors["upper_rotation_difference"].abs().mean()) if "upper_rotation_difference" in neighbors else np.nan,
        }

        rows.append(row)

    return pd.DataFrame(rows)
