from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.schema import (
    ClusterAssignment,
    Embedding,
    EmbeddingMapPoint,
    MotifSummary,
    Neighbor,
    Structure,
    create_all_tables,
    get_session_factory,
)

from src.core.config import settings

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DB_PATH = settings.db_path

DATASET_MASTER_CSV = settings.outputs_dir / "dataset_master.csv"
STRUCTURE_EMBEDDINGS_CSV = settings.outputs_dir / "graph_embeddings/embeddings/structure_embeddings.csv"

UMAP_CSV = settings.outputs_dir / "graph_embeddings/embeddings/embedding_umap_2d.csv"
PCA_CSV = settings.outputs_dir / "graph_embeddings/embeddings/embedding_pca_2d.csv"
TSNE_CSV = settings.outputs_dir / "graph_embeddings/embeddings/embedding_tsne_2d.csv"

CLUSTERS_CSV = settings.outputs_dir / "motif_discovery/cluster_assignments_primary.csv"
MOTIF_SUMMARY_CSV = settings.outputs_dir / "motif_discovery/motif_summary.csv"
NEIGHBORS_CSV = settings.outputs_dir / "similarity_search/neighbors.csv"

STABLE_ENERGY = -936.6398


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------
def read_csv_required(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(file_path)


def read_csv_optional(path: str) -> Optional[pd.DataFrame]:
    file_path = Path(path)
    if not file_path.exists():
        print(f"[skip] Optional file not found: {path}")
        return None
    return pd.read_csv(file_path)


# -----------------------------------------------------------------------------
# Table reset
# -----------------------------------------------------------------------------
def reset_tables(session: Session) -> None:
    """
    Clear table contents in dependency-safe order.
    """
    session.query(Neighbor).delete()
    session.query(EmbeddingMapPoint).delete()
    session.query(ClusterAssignment).delete()
    session.query(MotifSummary).delete()
    session.query(Embedding).delete()
    session.query(Structure).delete()
    session.commit()


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------
def require_columns(df: pd.DataFrame, required_cols: set[str], file_name: str) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} missing columns: {sorted(missing)}")


# -----------------------------------------------------------------------------
# Load structures
# -----------------------------------------------------------------------------
def load_structures(session: Session, dataset_master_df: pd.DataFrame) -> None:
    require_columns(
        dataset_master_df,
        {
            "structure_id",
            "relative_cif_path",
            "lower_rotation",
            "displacement",
            "upper_rotation",
            "energy",
            "delta_energy",
        },
        "dataset_master.csv",
    )

    records = []
    for row in dataset_master_df.itertuples(index=False):
        records.append(
            Structure(
                structure_id=str(row.structure_id),
                relative_cif_path=str(row.relative_cif_path),
                lower_rotation=float(row.lower_rotation),
                displacement=float(row.displacement),
                upper_rotation=float(row.upper_rotation),
                energy=float(row.energy),
                stable_energy=STABLE_ENERGY,
                delta_energy=float(row.delta_energy),
            )
        )

    session.bulk_save_objects(records)
    session.commit()
    print(f"[ok] Loaded structures: {len(records)}")


# -----------------------------------------------------------------------------
# Load embeddings
# -----------------------------------------------------------------------------
def load_embeddings(session: Session, embeddings_df: pd.DataFrame) -> None:
    if "structure_id" not in embeddings_df.columns:
        raise ValueError("structure_embeddings.csv missing column: structure_id")

    embedding_cols = sorted([col for col in embeddings_df.columns if col.startswith("emb_")])
    if not embedding_cols:
        raise ValueError("No embedding columns found. Expected emb_000, emb_001, ...")

    records = []
    for row in embeddings_df.itertuples(index=False):
        vector = [float(getattr(row, col)) for col in embedding_cols]
        records.append(
            Embedding(
                structure_id=str(row.structure_id),
                embedding_json=json.dumps(vector),
                embedding_dim=len(vector),
            )
        )

    session.bulk_save_objects(records)
    session.commit()
    print(f"[ok] Loaded embeddings: {len(records)} | dim={len(embedding_cols)}")


# -----------------------------------------------------------------------------
# Load 2D projection maps
# -----------------------------------------------------------------------------
def detect_xy_columns(df: pd.DataFrame, method: str) -> tuple[str, str]:
    candidate_map = {
        "umap": [("umap_1", "umap_2"), ("x", "y")],
        "pca": [("pca_1", "pca_2"), ("x", "y")],
        "tsne": [("tsne_1", "tsne_2"), ("x", "y")],
    }

    if method not in candidate_map:
        raise ValueError(f"Unsupported method: {method}")

    for x_col, y_col in candidate_map[method]:
        if {x_col, y_col}.issubset(df.columns):
            return x_col, y_col

    raise ValueError(
        f"Could not find valid coordinate columns for method={method}. "
        f"Available columns: {list(df.columns)}"
    )


def load_embedding_map(session: Session, map_df: pd.DataFrame, method: str) -> None:
    if "structure_id" not in map_df.columns:
        raise ValueError(f"{method} map CSV missing column: structure_id")

    x_col, y_col = detect_xy_columns(map_df, method)

    records = []
    for row in map_df.itertuples(index=False):
        records.append(
            EmbeddingMapPoint(
                structure_id=str(row.structure_id),
                method=method,
                x=float(getattr(row, x_col)),
                y=float(getattr(row, y_col)),
            )
        )

    session.bulk_save_objects(records)
    session.commit()
    print(f"[ok] Loaded map points: {len(records)} | method={method}")


# -----------------------------------------------------------------------------
# Load clusters
# -----------------------------------------------------------------------------
def load_clusters(session: Session, clusters_df: pd.DataFrame) -> None:
    if "structure_id" not in clusters_df.columns:
        raise ValueError("Cluster CSV missing column: structure_id")

    cluster_col = None
    for candidate in ["cluster_label", "cluster", "label"]:
        if candidate in clusters_df.columns:
            cluster_col = candidate
            break

    if cluster_col is None:
        raise ValueError("Cluster CSV missing cluster label column")

    confidence_col = None
    for candidate in ["confidence", "probability", "cluster_confidence", "membership_strength"]:
        if candidate in clusters_df.columns:
            confidence_col = candidate
            break

    records = []
    for row in clusters_df.itertuples(index=False):
        confidence = None
        if confidence_col is not None:
            value = getattr(row, confidence_col)
            if pd.notna(value):
                confidence = float(value)

        records.append(
            ClusterAssignment(
                structure_id=str(row.structure_id),
                cluster_label=int(getattr(row, cluster_col)),
                confidence=confidence,
            )
        )

    session.bulk_save_objects(records)
    session.commit()
    print(f"[ok] Loaded clusters: {len(records)}")


# -----------------------------------------------------------------------------
# Load motif summaries
# -----------------------------------------------------------------------------
def load_motif_summary(session: Session, motif_df: pd.DataFrame) -> None:
    require_columns(
        motif_df,
        {"cluster_label", "motif_name", "mean_delta_energy"},
        "motif_summary.csv",
    )

    description_col = "description" if "description" in motif_df.columns else None

    records = []
    for row in motif_df.itertuples(index=False):
        description = None
        if description_col is not None:
            value = getattr(row, description_col)
            if pd.notna(value):
                description = str(value)

        records.append(
            MotifSummary(
                cluster_label=int(row.cluster_label),
                motif_name=str(row.motif_name),
                mean_delta_energy=float(row.mean_delta_energy),
                description=description,
            )
        )

    session.bulk_save_objects(records)
    session.commit()
    print(f"[ok] Loaded motif summaries: {len(records)}")


# -----------------------------------------------------------------------------
# Load neighbors
# -----------------------------------------------------------------------------
def load_neighbors(session: Session, neighbors_df: pd.DataFrame) -> None:
    require_columns(
        neighbors_df,
        {"query_structure_id", "neighbor_structure_id"},
        "neighbors.csv",
    )

    rank_col = None
    for candidate in ["rank", "neighbor_rank"]:
        if candidate in neighbors_df.columns:
            rank_col = candidate
            break
    if rank_col is None:
        raise ValueError("neighbors.csv missing rank column")

    score_col = None
    for candidate in ["similarity_score", "score", "cosine_similarity", "distance"]:
        if candidate in neighbors_df.columns:
            score_col = candidate
            break
    if score_col is None:
        raise ValueError("neighbors.csv missing similarity/distance score column")

    metric_col = None
    for candidate in ["distance_metric", "metric"]:
        if candidate in neighbors_df.columns:
            metric_col = candidate
            break

    records = []
    for row in neighbors_df.itertuples(index=False):
        metric_value = "cosine"
        if metric_col is not None:
            value = getattr(row, metric_col)
            if pd.notna(value):
                metric_value = str(value)

        records.append(
            Neighbor(
                query_structure_id=str(getattr(row, "query_structure_id")),
                neighbor_structure_id=str(getattr(row, "neighbor_structure_id")),
                rank=int(getattr(row, rank_col)),
                similarity_score=float(getattr(row, score_col)),
                distance_metric=metric_value,
            )
        )

    session.bulk_save_objects(records)
    session.commit()
    print(f"[ok] Loaded neighbors: {len(records)}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    create_all_tables(DB_PATH)
    SessionLocal = get_session_factory(DB_PATH)

    dataset_master_df = read_csv_required(DATASET_MASTER_CSV)
    embeddings_df = read_csv_required(STRUCTURE_EMBEDDINGS_CSV)

    umap_df = read_csv_optional(UMAP_CSV)
    pca_df = read_csv_optional(PCA_CSV)
    tsne_df = read_csv_optional(TSNE_CSV)

    clusters_df = read_csv_optional(CLUSTERS_CSV)
    motif_df = read_csv_optional(MOTIF_SUMMARY_CSV)
    neighbors_df = read_csv_optional(NEIGHBORS_CSV)

    with SessionLocal() as session:
        reset_tables(session)

        load_structures(session, dataset_master_df)
        load_embeddings(session, embeddings_df)

        if umap_df is not None:
            load_embedding_map(session, umap_df, method="umap")

        if pca_df is not None:
            load_embedding_map(session, pca_df, method="pca")

        if tsne_df is not None:
            load_embedding_map(session, tsne_df, method="tsne")

        if clusters_df is not None:
            load_clusters(session, clusters_df)

        if motif_df is not None:
            load_motif_summary(session, motif_df)

        if neighbors_df is not None:
            load_neighbors(session, neighbors_df)

    print("\n[done] Database population completed successfully.")


if __name__ == "__main__":
    main()