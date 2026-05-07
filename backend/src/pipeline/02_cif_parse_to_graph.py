from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.io import read
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

ATOM_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
}

NODE_FEATURE_NAMES = ["is_C", "is_H", "is_O", "layer_label"]
EDGE_FEATURE_NAMES = ["distance", "is_intralayer", "is_interlayer"]


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path
    max_structures: int | None = None
    save_packed_dataset: bool = True
    distance_cutoff: float = 4.5
    include_self_loops: bool = False
    undirected_graph: bool = True
    expected_atom_count: int = 156
    expected_lower_count: int = 78
    expected_upper_count: int = 78

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def output_root(self) -> Path:
        return self.project_root / "outputs"

    @property
    def inventory_csv(self) -> Path:
        return self.output_root / "dataset_inventory.csv"

    @property
    def master_csv(self) -> Path:
        return self.output_root / "dataset_master.csv"

    @property
    def parsed_dir(self) -> Path:
        return self.output_root / "parsed_structures"

    @property
    def atom_table_dir(self) -> Path:
        return self.parsed_dir / "atom_tables"

    @property
    def structure_metadata_csv(self) -> Path:
        return self.parsed_dir / "structure_metadata.csv"

    @property
    def failed_parse_csv(self) -> Path:
        return self.parsed_dir / "failed_parses.csv"

    @property
    def layered_dir(self) -> Path:
        return self.output_root / "layered_structures"

    @property
    def layered_atom_table_dir(self) -> Path:
        return self.layered_dir / "atom_tables"

    @property
    def layer_metadata_csv(self) -> Path:
        return self.layered_dir / "layer_metadata.csv"

    @property
    def layer_validation_csv(self) -> Path:
        return self.layered_dir / "layer_validation.csv"

    @property
    def layer_failures_csv(self) -> Path:
        return self.layered_dir / "layer_failures.csv"

    @property
    def feature_dir(self) -> Path:
        return self.output_root / "structural_features"

    @property
    def feature_csv(self) -> Path:
        return self.feature_dir / "structure_features.csv"

    @property
    def feature_failures_csv(self) -> Path:
        return self.feature_dir / "feature_failures.csv"

    @property
    def graph_dir(self) -> Path:
        return self.output_root / "graphs"

    @property
    def graph_object_dir(self) -> Path:
        return self.graph_dir / "graph_objects"

    @property
    def graph_metadata_csv(self) -> Path:
        return self.graph_dir / "graph_metadata.csv"

    @property
    def graph_failures_csv(self) -> Path:
        return self.graph_dir / "graph_failures.csv"

    @property
    def featured_graph_dir(self) -> Path:
        return self.output_root / "featured_graphs"

    @property
    def featured_graph_object_dir(self) -> Path:
        return self.featured_graph_dir / "graph_objects"

    @property
    def featured_graph_metadata_csv(self) -> Path:
        return self.featured_graph_dir / "featured_graph_metadata.csv"

    @property
    def featured_graph_failures_csv(self) -> Path:
        return self.featured_graph_dir / "featured_graph_failures.csv"

    @property
    def processed_graph_dir(self) -> Path:
        return self.output_root / "processed_graphs"

    @property
    def processed_graph_object_dir(self) -> Path:
        return self.processed_graph_dir / "graph_objects"

    @property
    def dataset_index_csv(self) -> Path:
        return self.processed_graph_dir / "dataset_index.csv"

    @property
    def processing_failures_csv(self) -> Path:
        return self.processed_graph_dir / "processing_failures.csv"

    @property
    def packed_dataset_file(self) -> Path:
        return self.processed_graph_dir / "all_graphs.pkl"


# ============================================================
# Generic utilities
# ============================================================


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def limited(items: list[Path], max_items: int | None) -> list[Path]:
    return items[:max_items] if max_items is not None else items


def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ============================================================
# Parse CIFs
# ============================================================


def resolve_cif_path(relative_path: str, data_root: Path) -> Path:
    return data_root / relative_path


def read_structure(cif_path: Path):
    try:
        atoms = read(str(cif_path))
        if atoms is None or len(atoms) == 0:
            return None, "empty_structure", "No atoms found"
        return atoms, "ok", None
    except Exception as exc:  # noqa: BLE001
        return None, "parse_error", f"{type(exc).__name__}: {exc}"


def get_atom_labels(atoms) -> list[str]:
    if hasattr(atoms, "arrays") and "labels" in atoms.arrays:
        return list(atoms.arrays["labels"])
    return [f"{symbol}{i + 1}" for i, symbol in enumerate(atoms.get_chemical_symbols())]


def build_atom_table(atoms) -> pd.DataFrame:
    cart = atoms.get_positions()
    frac = atoms.get_scaled_positions()
    return pd.DataFrame(
        {
            "atom_index": np.arange(len(atoms), dtype=int),
            "atom_label": get_atom_labels(atoms),
            "symbol": atoms.get_chemical_symbols(),
            "atomic_number": atoms.get_atomic_numbers(),
            "frac_x": frac[:, 0],
            "frac_y": frac[:, 1],
            "frac_z": frac[:, 2],
            "x": cart[:, 0],
            "y": cart[:, 1],
            "z": cart[:, 2],
        }
    )


def build_structure_metadata(atoms, structure_id: str, relative_cif_path: str) -> dict[str, Any]:
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    counts = pd.Series(atoms.get_chemical_symbols()).value_counts().to_dict()
    return {
        "structure_id": structure_id,
        "relative_cif_path": relative_cif_path,
        "atom_count": int(len(atoms)),
        "cell_a": float(cell.lengths()[0]),
        "cell_b": float(cell.lengths()[1]),
        "cell_c": float(cell.lengths()[2]),
        "cell_alpha": float(cell.angles()[0]),
        "cell_beta": float(cell.angles()[1]),
        "cell_gamma": float(cell.angles()[2]),
        "pbc_x": bool(pbc[0]),
        "pbc_y": bool(pbc[1]),
        "pbc_z": bool(pbc[2]),
        "count_C": int(counts.get("C", 0)),
        "count_H": int(counts.get("H", 0)),
        "count_O": int(counts.get("O", 0)),
        "unique_elements": json.dumps(sorted(set(atoms.get_chemical_symbols()))),
    }


def parse_cifs(config: PipelineConfig) -> None:
    ensure_dirs([config.parsed_dir, config.atom_table_dir])
    df_inventory = pd.read_csv(config.inventory_csv)
    if config.max_structures is not None:
        df_inventory = df_inventory.head(config.max_structures).copy()

    metadata_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    iterator = df_inventory.itertuples(index=False)
    for row in tqdm(iterator, total=len(df_inventory), desc="Task 11: parse CIFs"):
        cif_path = resolve_cif_path(row.relative_cif_path, config.data_root)
        atoms, status, error = read_structure(cif_path)

        if status != "ok":
            failure_rows.append(
                {
                    "structure_id": row.structure_id,
                    "relative_cif_path": row.relative_cif_path,
                    "status": status,
                    "error": error,
                }
            )
            continue

        atom_table = build_atom_table(atoms)
        atom_table["structure_id"] = row.structure_id
        atom_table["relative_cif_path"] = row.relative_cif_path
        atom_table.to_csv(config.atom_table_dir / f"{row.structure_id}.csv", index=False)

        metadata_rows.append(
            build_structure_metadata(atoms, row.structure_id, row.relative_cif_path)
        )

    write_csv(metadata_rows, config.structure_metadata_csv)
    write_csv(failure_rows, config.failed_parse_csv)


# ============================================================
#  Assign lower/upper layers
# ============================================================


def assign_layers_by_z(df_atoms: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    required = {"atom_index", "z"}
    missing = required - set(df_atoms.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df_atoms) != config.expected_atom_count:
        raise ValueError(f"Expected {config.expected_atom_count} atoms, found {len(df_atoms)}")

    df_sorted = df_atoms.sort_values(["z", "atom_index"], ascending=[True, True]).reset_index(drop=True)
    df_sorted["z_rank"] = np.arange(len(df_sorted), dtype=int)
    df_sorted["layer_label"] = 1
    df_sorted.loc[: config.expected_lower_count - 1, "layer_label"] = 0
    df_sorted["layer_name"] = df_sorted["layer_label"].map({0: "lower", 1: "upper"})
    return df_sorted


def build_layer_validation(df_layered: pd.DataFrame, structure_id: str, config: PipelineConfig) -> dict[str, Any]:
    lower_mask = df_layered["layer_label"] == 0
    upper_mask = df_layered["layer_label"] == 1
    lower_count = int(lower_mask.sum())
    upper_count = int(upper_mask.sum())
    lower_z_max = float(df_layered.loc[lower_mask, "z"].max())
    upper_z_min = float(df_layered.loc[upper_mask, "z"].min())
    return {
        "structure_id": structure_id,
        "atom_count": int(len(df_layered)),
        "lower_count": lower_count,
        "upper_count": upper_count,
        "lower_ok": lower_count == config.expected_lower_count,
        "upper_ok": upper_count == config.expected_upper_count,
        "total_ok": len(df_layered) == config.expected_atom_count,
        "z_min": float(df_layered["z"].min()),
        "z_max": float(df_layered["z"].max()),
        "lower_z_max": lower_z_max,
        "upper_z_min": upper_z_min,
        "layer_gap": upper_z_min - lower_z_max,
    }


def assign_layers(config: PipelineConfig) -> None:
    ensure_dirs([config.layered_dir, config.layered_atom_table_dir])
    atom_table_files = limited(sorted(config.atom_table_dir.glob("*.csv")), config.max_structures)

    metadata_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for file_path in tqdm(atom_table_files, desc="Task 12: assign layers"):
        df_atoms = None
        try:
            df_atoms = pd.read_csv(file_path)
            structure_id = str(df_atoms["structure_id"].iloc[0])
            relative_cif_path = str(df_atoms["relative_cif_path"].iloc[0])
            df_layered = assign_layers_by_z(df_atoms, config)
            df_layered.to_csv(config.layered_atom_table_dir / file_path.name, index=False)

            metadata_rows.append(
                {
                    "structure_id": structure_id,
                    "relative_cif_path": relative_cif_path,
                    "atom_table_file": file_path.name,
                    "layered_atom_table_file": file_path.name,
                }
            )
            validation_rows.append(build_layer_validation(df_layered, structure_id, config))
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "file_name": file_path.name,
                    "structure_id": None if df_atoms is None else df_atoms.get("structure_id", pd.Series([None])).iloc[0],
                    "relative_cif_path": None if df_atoms is None else df_atoms.get("relative_cif_path", pd.Series([None])).iloc[0],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_csv(metadata_rows, config.layer_metadata_csv)
    write_csv(validation_rows, config.layer_validation_csv)
    write_csv(failure_rows, config.layer_failures_csv)


# ============================================================
#  Structural descriptors
# ============================================================


def validate_layered_table(df: pd.DataFrame, config: PipelineConfig) -> None:
    required = {"atom_index", "symbol", "x", "y", "z", "layer_label", "structure_id", "relative_cif_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df) != config.expected_atom_count:
        raise ValueError(f"Expected {config.expected_atom_count} atoms, found {len(df)}")
    lower_count = int((df["layer_label"] == 0).sum())
    upper_count = int((df["layer_label"] == 1).sum())
    if lower_count != config.expected_lower_count:
        raise ValueError(f"Expected {config.expected_lower_count} lower atoms, found {lower_count}")
    if upper_count != config.expected_upper_count:
        raise ValueError(f"Expected {config.expected_upper_count} upper atoms, found {upper_count}")


def get_positions(df: pd.DataFrame) -> np.ndarray:
    return df[["x", "y", "z"]].to_numpy(dtype=float)


def pairwise_distance_matrix(positions_a: np.ndarray, positions_b: np.ndarray | None = None) -> np.ndarray:
    positions_b = positions_a if positions_b is None else positions_b
    diff = positions_a[:, None, :] - positions_b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def compute_centroid(df: pd.DataFrame) -> np.ndarray:
    return get_positions(df).mean(axis=0)


def compute_center_of_mass(df: pd.DataFrame) -> np.ndarray:
    masses = df["symbol"].map(ATOM_MASSES).to_numpy(dtype=float)
    return np.average(get_positions(df), axis=0, weights=masses)


def summarize_upper_triangle_distances(dist: np.ndarray) -> dict[str, float | int]:
    values = dist[np.triu_indices(dist.shape[0], k=1)]
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_flat_distances(dist: np.ndarray) -> dict[str, float | int]:
    values = dist.ravel()
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_nearest_neighbors(df: pd.DataFrame) -> dict[str, float]:
    dist = pairwise_distance_matrix(get_positions(df))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    return {
        "nn_mean": float(nearest.mean()),
        "nn_std": float(nearest.std()),
        "nn_min": float(nearest.min()),
        "nn_max": float(nearest.max()),
    }


def get_atom_type_counts(df: pd.DataFrame, prefix: str = "") -> dict[str, int]:
    counts = df["symbol"].value_counts().to_dict()
    return {
        f"{prefix}count_C": int(counts.get("C", 0)),
        f"{prefix}count_H": int(counts.get("H", 0)),
        f"{prefix}count_O": int(counts.get("O", 0)),
    }


def build_structure_features(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    validate_layered_table(df, config)
    df_lower = df[df["layer_label"] == 0].copy()
    df_upper = df[df["layer_label"] == 1].copy()

    lower_centroid = compute_centroid(df_lower)
    upper_centroid = compute_centroid(df_upper)
    lower_com = compute_center_of_mass(df_lower)
    upper_com = compute_center_of_mass(df_upper)

    lower_intra = summarize_upper_triangle_distances(pairwise_distance_matrix(get_positions(df_lower)))
    upper_intra = summarize_upper_triangle_distances(pairwise_distance_matrix(get_positions(df_upper)))
    inter = summarize_flat_distances(pairwise_distance_matrix(get_positions(df_lower), get_positions(df_upper)))
    nn_all = summarize_nearest_neighbors(df)
    lower_nn = summarize_nearest_neighbors(df_lower)
    upper_nn = summarize_nearest_neighbors(df_upper)

    return {
        "structure_id": df["structure_id"].iloc[0],
        "relative_cif_path": df["relative_cif_path"].iloc[0],
        "atom_count": int(len(df)),
        "lower_atom_count": int(len(df_lower)),
        "upper_atom_count": int(len(df_upper)),
        **get_atom_type_counts(df),
        **get_atom_type_counts(df_lower, prefix="lower_"),
        **get_atom_type_counts(df_upper, prefix="upper_"),
        "lower_centroid_x": float(lower_centroid[0]),
        "lower_centroid_y": float(lower_centroid[1]),
        "lower_centroid_z": float(lower_centroid[2]),
        "upper_centroid_x": float(upper_centroid[0]),
        "upper_centroid_y": float(upper_centroid[1]),
        "upper_centroid_z": float(upper_centroid[2]),
        "lower_com_x": float(lower_com[0]),
        "lower_com_y": float(lower_com[1]),
        "lower_com_z": float(lower_com[2]),
        "upper_com_x": float(upper_com[0]),
        "upper_com_y": float(upper_com[1]),
        "upper_com_z": float(upper_com[2]),
        "centroid_separation": float(np.linalg.norm(upper_centroid - lower_centroid)),
        "com_separation": float(np.linalg.norm(upper_com - lower_com)),
        "lower_intralayer_dist_count": lower_intra["count"],
        "lower_intralayer_dist_mean": lower_intra["mean"],
        "lower_intralayer_dist_std": lower_intra["std"],
        "lower_intralayer_dist_min": lower_intra["min"],
        "lower_intralayer_dist_max": lower_intra["max"],
        "upper_intralayer_dist_count": upper_intra["count"],
        "upper_intralayer_dist_mean": upper_intra["mean"],
        "upper_intralayer_dist_std": upper_intra["std"],
        "upper_intralayer_dist_min": upper_intra["min"],
        "upper_intralayer_dist_max": upper_intra["max"],
        "interlayer_dist_count": inter["count"],
        "interlayer_dist_mean": inter["mean"],
        "interlayer_dist_std": inter["std"],
        "interlayer_dist_min": inter["min"],
        "interlayer_dist_max": inter["max"],
        **nn_all,
        "lower_nn_mean": lower_nn["nn_mean"],
        "lower_nn_std": lower_nn["nn_std"],
        "lower_nn_min": lower_nn["nn_min"],
        "lower_nn_max": lower_nn["nn_max"],
        "upper_nn_mean": upper_nn["nn_mean"],
        "upper_nn_std": upper_nn["nn_std"],
        "upper_nn_min": upper_nn["nn_min"],
        "upper_nn_max": upper_nn["nn_max"],
    }


def extract_structural_features(config: PipelineConfig) -> None:
    ensure_dirs([config.feature_dir])
    atom_table_files = limited(sorted(config.layered_atom_table_dir.glob("*.csv")), config.max_structures)

    feature_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for file_path in tqdm(atom_table_files, desc="Task 13: structural features"):
        df = None
        try:
            df = pd.read_csv(file_path)
            feature_rows.append(build_structure_features(df, config))
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "file_name": file_path.name,
                    "structure_id": None if df is None else df.get("structure_id", pd.Series([None])).iloc[0],
                    "relative_cif_path": None if df is None else df.get("relative_cif_path", pd.Series([None])).iloc[0],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_csv(feature_rows, config.feature_csv)
    write_csv(failure_rows, config.feature_failures_csv)


# ============================================================
#  Raw graph construction
# ============================================================


def validate_graph_input(df: pd.DataFrame) -> None:
    required = {
        "atom_index",
        "atom_label",
        "symbol",
        "atomic_number",
        "x",
        "y",
        "z",
        "layer_label",
        "structure_id",
        "relative_cif_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty atom table")


def get_edge_type(layer_i: int, layer_j: int) -> int:
    return 0 if layer_i == layer_j else 1


def build_edge_table(df_atoms: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    positions = get_positions(df_atoms)
    dist_matrix = pairwise_distance_matrix(positions)
    atom_indices = df_atoms["atom_index"].to_numpy(dtype=int)
    layer_labels = df_atoms["layer_label"].to_numpy(dtype=int)
    edge_rows: list[dict[str, Any]] = []
    n_atoms = len(df_atoms)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if not config.include_self_loops and i == j:
                continue
            if config.undirected_graph and j <= i:
                continue

            distance = float(dist_matrix[i, j])
            if distance > config.distance_cutoff:
                continue

            edge_type = get_edge_type(int(layer_labels[i]), int(layer_labels[j]))
            forward = {
                "source_atom_index": int(atom_indices[i]),
                "target_atom_index": int(atom_indices[j]),
                "distance": distance,
                "edge_type": edge_type,
                "edge_type_name": "intralayer" if edge_type == 0 else "interlayer",
                "source_layer": int(layer_labels[i]),
                "target_layer": int(layer_labels[j]),
            }
            edge_rows.append(forward)

            if config.undirected_graph:
                edge_rows.append(
                    {
                        "source_atom_index": int(atom_indices[j]),
                        "target_atom_index": int(atom_indices[i]),
                        "distance": distance,
                        "edge_type": edge_type,
                        "edge_type_name": forward["edge_type_name"],
                        "source_layer": int(layer_labels[j]),
                        "target_layer": int(layer_labels[i]),
                    }
                )

    return pd.DataFrame(edge_rows)


def build_node_table(df_atoms: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "atom_index",
        "atom_label",
        "symbol",
        "atomic_number",
        "x",
        "y",
        "z",
        "frac_x",
        "frac_y",
        "frac_z",
        "layer_label",
        "layer_name",
        "structure_id",
        "relative_cif_path",
    ]
    columns = [col for col in preferred if col in df_atoms.columns]
    return df_atoms[columns].copy()


def build_graph_metadata(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, structure_id: str, relative_cif_path: str, config: PipelineConfig) -> dict[str, Any]:
    metadata = {
        "structure_id": structure_id,
        "relative_cif_path": relative_cif_path,
        "node_count": int(len(df_nodes)),
        "edge_count": int(len(df_edges)),
        "intralayer_edge_count": int((df_edges.get("edge_type", pd.Series(dtype=int)) == 0).sum()),
        "interlayer_edge_count": int((df_edges.get("edge_type", pd.Series(dtype=int)) == 1).sum()),
        "distance_cutoff": float(config.distance_cutoff),
        "has_edges": bool(len(df_edges) > 0),
    }
    if not df_edges.empty:
        metadata.update(
            {
                "edge_distance_mean": float(df_edges["distance"].mean()),
                "edge_distance_std": float(df_edges["distance"].std(ddof=0)),
                "edge_distance_min": float(df_edges["distance"].min()),
                "edge_distance_max": float(df_edges["distance"].max()),
            }
        )
    else:
        metadata.update(
            {
                "edge_distance_mean": np.nan,
                "edge_distance_std": np.nan,
                "edge_distance_min": np.nan,
                "edge_distance_max": np.nan,
            }
        )
    return metadata


def attach_structure_targets(metadata: dict[str, Any], df_master: pd.DataFrame) -> dict[str, Any]:
    row = df_master.loc[df_master["structure_id"] == metadata["structure_id"]]
    if row.empty:
        metadata.update(
            {
                "energy": np.nan,
                "delta_energy": np.nan,
                "lower_rotation": np.nan,
                "displacement": np.nan,
                "upper_rotation": np.nan,
            }
        )
        return metadata

    record = row.iloc[0]
    metadata.update(
        {
            "energy": float(record["energy"]) if pd.notna(record["energy"]) else np.nan,
            "delta_energy": float(record["delta_energy"]) if pd.notna(record["delta_energy"]) else np.nan,
            "lower_rotation": float(record["lower_rotation"]) if pd.notna(record["lower_rotation"]) else np.nan,
            "displacement": float(record["displacement"]) if pd.notna(record["displacement"]) else np.nan,
            "upper_rotation": float(record["upper_rotation"]) if pd.notna(record["upper_rotation"]) else np.nan,
        }
    )
    return metadata


def build_raw_graph(df_atoms: pd.DataFrame, df_master: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    validate_graph_input(df_atoms)
    structure_id = str(df_atoms["structure_id"].iloc[0])
    relative_cif_path = str(df_atoms["relative_cif_path"].iloc[0])
    nodes = build_node_table(df_atoms)
    edges = build_edge_table(df_atoms, config)
    metadata = build_graph_metadata(nodes, edges, structure_id, relative_cif_path, config)
    metadata = attach_structure_targets(metadata, df_master)
    return {
        "structure_id": structure_id,
        "relative_cif_path": relative_cif_path,
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }


def build_graphs(config: PipelineConfig) -> None:
    ensure_dirs([config.graph_dir, config.graph_object_dir])
    df_master = pd.read_csv(config.master_csv)
    atom_table_files = limited(sorted(config.layered_atom_table_dir.glob("*.csv")), config.max_structures)

    metadata_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for file_path in tqdm(atom_table_files, desc="Task 14: build graphs"):
        df_atoms = None
        try:
            df_atoms = pd.read_csv(file_path)
            graph = build_raw_graph(df_atoms, df_master, config)
            save_path = config.graph_object_dir / f"{graph['structure_id']}.pkl"
            save_pickle(graph, save_path)
            row = graph["metadata"].copy()
            row["graph_file"] = save_path.name
            metadata_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "file_name": file_path.name,
                    "structure_id": None if df_atoms is None else df_atoms.get("structure_id", pd.Series([None])).iloc[0],
                    "relative_cif_path": None if df_atoms is None else df_atoms.get("relative_cif_path", pd.Series([None])).iloc[0],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_csv(metadata_rows, config.graph_metadata_csv)
    write_csv(failure_rows, config.graph_failures_csv)


# ============================================================
#  Feature graph construction
# ============================================================


def validate_raw_graph(graph: dict[str, Any]) -> None:
    required_keys = {"structure_id", "nodes", "edges", "metadata"}
    missing = required_keys - set(graph.keys())
    if missing:
        raise ValueError(f"Missing graph keys: {sorted(missing)}")

    node_required = {"atom_index", "symbol", "x", "y", "z", "layer_label"}
    node_missing = node_required - set(graph["nodes"].columns)
    if node_missing:
        raise ValueError(f"Missing node columns: {sorted(node_missing)}")

    if not graph["edges"].empty:
        edge_required = {"source_atom_index", "target_atom_index", "distance", "edge_type"}
        edge_missing = edge_required - set(graph["edges"].columns)
        if edge_missing:
            raise ValueError(f"Missing edge columns: {sorted(edge_missing)}")


def build_node_features(df_nodes: pd.DataFrame) -> np.ndarray:
    symbols = df_nodes["symbol"].astype(str).to_numpy()
    layer = df_nodes["layer_label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    is_c = (symbols == "C").astype(np.float32).reshape(-1, 1)
    is_h = (symbols == "H").astype(np.float32).reshape(-1, 1)
    is_o = (symbols == "O").astype(np.float32).reshape(-1, 1)
    return np.hstack([is_c, is_h, is_o, layer]).astype(np.float32)


def build_positions(df_nodes: pd.DataFrame) -> np.ndarray:
    return df_nodes[["x", "y", "z"]].to_numpy(dtype=np.float32)


def build_edge_index(df_edges: pd.DataFrame) -> np.ndarray:
    if df_edges.empty:
        return np.empty((2, 0), dtype=np.int64)
    return df_edges[["source_atom_index", "target_atom_index"]].to_numpy(dtype=np.int64).T


def build_edge_features(df_edges: pd.DataFrame) -> np.ndarray:
    if df_edges.empty:
        return np.empty((0, 3), dtype=np.float32)
    distance = df_edges["distance"].to_numpy(dtype=np.float32).reshape(-1, 1)
    edge_type = df_edges["edge_type"].to_numpy(dtype=int)
    is_intralayer = (edge_type == 0).astype(np.float32).reshape(-1, 1)
    is_interlayer = (edge_type == 1).astype(np.float32).reshape(-1, 1)
    return np.hstack([distance, is_intralayer, is_interlayer]).astype(np.float32)


def build_featured_graph(raw_graph: dict[str, Any]) -> dict[str, Any]:
    validate_raw_graph(raw_graph)
    df_nodes = raw_graph["nodes"].copy()
    df_edges = raw_graph["edges"].copy()
    return {
        "structure_id": raw_graph["structure_id"],
        "relative_cif_path": raw_graph.get("relative_cif_path"),
        "x": build_node_features(df_nodes),
        "pos": build_positions(df_nodes),
        "edge_index": build_edge_index(df_edges),
        "edge_attr": build_edge_features(df_edges),
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "metadata": raw_graph["metadata"].copy(),
    }


def build_featured_graphs(config: PipelineConfig) -> None:
    ensure_dirs([config.featured_graph_dir, config.featured_graph_object_dir])
    graph_files = limited(sorted(config.graph_object_dir.glob("*.pkl")), config.max_structures)

    metadata_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for graph_file in tqdm(graph_files, desc="Task 15: build featured graphs"):
        raw_graph = None
        try:
            raw_graph = load_pickle(graph_file)
            featured_graph = build_featured_graph(raw_graph)
            save_path = config.featured_graph_object_dir / f"{featured_graph['structure_id']}.pkl"
            save_pickle(featured_graph, save_path)

            row = featured_graph["metadata"].copy()
            row.update(
                {
                    "graph_file": graph_file.name,
                    "featured_graph_file": save_path.name,
                    "node_feature_dim": int(featured_graph["x"].shape[1]),
                    "edge_feature_dim": int(featured_graph["edge_attr"].shape[1]),
                    "num_nodes": int(featured_graph["x"].shape[0]),
                    "num_edges": int(featured_graph["edge_attr"].shape[0]),
                }
            )
            metadata_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "graph_file": graph_file.name,
                    "structure_id": None if raw_graph is None else raw_graph.get("structure_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_csv(metadata_rows, config.featured_graph_metadata_csv)
    write_csv(failure_rows, config.featured_graph_failures_csv)


# ============================================================
# Save processed dataset
# ============================================================


def validate_featured_graph(graph: dict[str, Any]) -> None:
    required = {"structure_id", "x", "pos", "edge_index", "edge_attr", "metadata"}
    missing = required - set(graph.keys())
    if missing:
        raise ValueError(f"Missing graph keys: {sorted(missing)}")

    x = graph["x"]
    pos = graph["pos"]
    edge_index = graph["edge_index"]
    edge_attr = graph["edge_attr"]

    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise TypeError(f"x must be a 2D numpy array, got {type(x)} with shape {getattr(x, 'shape', None)}")
    if not isinstance(pos, np.ndarray) or pos.ndim != 2 or pos.shape[1] != 3:
        raise TypeError(f"pos must be [N, 3], got {type(pos)} with shape {getattr(pos, 'shape', None)}")
    if not isinstance(edge_index, np.ndarray) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TypeError(f"edge_index must be [2, E], got {type(edge_index)} with shape {getattr(edge_index, 'shape', None)}")
    if not isinstance(edge_attr, np.ndarray) or edge_attr.ndim != 2:
        raise TypeError(f"edge_attr must be 2D, got {type(edge_attr)} with shape {getattr(edge_attr, 'shape', None)}")
    if x.shape[0] != pos.shape[0]:
        raise ValueError("x and pos must have same number of nodes")
    if edge_index.shape[1] != edge_attr.shape[0]:
        raise ValueError("edge_index and edge_attr must have same number of edges")


def build_processed_graph(featured_graph: dict[str, Any]) -> dict[str, Any]:
    validate_featured_graph(featured_graph)
    metadata = featured_graph["metadata"].copy()
    return {
        "structure_id": featured_graph["structure_id"],
        "relative_cif_path": featured_graph.get("relative_cif_path"),
        "x": featured_graph["x"].astype(np.float32),
        "pos": featured_graph["pos"].astype(np.float32),
        "edge_index": featured_graph["edge_index"].astype(np.int64),
        "edge_attr": featured_graph["edge_attr"].astype(np.float32),
        "node_feature_names": list(featured_graph.get("node_feature_names", [])),
        "edge_feature_names": list(featured_graph.get("edge_feature_names", [])),
        "targets": {
            "energy": metadata.get("energy", np.nan),
            "delta_energy": metadata.get("delta_energy", np.nan),
        },
        "metadata": metadata,
    }


def build_dataset_index_row(processed_graph: dict[str, Any], file_name: str) -> dict[str, Any]:
    metadata = processed_graph["metadata"]
    return {
        "structure_id": processed_graph["structure_id"],
        "relative_cif_path": processed_graph.get("relative_cif_path"),
        "graph_file": file_name,
        "num_nodes": int(processed_graph["x"].shape[0]),
        "num_edges": int(processed_graph["edge_attr"].shape[0]),
        "node_feature_dim": int(processed_graph["x"].shape[1]),
        "edge_feature_dim": int(processed_graph["edge_attr"].shape[1]),
        "energy": metadata.get("energy", np.nan),
        "delta_energy": metadata.get("delta_energy", np.nan),
        "lower_rotation": metadata.get("lower_rotation", np.nan),
        "displacement": metadata.get("displacement", np.nan),
        "upper_rotation": metadata.get("upper_rotation", np.nan),
    }


def save_processed_graphs(config: PipelineConfig) -> None:
    ensure_dirs([config.processed_graph_dir, config.processed_graph_object_dir])
    featured_graph_files = limited(sorted(config.featured_graph_object_dir.glob("*.pkl")), config.max_structures)

    dataset_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    all_graphs: list[dict[str, Any]] = []

    for graph_file in tqdm(featured_graph_files, desc="Task 16: save processed dataset"):
        featured_graph = None
        try:
            featured_graph = load_pickle(graph_file)
            processed_graph = build_processed_graph(featured_graph)
            save_path = config.processed_graph_object_dir / f"{processed_graph['structure_id']}.pkl"
            save_pickle(processed_graph, save_path)
            dataset_rows.append(build_dataset_index_row(processed_graph, save_path.name))
            if config.save_packed_dataset:
                all_graphs.append(processed_graph)
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(
                {
                    "input_file": graph_file.name,
                    "structure_id": None if featured_graph is None else featured_graph.get("structure_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_csv(dataset_rows, config.dataset_index_csv)
    write_csv(failure_rows, config.processing_failures_csv)
    if config.save_packed_dataset:
        save_pickle(all_graphs, config.packed_dataset_file)


# ============================================================
# Main
# ============================================================

def main() -> None:
    config = PipelineConfig(project_root=Path.cwd().resolve())

    print("=" * 70)
    print("PHASE C PIPELINE: CIF -> LAYERS -> FEATURES -> GRAPHS -> DATASET")
    print("=" * 70)
    print(f"Project root      : {config.project_root}")
    print(f"Data root         : {config.data_root}")
    print(f"Output root       : {config.output_root}")
    print(f"Distance cutoff   : {config.distance_cutoff}")
    print(f"Max structures    : {config.max_structures}")
    print("=" * 70)

    parse_cifs(config)
    assign_layers(config)
    extract_structural_features(config)
    build_graphs(config)
    build_featured_graphs(config)
    save_processed_graphs(config)

    print("\nPipeline complete.")
    print(f"Processed dataset index: {config.dataset_index_csv}")
    if config.save_packed_dataset:
        print(f"Packed dataset        : {config.packed_dataset_file}")


if __name__ == "__main__":
    main()