from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from src.models.gnn_encoder import SSLGNN


# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
MODEL_PATH = Path("outputs/graph_embeddings/models/ssl_gnn_best.pt")
GRAPHS_PATH = Path("outputs/processed_graphs/all_graphs.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 128
EMBEDDING_DIM = 128
PROJECTION_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.10


# -----------------------------------------------------------------------------
# Utilities from your notebook
# -----------------------------------------------------------------------------
def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_dict_to_pyg_data(graph: dict[str, Any]) -> Data:
    x = torch.tensor(graph["x"], dtype=torch.float32)
    pos = torch.tensor(graph["pos"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph["edge_attr"], dtype=torch.float32)

    pos_centered = pos - pos.mean(dim=0, keepdim=True)
    pos_scale = pos_centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    pos_norm = pos_centered / pos_scale

    x_full = torch.cat([x, pos_norm], dim=1)

    metadata = graph.get("metadata", {})
    targets = graph.get("targets", {})

    data = Data(
        x=x_full,
        pos=pos,
        pos_norm=pos_norm,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_energy=torch.tensor([float(targets.get("energy", np.nan))], dtype=torch.float32),
        y_delta_energy=torch.tensor([float(targets.get("delta_energy", np.nan))], dtype=torch.float32),
    )

    data.structure_id = str(graph["structure_id"])
    data.relative_cif_path = graph.get("relative_cif_path")
    data.lower_rotation = float(metadata.get("lower_rotation", np.nan))
    data.displacement = float(metadata.get("displacement", np.nan))
    data.upper_rotation = float(metadata.get("upper_rotation", np.nan))

    return data


@lru_cache(maxsize=1)
def load_dataset() -> list[Data]:
    if not GRAPHS_PATH.exists():
        raise FileNotFoundError(f"Processed graphs file not found: {GRAPHS_PATH}")

    raw_graphs = load_pickle(GRAPHS_PATH)
    dataset = [graph_dict_to_pyg_data(graph) for graph in raw_graphs]
    return dataset


def find_graph_by_structure_id(structure_id: str) -> Data:
    dataset = load_dataset()

    for data in dataset:
        if getattr(data, "structure_id", None) == structure_id:
            return data

    raise KeyError(f"Structure graph not found: {structure_id}")


@lru_cache(maxsize=1)
def load_model() -> SSLGNN:
    dataset = load_dataset()
    if not dataset:
        raise ValueError("Dataset is empty")

    sample = dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]

    model = SSLGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


@torch.no_grad()
def infer_embedding_for_structure(structure_id: str) -> list[float]:
    model = load_model()
    data = find_graph_by_structure_id(structure_id)

    data = data.to(DEVICE)

    # For single graph inference, PyG expects batch info.
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)

    embedding = model.encode(data)

    if embedding.ndim > 1:
        embedding = embedding.squeeze(0)

    return embedding.detach().cpu().numpy().astype(float).tolist()