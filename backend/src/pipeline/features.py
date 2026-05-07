from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from tqdm.auto import tqdm

from src.models.gnn_encoder import SSLGNN


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


def load_graph_embeddings_dataset(processed_graphs_path: str | Path) -> list[Data]:
    with open(processed_graphs_path, "rb") as f:
        raw_graphs = pickle.load(f)
    dataset = [graph_dict_to_pyg_data(g) for g in raw_graphs]
    return dataset


class GraphAugmenter:
    def __init__(
        self,
        feature_mask_prob: float = 0.10,
        edge_dropout_prob: float = 0.15,
        pos_noise_std: float = 0.02,
    ) -> None:
        self.feature_mask_prob = feature_mask_prob
        self.edge_dropout_prob = edge_dropout_prob
        self.pos_noise_std = pos_noise_std

    def __call__(self, batch: Data) -> Data:
        out = copy.deepcopy(batch)

        if self.feature_mask_prob > 0:
            feature_mask = torch.rand_like(out.x) < self.feature_mask_prob
            out.x = out.x.masked_fill(feature_mask, 0.0)

        if self.pos_noise_std > 0:
            noise = torch.randn_like(out.pos_norm) * self.pos_noise_std
            out.x[:, -3:] = out.x[:, -3:] + noise

        if self.edge_dropout_prob > 0 and out.edge_index.numel() > 0:
            edge_index_new, edge_mask = dropout_edge(
                out.edge_index,
                p=self.edge_dropout_prob,
                force_undirected=False,
                training=True,
            )
            out.edge_index = edge_index_new
            out.edge_attr = out.edge_attr[edge_mask]

        return out


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -9e15)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    loss = F.cross_entropy(similarity, targets)
    return loss


def train_one_epoch(
    model: SSLGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    augmenter: GraphAugmenter,
    temperature: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        batch_view_1 = augmenter(batch)
        batch_view_2 = augmenter(batch)

        _, proj_1 = model(batch_view_1)
        _, proj_2 = model(batch_view_2)

        loss = nt_xent_loss(proj_1, proj_2, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch.num_graphs
        total_loss += float(loss.item()) * batch_size
        total_graphs += batch_size

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def extract_embeddings(
    model: SSLGNN, dataset: list[Data], batch_size: int, device: torch.device
) -> tuple[pd.DataFrame, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    all_embeddings = []

    for batch in tqdm(loader, desc="Extracting embeddings"):
        batch = batch.to(device)
        z = model.encode(batch).cpu().numpy()
        batch_size_actual = batch.num_graphs

        structure_ids = batch.structure_id
        relative_paths = batch.relative_cif_path

        lower_rotation = batch.lower_rotation.cpu().numpy()
        displacement = batch.displacement.cpu().numpy()
        upper_rotation = batch.upper_rotation.cpu().numpy()
        energy = batch.y_energy.cpu().numpy().reshape(-1)
        delta_energy = batch.y_delta_energy.cpu().numpy().reshape(-1)

        for i in range(batch_size_actual):
            row = {
                "structure_id": structure_ids[i],
                "relative_cif_path": relative_paths[i],
                "lower_rotation": float(lower_rotation[i]),
                "displacement": float(displacement[i]),
                "upper_rotation": float(upper_rotation[i]),
                "energy": float(energy[i]),
                "delta_energy": float(delta_energy[i]),
            }
            for j in range(z.shape[1]):
                row[f"emb_{j:03d}"] = float(z[i, j])
            rows.append(row)

        all_embeddings.append(z)

    embedding_df = pd.DataFrame(rows)
    embedding_matrix = np.vstack(all_embeddings)
    return embedding_df, embedding_matrix
