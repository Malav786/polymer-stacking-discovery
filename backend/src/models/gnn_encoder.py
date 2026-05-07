from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        projection_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(conv_mlp, edge_dim=hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.embedding_head = MLP([hidden_dim, hidden_dim, embedding_dim], dropout=dropout)
        self.projection_head = MLP([embedding_dim, hidden_dim, projection_dim], dropout=dropout)

    def encode(self, data: Data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, e)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, data.batch)
        z = self.embedding_head(g)
        return z

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.projection_head(z)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(data)
        p = self.project(z)
        return z, p