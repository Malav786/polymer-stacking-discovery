from typing import Optional
from pydantic import BaseModel

class InferenceEmbeddingResponse(BaseModel):
    structure_id: str
    embedding_dim: int
    embedding: list[float]

class StructureListItem(BaseModel):
    structure_id: str
    relative_cif_path: str
    lower_rotation: float
    displacement: float
    upper_rotation: float
    energy: float
    stable_energy: float
    delta_energy: float

class StructureDetailResponse(BaseModel):
    structure_id: str
    relative_cif_path: str
    lower_rotation: float
    displacement: float
    upper_rotation: float
    energy: float
    stable_energy: float
    delta_energy: float
    cluster_label: Optional[int]
    cluster_confidence: Optional[float]
    embedding_dim: Optional[int]
    embedding: Optional[list[float]]

class ClusterStructureItem(BaseModel):
    structure_id: str
    relative_cif_path: str
    delta_energy: float
    energy: float
    cluster_label: int
    confidence: Optional[float]

class NeighborItem(BaseModel):
    neighbor_structure_id: str
    rank: int
    similarity_score: float
    distance_metric: str

class EmbeddingMapItem(BaseModel):
    structure_id: str
    method: str
    x: float
    y: float
    cluster_label: Optional[int]
    delta_energy: Optional[float]

class MapPointResponse(BaseModel):
    method: str
    x: float
    y: float

class StructureNeighborResponse(BaseModel):
    neighbor_structure_id: str
    rank: int
    similarity_score: float
    distance_metric: str
    cluster_label: Optional[int]
    delta_energy: Optional[float]
    relative_cif_path: Optional[str]

class StructureViewResponse(BaseModel):
    structure_id: str
    relative_cif_path: str
    lower_rotation: float
    displacement: float
    upper_rotation: float
    energy: float
    stable_energy: float
    delta_energy: float
    cluster_label: Optional[int]
    cluster_confidence: Optional[float]
    map_points: list[MapPointResponse]
    neighbors: list[StructureNeighborResponse]

class SearchResultItem(BaseModel):
    structure_id: str
    relative_cif_path: str
    lower_rotation: float
    displacement: float
    upper_rotation: float
    energy: float
    delta_energy: float
    cluster_label: Optional[int]
    umap_x: Optional[float]
    umap_y: Optional[float]

class MotifSummaryItem(BaseModel):
    cluster_label: int
    motif_name: str
    mean_delta_energy: float
    description: Optional[str]
