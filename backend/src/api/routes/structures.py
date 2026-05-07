import json
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from src.core.config import settings
from src.db.schema import (
    ClusterAssignment,
    Embedding,
    EmbeddingMapPoint,
    Neighbor,
    Structure,
)
from src.api.deps import get_db
from src.api.schemas import (
    StructureListItem,
    StructureDetailResponse,
    StructureViewResponse,
    NeighborItem,
    MapPointResponse,
    StructureNeighborResponse
)

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]

def parse_embedding_json(embedding_json: str | None) -> list[float] | None:
    if not embedding_json:
        return None
    return json.loads(embedding_json)

@router.get("/structures", response_model=list[StructureListItem])
def get_structures(
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    rows = (
        db.query(Structure)
        .order_by(Structure.structure_id)
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        StructureListItem(
            structure_id=row.structure_id,
            relative_cif_path=row.relative_cif_path,
            lower_rotation=row.lower_rotation,
            displacement=row.displacement,
            upper_rotation=row.upper_rotation,
            energy=row.energy,
            stable_energy=row.stable_energy,
            delta_energy=row.delta_energy,
        )
        for row in rows
    ]

@router.get("/structure/{structure_id}", response_model=StructureDetailResponse)
def get_structure(structure_id: str, db: DbSession):
    structure = db.query(Structure).filter(Structure.structure_id == structure_id).first()
    if structure is None:
        raise HTTPException(status_code=404, detail="Structure not found")

    cluster = db.query(ClusterAssignment).filter(ClusterAssignment.structure_id == structure_id).first()
    embedding = db.query(Embedding).filter(Embedding.structure_id == structure_id).first()

    return StructureDetailResponse(
        structure_id=structure.structure_id,
        relative_cif_path=structure.relative_cif_path,
        lower_rotation=structure.lower_rotation,
        displacement=structure.displacement,
        upper_rotation=structure.upper_rotation,
        energy=structure.energy,
        stable_energy=structure.stable_energy,
        delta_energy=structure.delta_energy,
        cluster_label=cluster.cluster_label if cluster else None,
        cluster_confidence=cluster.confidence if cluster else None,
        embedding_dim=embedding.embedding_dim if embedding else None,
        embedding=parse_embedding_json(embedding.embedding_json) if embedding else None,
    )

@router.get("/structure/{structure_id}/cif", response_class=PlainTextResponse)
def get_structure_cif(structure_id: str, db: DbSession):
    structure = db.query(Structure).filter(Structure.structure_id == structure_id).first()
    if structure is None:
        raise HTTPException(status_code=404, detail="Structure not found")

    cif_path = settings.data_dir / structure.relative_cif_path

    if not cif_path.exists():
        raise HTTPException(status_code=404, detail=f"CIF file not found at: {cif_path}")

    return cif_path.read_text(encoding="utf-8", errors="ignore")

@router.get("/structure-view/{structure_id}", response_model=StructureViewResponse)
def get_structure_view(
    structure_id: str,
    db: DbSession,
    neighbor_limit: int = Query(default=10, ge=1, le=50),
):
    structure = db.query(Structure).filter(Structure.structure_id == structure_id).first()
    if structure is None:
        raise HTTPException(status_code=404, detail="Structure not found")

    cluster = db.query(ClusterAssignment).filter(ClusterAssignment.structure_id == structure_id).first()

    map_rows = (
        db.query(EmbeddingMapPoint)
        .filter(EmbeddingMapPoint.structure_id == structure_id)
        .order_by(EmbeddingMapPoint.method.asc())
        .all()
    )

    neighbor_rows = (
        db.query(Neighbor, Structure, ClusterAssignment)
        .join(Structure, Neighbor.neighbor_structure_id == Structure.structure_id)
        .outerjoin(ClusterAssignment, Neighbor.neighbor_structure_id == ClusterAssignment.structure_id)
        .filter(Neighbor.query_structure_id == structure_id)
        .order_by(Neighbor.rank.asc())
        .limit(neighbor_limit)
        .all()
    )

    neighbors = [
        StructureNeighborResponse(
            neighbor_structure_id=neighbor.neighbor_structure_id,
            rank=neighbor.rank,
            similarity_score=neighbor.similarity_score,
            distance_metric=neighbor.distance_metric,
            cluster_label=neighbor_cluster.cluster_label if neighbor_cluster else None,
            delta_energy=neighbor_structure.delta_energy,
            relative_cif_path=neighbor_structure.relative_cif_path,
        )
        for neighbor, neighbor_structure, neighbor_cluster in neighbor_rows
    ]

    map_points = [
        MapPointResponse(method=row.method, x=row.x, y=row.y)
        for row in map_rows
    ]

    return StructureViewResponse(
        structure_id=structure.structure_id,
        relative_cif_path=structure.relative_cif_path,
        lower_rotation=structure.lower_rotation,
        displacement=structure.displacement,
        upper_rotation=structure.upper_rotation,
        energy=structure.energy,
        stable_energy=structure.stable_energy,
        delta_energy=structure.delta_energy,
        cluster_label=cluster.cluster_label if cluster else None,
        cluster_confidence=cluster.confidence if cluster else None,
        map_points=map_points,
        neighbors=neighbors,
    )

@router.get("/neighbors/{structure_id}", response_model=list[StructureNeighborResponse])
def get_neighbors(
    structure_id: str,
    db: DbSession,
    limit: int = Query(default=10, ge=1, le=100),
):
    structure_exists = db.query(Structure).filter(Structure.structure_id == structure_id).first()
    if structure_exists is None:
        raise HTTPException(status_code=404, detail="Structure not found")

    rows = (
        db.query(Neighbor, Structure, ClusterAssignment)
        .join(Structure, Neighbor.neighbor_structure_id == Structure.structure_id)
        .outerjoin(ClusterAssignment, Neighbor.neighbor_structure_id == ClusterAssignment.structure_id)
        .filter(Neighbor.query_structure_id == structure_id)
        .order_by(Neighbor.rank.asc())
        .limit(limit)
        .all()
    )

    return [
        StructureNeighborResponse(
            neighbor_structure_id=neighbor.neighbor_structure_id,
            rank=neighbor.rank,
            similarity_score=neighbor.similarity_score,
            distance_metric=neighbor.distance_metric,
            cluster_label=neighbor_cluster.cluster_label if neighbor_cluster else None,
            delta_energy=neighbor_structure.delta_energy,
            relative_cif_path=neighbor_structure.relative_cif_path,
        )
        for neighbor, neighbor_structure, neighbor_cluster in rows
    ]
