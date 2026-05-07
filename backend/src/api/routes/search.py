from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.schema import ClusterAssignment, EmbeddingMapPoint, Structure
from src.api.deps import get_db
from src.api.schemas import SearchResultItem

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]

@router.get("/search", response_model=list[SearchResultItem])
def search_structures(
    db: DbSession,
    q: Optional[str] = Query(default=None),
    cluster_label: Optional[int] = Query(default=None),
    min_delta_energy: Optional[float] = Query(default=None),
    max_delta_energy: Optional[float] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    query = (
        db.query(Structure, ClusterAssignment, EmbeddingMapPoint)
        .outerjoin(
            ClusterAssignment,
            Structure.structure_id == ClusterAssignment.structure_id,
        )
        .outerjoin(
            EmbeddingMapPoint,
            (Structure.structure_id == EmbeddingMapPoint.structure_id)
            & (EmbeddingMapPoint.method == "umap"),
        )
    )

    if q:
        like_pattern = f"%{q}%"
        query = query.filter(
            (Structure.structure_id.ilike(like_pattern))
            | (Structure.relative_cif_path.ilike(like_pattern))
        )

    if cluster_label is not None:
        query = query.filter(ClusterAssignment.cluster_label == cluster_label)

    if min_delta_energy is not None:
        query = query.filter(Structure.delta_energy >= min_delta_energy)

    if max_delta_energy is not None:
        query = query.filter(Structure.delta_energy <= max_delta_energy)

    rows = (
        query.order_by(Structure.delta_energy.asc())
        .limit(limit)
        .all()
    )

    return [
        SearchResultItem(
            structure_id=structure.structure_id,
            relative_cif_path=structure.relative_cif_path,
            lower_rotation=structure.lower_rotation,
            displacement=structure.displacement,
            upper_rotation=structure.upper_rotation,
            energy=structure.energy,
            delta_energy=structure.delta_energy,
            cluster_label=cluster.cluster_label if cluster else None,
            umap_x=map_point.x if map_point else None,
            umap_y=map_point.y if map_point else None,
        )
        for structure, cluster, map_point in rows
    ]
