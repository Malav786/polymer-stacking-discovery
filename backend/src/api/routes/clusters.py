from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.db.schema import ClusterAssignment, MotifSummary, Structure
from src.api.deps import get_db
from src.api.schemas import ClusterStructureItem, MotifSummaryItem

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]

@router.get("/cluster/{cluster_label}", response_model=list[ClusterStructureItem])
def get_cluster(
    cluster_label: int,
    db: DbSession,
    limit: int = Query(default=200, ge=1, le=1000),
):
    rows = (
        db.query(Structure, ClusterAssignment)
        .join(
            ClusterAssignment,
            Structure.structure_id == ClusterAssignment.structure_id,
        )
        .filter(ClusterAssignment.cluster_label == cluster_label)
        .order_by(Structure.delta_energy.asc())
        .limit(limit)
        .all()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Cluster not found or empty")

    return [
        ClusterStructureItem(
            structure_id=structure.structure_id,
            relative_cif_path=structure.relative_cif_path,
            delta_energy=structure.delta_energy,
            energy=structure.energy,
            cluster_label=cluster.cluster_label,
            confidence=cluster.confidence,
        )
        for structure, cluster in rows
    ]

@router.get("/motif-summary", response_model=list[MotifSummaryItem])
def get_motif_summary(db: DbSession):
    rows = (
        db.query(MotifSummary)
        .order_by(MotifSummary.mean_delta_energy.asc())
        .all()
    )

    return [
        MotifSummaryItem(
            cluster_label=row.cluster_label,
            motif_name=row.motif_name,
            mean_delta_energy=row.mean_delta_energy,
            description=row.description,
        )
        for row in rows
    ]
