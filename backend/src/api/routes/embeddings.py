from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.db.schema import ClusterAssignment, EmbeddingMapPoint, Structure
from src.api.deps import get_db
from src.api.schemas import InferenceEmbeddingResponse, EmbeddingMapItem
from src.models.inference import infer_embedding_for_structure

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]

@router.get("/infer-embedding/{structure_id}", response_model=InferenceEmbeddingResponse)
def infer_embedding(structure_id: str, db: DbSession):
    structure = db.query(Structure).filter(Structure.structure_id == structure_id).first()
    if structure is None:
        raise HTTPException(status_code=404, detail="Structure not found")

    try:
        embedding = infer_embedding_for_structure(structure_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Processed graph not found for structure")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return InferenceEmbeddingResponse(
        structure_id=structure_id,
        embedding_dim=len(embedding),
        embedding=embedding,
    )

@router.get("/embedding-map", response_model=list[EmbeddingMapItem])
def get_embedding_map(
    db: DbSession,
    method: str = Query(default="umap", pattern="^(umap|pca|tsne)$"),
    limit: int = Query(default=5000, ge=1, le=10000),
):
    rows = (
        db.query(EmbeddingMapPoint, Structure, ClusterAssignment)
        .join(
            Structure,
            EmbeddingMapPoint.structure_id == Structure.structure_id,
        )
        .outerjoin(
            ClusterAssignment,
            EmbeddingMapPoint.structure_id == ClusterAssignment.structure_id,
        )
        .filter(EmbeddingMapPoint.method == method)
        .limit(limit)
        .all()
    )

    return [
        EmbeddingMapItem(
            structure_id=map_point.structure_id,
            method=map_point.method,
            x=map_point.x,
            y=map_point.y,
            cluster_label=cluster.cluster_label if cluster else None,
            delta_energy=structure.delta_energy,
        )
        for map_point, structure, cluster in rows
    ]
