from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# -----------------------------------------------------------------------------
# Main tables
# -----------------------------------------------------------------------------
class Structure(Base):
    __tablename__ = "structures"

    structure_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    relative_cif_path: Mapped[str] = mapped_column(String(255), nullable=False)

    lower_rotation: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    displacement: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    upper_rotation: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    energy: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    stable_energy: Mapped[float] = mapped_column(Float, nullable=False, default=-936.6398)
    delta_energy: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    embedding: Mapped[Optional["Embedding"]] = relationship(
        back_populates="structure",
        uselist=False,
        cascade="all, delete-orphan",
    )

    cluster_assignment: Mapped[Optional["ClusterAssignment"]] = relationship(
        back_populates="structure",
        uselist=False,
        cascade="all, delete-orphan",
    )

    map_points: Mapped[List["EmbeddingMapPoint"]] = relationship(
        back_populates="structure",
        cascade="all, delete-orphan",
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    structure_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("structures.structure_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Stored as JSON string: [0.123, -0.456, ...]
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=128)

    structure: Mapped["Structure"] = relationship(back_populates="embedding")


class ClusterAssignment(Base):
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    structure_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("structures.structure_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    cluster_label: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    structure: Mapped["Structure"] = relationship(back_populates="cluster_assignment")


class MotifSummary(Base):
    __tablename__ = "motif_summary"

    cluster_label: Mapped[int] = mapped_column(Integer, primary_key=True)
    motif_name: Mapped[str] = mapped_column(String(100), nullable=False)
    mean_delta_energy: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class EmbeddingMapPoint(Base):
    __tablename__ = "embedding_map_points"
    __table_args__ = (
        UniqueConstraint("structure_id", "method", name="uq_structure_method"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    structure_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("structures.structure_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    method: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)

    structure: Mapped["Structure"] = relationship(back_populates="map_points")


class Neighbor(Base):
    __tablename__ = "neighbors"
    __table_args__ = (
        UniqueConstraint(
            "query_structure_id",
            "neighbor_structure_id",
            "distance_metric",
            name="uq_neighbor_pair_metric",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    query_structure_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("structures.structure_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    neighbor_structure_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("structures.structure_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    distance_metric: Mapped[str] = mapped_column(String(20), nullable=False, default="cosine")


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
from src.core.config import settings

def get_engine(db_path: Path | str | None = None):
    db_file = Path(db_path) if db_path else settings.db_path
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_file}", echo=False, future=True)


def get_session_factory(db_path: Path | str | None = None):
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def create_all_tables(db_path: Path | str | None = None) -> None:
    db_file = Path(db_path) if db_path else settings.db_path
    engine = get_engine(db_file)
    Base.metadata.create_all(engine)
    print(f"Database schema created at: {db_file}")


if __name__ == "__main__":
    create_all_tables()