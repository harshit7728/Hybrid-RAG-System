import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    Computed,
    func
)

from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR

from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector

from app.db.database import Base
from app.core.config import settings


class Document(Base):

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="processed"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )


class DocumentChunk(Base):

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    embedding = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION),
        nullable=True
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False
    )

    search_vector = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', content)",
            persisted=True
        )
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    document = relationship(
        "Document",
        back_populates="chunks"
    )

    __table_args__ = (

        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            }
        ),

        Index(
            "ix_chunks_search_vector",
            "search_vector",
            postgresql_using="gin"
        ),

        Index(
            "ix_chunks_document_index",
            "document_id",
            "chunk_index"
        )
    )