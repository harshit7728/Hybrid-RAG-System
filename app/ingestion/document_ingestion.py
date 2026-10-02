import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk

from app.services.pdf_service import extract_pdf_pages
from app.services.chunking_service import create_chunks
from app.services.embedding_service import generate_embeddings


async def ingest_pdf(
    db: AsyncSession,
    filename: str,
    pdf_bytes: bytes
):

    document = Document(
        filename=filename,
        status="processing"
    )

    db.add(document)

    await db.flush()

    try:

        # 1. Extract PDF text
        pages = extract_pdf_pages(pdf_bytes)

        if not pages:
            raise ValueError(
                "No extractable text found in PDF"
            )

        # 2. Create chunks
        chunks = create_chunks(pages)

        if not chunks:
            raise ValueError(
                "No chunks generated"
            )

        # 3. Generate embeddings in batches
        batch_size = 50

        for start in range(0, len(chunks), batch_size):

            batch = chunks[start:start + batch_size]

            texts = [
                chunk["content"]
                for chunk in batch
            ]

            embeddings = await generate_embeddings(texts)

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count does not match chunk count"
                )

            for chunk, embedding in zip(batch, embeddings):

                db_chunk = DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    chunk_index=chunks.index(chunk),
                    content=chunk["content"],
                    embedding=embedding,
                    metadata_json={
                        "filename": filename,
                        "page_number": chunk["page_number"]
                    }
                )

                db.add(db_chunk)

        document.status = "processed"

        await db.commit()

        return {
            "document_id": str(document.id),
            "filename": filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "status": "processed"
        }

    except Exception:

        await db.rollback()
        raise