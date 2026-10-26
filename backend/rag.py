"""Chunking, embedding, indexing, and retrieval of webpage content."""

import logging
import re
from functools import lru_cache
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  


from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)
from schemas import IndexPageRequest

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    """Clean webpage text and split it into overlapping chunks."""
    return SPLITTER.split_text(clean_text(text))


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Load the embedding model and create the in-memory ChromaDB once."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_page(request: IndexPageRequest) -> tuple[int, str]:
    """Create a new page index while preserving the previous valid index.

    Returns the number of stored chunks and the new index version.
    """
    store = get_vector_store()
    index_version = str(uuid4())

    documents = []

    for source in request.sources:
        for chunk in chunk_text(source.text):
            metadata = {
                "session_id": request.session_id,
                "index_version": index_version,
                "page_title": request.page_title,
            }

            if source.heading:
                metadata["heading"] = source.heading

            documents.append(Document(page_content=chunk, metadata=metadata))

    if not documents:
        raise ValueError("No usable text chunks were created.")

    try:
        store.add_documents(documents)
    except Exception:
        # Remove any partial new index; preserve the previous version.
        try:
            store.delete(
                where={
                    "$and": [
                        {"session_id": {"$eq": request.session_id}},
                        {"index_version": {"$eq": index_version}},
                    ]
                }
            )
        except Exception:
            logger.exception("Failed to remove partial index %s", index_version)

        raise

    # The new index is stored. Now remove older versions for this session.
    # Best effort: retrieval filters by version, so leftovers are never used.
    try:
        store.delete(
            where={
                "$and": [
                    {"session_id": {"$eq": request.session_id}},
                    {"index_version": {"$ne": index_version}},
                ]
            }
        )
    except Exception:
        logger.exception("Failed to remove old indexes for %s", request.session_id)

    return len(documents), index_version


def retrieve_relevant_chunks(
    session_id: str, index_version: str, question: str
) -> list[Document]:
    """Retrieve relevant chunks from the session's active page index."""
    return get_vector_store().similarity_search(
        query=question,
        k=TOP_K,
        filter={
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"index_version": {"$eq": index_version}},
            ]
        },
    )


def format_retrieved_context(documents: list[Document]) -> str:
    """Format retrieved chunks with source identifiers for the LLM."""

    if not documents:
        return "No relevant page content was found."

    return "\n\n---\n\n".join(
        f"[Heading: {document.metadata.get('heading', 'None')}]\n"
        f"{document.page_content}"
        for document in documents
    )
