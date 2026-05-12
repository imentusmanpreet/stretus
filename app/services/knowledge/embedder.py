"""
Embedder module for stock matching with ChromaDB.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "stock_universe"
STOCK_UNIVERSE_RECORD_TYPE = "stock"

_chroma_client = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create a ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ))
    return _chroma_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    This is a placeholder implementation that uses ChromaDB's default embedding function.
    """
    # ChromaDB will handle embeddings internally when we use collection.query()
    # For now, return a simple placeholder that ChromaDB can work with
    # In production, this would use a proper embedding model
    return [[0.0] * 384 for _ in texts]  # 384-dimensional embeddings
