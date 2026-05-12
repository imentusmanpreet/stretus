"""
Text embedding utilities for semantic search.
"""
from __future__ import annotations


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    This is a placeholder implementation.
    In production, this would use a proper embedding model like sentence-transformers.
    """
    # Return placeholder embeddings
    # ChromaDB will use its default embedding function if these aren't used
    return [[0.0] * 384 for _ in texts]  # 384-dimensional embeddings
