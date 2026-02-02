"""
Simple embedding provider - Zero dependency mode.

Creates deterministic embeddings using character-level hashing.
Not as good as neural embeddings but works without any ML libraries.
Useful for testing and lightweight deployments.
"""

import hashlib
from typing import List
import numpy as np


class SimpleEmbeddings:
    """
    Hash-based embeddings - no ML dependencies required.

    Creates deterministic, somewhat-semantic embeddings by:
    1. Hashing n-grams of the text
    2. Creating a sparse-then-dense representation
    3. Normalizing to unit length

    Not as good as sentence-transformers, but works offline with just numpy.
    """

    def __init__(
        self,
        dimension: int = 384,
        ngram_range: tuple = (2, 4),
        **kwargs  # Ignore extra params
    ):
        """
        Initialize simple embeddings.

        Args:
            dimension: Output embedding dimension
            ngram_range: Range of n-gram sizes to use
        """
        self.dimension = dimension
        self.ngram_range = ngram_range

    def embed(self, text: str) -> List[float]:
        """
        Create embedding from text using hash-based approach.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        text = text.lower().strip()

        # Create accumulator
        embedding = np.zeros(self.dimension)

        # Process character n-grams
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # Hash the ngram
                h = hashlib.md5(ngram.encode()).hexdigest()
                # Use hash to determine position and value
                pos = int(h[:8], 16) % self.dimension
                val = (int(h[8:16], 16) / (16**8)) * 2 - 1  # Range [-1, 1]
                embedding[pos] += val

        # Process word tokens for word-level semantics
        words = text.split()
        for word in words:
            h = hashlib.sha256(word.encode()).hexdigest()
            # Multiple positions per word for richer representation
            for i in range(3):
                pos = int(h[i*8:(i+1)*8], 16) % self.dimension
                val = (int(h[(i+3)*8:(i+4)*8], 16) / (16**8)) * 2 - 1
                embedding[pos] += val * 2  # Words matter more

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [self.embed(text) for text in texts]
