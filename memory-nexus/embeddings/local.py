"""
Local Embeddings using sentence-transformers.

Zero API costs, works offline, good for medical privacy.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """
    Local embedding generation using sentence-transformers.

    Recommended models:
    - all-MiniLM-L6-v2: Fast, good quality (384 dims) - DEFAULT
    - all-mpnet-base-v2: Highest quality (768 dims)
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual (good for TCM texts)
    """

    # Model dimensions for common models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-mpnet-base-dot-v1": 768,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding model.

        Args:
            model_name: HuggingFace model name or path
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Local embeddings require: pip install sentence-transformers\n"
                "This will also install torch."
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.MODEL_DIMENSIONS.get(
            model_name,
            self.model.get_sentence_embedding_dimension()
        )

        logger.info(f"Loaded local embedding model: {model_name} (dim={self._dimension})")

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched for efficiency)."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
