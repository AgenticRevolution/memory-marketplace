"""
Semantic Storage - Vector Similarity Search

Default: FAISS (local, zero external services)
Optional: Pinecone (cloud scale)

This is the "semantic memory" tier - finds similar content by meaning.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SemanticStore(ABC):
    """Abstract base for semantic/vector storage backends."""

    @abstractmethod
    def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Add a vector with metadata."""
        pass

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors. Returns list of (id, score, metadata)."""
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get vector and metadata by ID."""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a vector."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of vectors stored."""
        pass


class FAISSSemanticStore(SemanticStore):
    """
    FAISS-based semantic storage with SQLite metadata.

    Uses Facebook's FAISS for efficient similarity search.
    Stores metadata in SQLite for filtering.

    Requires: pip install faiss-cpu (or faiss-gpu)
    """

    def __init__(
        self,
        dimension: int = 384,  # Default for all-MiniLM-L6-v2
        db_path: str = "~/.memory-nexus-lite/semantic.db",
        index_path: str = "~/.memory-nexus-lite/faiss.index"
    ):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS support requires: pip install faiss-cpu\n"
                "For GPU: pip install faiss-gpu"
            )

        self.dimension = dimension
        self.db_path = Path(db_path).expanduser()
        self.index_path = Path(index_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()
        self._load_or_create_index()

        logger.info(f"FAISS semantic store initialized (dim={dimension})")

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize metadata database."""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    faiss_idx INTEGER NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_vectors_faiss
                ON vectors(faiss_idx);
            """)

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        if self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = self.faiss.IndexFlatIP(self.dimension)
            logger.info("Created new FAISS index")

        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._rebuild_mappings()

    def _rebuild_mappings(self) -> None:
        """Rebuild ID mappings from database."""
        cursor = self._conn.execute("SELECT id, faiss_idx FROM vectors")
        for row in cursor.fetchall():
            self._id_to_idx[row["id"]] = row["faiss_idx"]
            self._idx_to_id[row["faiss_idx"]] = row["id"]

    def _normalize(self, embedding: List[float]) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        """Add a vector with metadata."""
        if id in self._id_to_idx:
            # Update existing - delete first
            self.delete(id)

        vec = self._normalize(embedding)
        faiss_idx = self.index.ntotal

        self.index.add(vec)

        with self._conn:
            self._conn.execute(
                "INSERT INTO vectors (id, faiss_idx, metadata) VALUES (?, ?, ?)",
                (id, faiss_idx, json.dumps(metadata or {}))
            )

        self._id_to_idx[id] = faiss_idx
        self._idx_to_id[faiss_idx] = id

        return True

    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if self.index.ntotal == 0:
            return []

        vec = self._normalize(embedding)

        # Search more than needed if filtering
        search_limit = limit * 3 if filter else limit

        scores, indices = self.index.search(vec, min(search_limit, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue

            id = self._idx_to_id.get(int(idx))
            if not id:
                continue

            # Get metadata
            cursor = self._conn.execute(
                "SELECT metadata FROM vectors WHERE id = ?", (id,)
            )
            row = cursor.fetchone()
            if not row:
                continue

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            # Apply filter
            if filter:
                match = all(
                    metadata.get(k) == v
                    for k, v in filter.items()
                )
                if not match:
                    continue

            results.append((id, float(score), metadata))

            if len(results) >= limit:
                break

        return results

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get vector and metadata by ID."""
        if id not in self._id_to_idx:
            return None

        faiss_idx = self._id_to_idx[id]

        # Reconstruct vector from index
        vec = np.zeros((1, self.dimension), dtype=np.float32)
        self.index.reconstruct(faiss_idx, vec[0])

        cursor = self._conn.execute(
            "SELECT metadata FROM vectors WHERE id = ?", (id,)
        )
        row = cursor.fetchone()
        metadata = json.loads(row["metadata"]) if row and row["metadata"] else {}

        return vec[0].tolist(), metadata

    def delete(self, id: str) -> bool:
        """Delete a vector (marks as deleted, doesn't remove from index)."""
        if id not in self._id_to_idx:
            return False

        with self._conn:
            self._conn.execute("DELETE FROM vectors WHERE id = ?", (id,))

        faiss_idx = self._id_to_idx.pop(id)
        self._idx_to_id.pop(faiss_idx, None)

        return True

    def clear(self) -> None:
        """Clear all vectors."""
        with self._conn:
            self._conn.execute("DELETE FROM vectors")

        self.index = self.faiss.IndexFlatIP(self.dimension)
        self._id_to_idx.clear()
        self._idx_to_id.clear()

    @property
    def count(self) -> int:
        """Number of active vectors."""
        cursor = self._conn.execute("SELECT COUNT(*) as count FROM vectors")
        return cursor.fetchone()["count"]

    def save(self) -> None:
        """Persist index to disk."""
        self.faiss.write_index(self.index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")

    def close(self) -> None:
        """Save and close."""
        self.save()
        if hasattr(self._local, "conn"):
            self._local.conn.close()


class SimpleSemanticStore(SemanticStore):
    """
    Simple SQLite-only semantic store using numpy.

    Zero external dependencies - just numpy (which you already have).
    Slower than FAISS but works everywhere.
    Good for small datasets (<10k vectors).
    """

    def __init__(
        self,
        dimension: int = 384,
        db_path: str = "~/.memory-nexus-lite/semantic_simple.db"
    ):
        self.dimension = dimension
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

        logger.info(f"Simple semantic store initialized (dim={dimension})")

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def _normalize(self, embedding: List[float]) -> np.ndarray:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> bool:
        vec = self._normalize(embedding)
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO vectors (id, embedding, metadata)
                   VALUES (?, ?, ?)""",
                (id, vec.tobytes(), json.dumps(metadata or {}))
            )
        return True

    def search(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        query_vec = self._normalize(embedding)

        cursor = self._conn.execute("SELECT id, embedding, metadata FROM vectors")
        results = []

        for row in cursor.fetchall():
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(np.dot(query_vec, vec))  # Cosine similarity

            if score < threshold:
                continue

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            if filter:
                match = all(metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue

            results.append((row["id"], score, metadata))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        cursor = self._conn.execute(
            "SELECT embedding, metadata FROM vectors WHERE id = ?", (id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return vec.tolist(), metadata

    def delete(self, id: str) -> bool:
        with self._conn:
            cursor = self._conn.execute("DELETE FROM vectors WHERE id = ?", (id,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM vectors")

    @property
    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) as count FROM vectors")
        return cursor.fetchone()["count"]

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()


def create_semantic_store(
    backend: str = "simple",
    dimension: int = 384,
    **kwargs
) -> SemanticStore:
    """
    Factory function to create semantic store.

    Args:
        backend: "simple", "faiss", or "pinecone://..."
        dimension: Vector dimension (default 384 for MiniLM)
        **kwargs: Additional arguments for the backend

    Returns:
        SemanticStore instance
    """
    if backend == "simple":
        return SimpleSemanticStore(dimension=dimension, **kwargs)

    elif backend == "faiss":
        return FAISSSemanticStore(dimension=dimension, **kwargs)

    elif backend.startswith("pinecone://"):
        raise NotImplementedError(
            "Pinecone backend coming soon. Use 'simple' or 'faiss' for now."
        )

    else:
        raise ValueError(f"Unknown semantic backend: {backend}")
