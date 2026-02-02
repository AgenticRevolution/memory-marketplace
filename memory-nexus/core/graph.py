"""
Graph Storage - Relationship Memory

Default: SQLite (zero dependencies)
Optional: Neo4j (for complex graph queries)

This is the "relational memory" tier - tracks connections between memories.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphStore(ABC):
    """Abstract base for graph storage backends."""

    @abstractmethod
    def add_node(self, id: str, labels: List[str] = None, properties: Dict[str, Any] = None) -> bool:
        """Add a node with labels and properties."""
        pass

    @abstractmethod
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """Add a relationship between nodes."""
        pass

    @abstractmethod
    def get_node(self, id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        pass

    @abstractmethod
    def get_relationships(
        self,
        node_id: str,
        rel_type: str = None,
        direction: str = "both"  # "out", "in", "both"
    ) -> List[Dict[str, Any]]:
        """Get relationships for a node."""
        pass

    @abstractmethod
    def get_related_nodes(
        self,
        node_id: str,
        rel_type: str = None,
        depth: int = 1,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get nodes related to given node up to depth."""
        pass

    @abstractmethod
    def delete_node(self, id: str, cascade: bool = True) -> bool:
        """Delete a node and optionally its relationships."""
        pass

    @abstractmethod
    def delete_relationship(self, source_id: str, target_id: str, rel_type: str = None) -> bool:
        """Delete relationship(s) between nodes."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
        pass


class SQLiteGraphStore(GraphStore):
    """
    SQLite-based graph storage.

    Simple but effective - good for most use cases.
    Supports traversal up to configurable depth.
    """

    def __init__(self, db_path: str = "~/.memory-nexus-lite/graph.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

        logger.info(f"SQLite graph store initialized at {self.db_path}")

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
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    labels TEXT,
                    properties TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    rel_type TEXT NOT NULL,
                    properties TEXT,
                    weight REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    UNIQUE(source_id, target_id, rel_type)
                );

                CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
                CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
                CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(rel_type);
            """)

    def add_node(self, id: str, labels: List[str] = None, properties: Dict[str, Any] = None) -> bool:
        labels_str = json.dumps(labels or [])
        props_str = json.dumps(properties or {})

        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO nodes (id, labels, properties)
                   VALUES (?, ?, ?)""",
                (id, labels_str, props_str)
            )
        return True

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        # Ensure nodes exist
        self.add_node(source_id)
        self.add_node(target_id)

        props_str = json.dumps(properties or {})
        weight = (properties or {}).get("weight", 1.0)

        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO relationships
                   (source_id, target_id, rel_type, properties, weight)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, target_id, rel_type, props_str, weight)
            )
        return True

    def get_node(self, id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT id, labels, properties FROM nodes WHERE id = ?", (id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "id": row["id"],
            "labels": json.loads(row["labels"]) if row["labels"] else [],
            "properties": json.loads(row["properties"]) if row["properties"] else {}
        }

    def get_relationships(
        self,
        node_id: str,
        rel_type: str = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        results = []

        if direction in ("out", "both"):
            query = "SELECT * FROM relationships WHERE source_id = ?"
            params = [node_id]
            if rel_type:
                query += " AND rel_type = ?"
                params.append(rel_type)

            cursor = self._conn.execute(query, params)
            for row in cursor.fetchall():
                results.append({
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "rel_type": row["rel_type"],
                    "properties": json.loads(row["properties"]) if row["properties"] else {},
                    "weight": row["weight"],
                    "direction": "out"
                })

        if direction in ("in", "both"):
            query = "SELECT * FROM relationships WHERE target_id = ?"
            params = [node_id]
            if rel_type:
                query += " AND rel_type = ?"
                params.append(rel_type)

            cursor = self._conn.execute(query, params)
            for row in cursor.fetchall():
                results.append({
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "rel_type": row["rel_type"],
                    "properties": json.loads(row["properties"]) if row["properties"] else {},
                    "weight": row["weight"],
                    "direction": "in"
                })

        return results

    def get_related_nodes(
        self,
        node_id: str,
        rel_type: str = None,
        depth: int = 1,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """BFS traversal to find related nodes."""
        visited = {node_id}
        current_level = [node_id]
        results = []

        for d in range(depth):
            next_level = []

            for current_id in current_level:
                rels = self.get_relationships(current_id, rel_type, direction)

                for rel in rels:
                    other_id = rel["target_id"] if rel["source_id"] == current_id else rel["source_id"]

                    if other_id not in visited:
                        visited.add(other_id)
                        next_level.append(other_id)

                        node = self.get_node(other_id)
                        if node:
                            node["_depth"] = d + 1
                            node["_relationship"] = rel
                            results.append(node)

            current_level = next_level

        return results

    def delete_node(self, id: str, cascade: bool = True) -> bool:
        with self._conn:
            if cascade:
                self._conn.execute(
                    "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                    (id, id)
                )

            cursor = self._conn.execute("DELETE FROM nodes WHERE id = ?", (id,))
            return cursor.rowcount > 0

    def delete_relationship(self, source_id: str, target_id: str, rel_type: str = None) -> bool:
        with self._conn:
            if rel_type:
                cursor = self._conn.execute(
                    "DELETE FROM relationships WHERE source_id = ? AND target_id = ? AND rel_type = ?",
                    (source_id, target_id, rel_type)
                )
            else:
                cursor = self._conn.execute(
                    "DELETE FROM relationships WHERE source_id = ? AND target_id = ?",
                    (source_id, target_id)
                )
            return cursor.rowcount > 0

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM relationships")
            self._conn.execute("DELETE FROM nodes")

    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        node_count = self._conn.execute(
            "SELECT COUNT(*) as count FROM nodes"
        ).fetchone()["count"]

        rel_count = self._conn.execute(
            "SELECT COUNT(*) as count FROM relationships"
        ).fetchone()["count"]

        return {"nodes": node_count, "relationships": rel_count}

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()


def create_graph_store(backend: str = "sqlite", **kwargs) -> GraphStore:
    """
    Factory function to create graph store.

    Args:
        backend: "sqlite" or "neo4j://..."
        **kwargs: Additional arguments for the backend

    Returns:
        GraphStore instance
    """
    if backend == "sqlite" or backend.startswith("sqlite://"):
        db_path = kwargs.get("db_path", "~/.memory-nexus-lite/graph.db")
        if backend.startswith("sqlite://"):
            db_path = backend.replace("sqlite://", "")
        return SQLiteGraphStore(db_path)

    elif backend.startswith("neo4j://"):
        raise NotImplementedError(
            "Neo4j backend coming soon. Use SQLite for now."
        )

    else:
        raise ValueError(f"Unknown graph backend: {backend}")
