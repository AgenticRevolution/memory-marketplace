"""
Data models for Memory Nexus Lite

Clean, simple Pydantic models - no over-abstraction.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class Memory:
    """A single memory unit stored in the system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    context: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0.0 to 1.0
    embedding: Optional[List[float]] = None

    # Lifecycle tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Decay and reinforcement
    decay_score: float = 1.0  # Starts at 1.0, decays over time
    reinforcement_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "context": self.context,
            "metadata": self.metadata,
            "importance": self.importance,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "decay_score": self.decay_score,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("accessed_at"), str):
            data["accessed_at"] = datetime.fromisoformat(data["accessed_at"])
        return cls(**data)

    def touch(self) -> None:
        """Mark as accessed - reinforces the memory."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
        # Reinforce: accessing a memory strengthens it
        self.decay_score = min(1.0, self.decay_score + 0.1)
        self.reinforcement_count += 1


@dataclass
class MemoryQuery:
    """A query against the memory store."""

    query: str
    context: Optional[str] = None
    limit: int = 10
    threshold: float = 0.7  # Minimum similarity for semantic search
    include_related: bool = False
    related_depth: int = 1


@dataclass
class MemoryResult:
    """Result from a memory query."""

    memory: Memory
    score: float  # Relevance/similarity score
    source: str = "search"  # "search", "cache", "prefetch"
    related: List["MemoryResult"] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship between two memories."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship_type: str = "RELATED_TO"
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "metadata": self.metadata,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class QueryPattern:
    """A learned query pattern for self-nourishing."""

    pattern_hash: str = ""
    query_template: str = ""
    context: Optional[str] = None
    frequency: int = 0
    avg_result_count: float = 0
    success_rate: float = 0
    follow_up_patterns: List[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_hash": self.pattern_hash,
            "query_template": self.query_template,
            "context": self.context,
            "frequency": self.frequency,
            "avg_result_count": self.avg_result_count,
            "success_rate": self.success_rate,
            "follow_up_patterns": self.follow_up_patterns,
            "last_seen": self.last_seen.isoformat(),
        }
