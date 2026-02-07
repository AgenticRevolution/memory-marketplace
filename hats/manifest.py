"""
Hat Manifest - Schema and validation for hat metadata.

Each hat has a manifest describing its contents, quality metrics,
pricing tier, and sample queries. Stored server-side in the hat registry.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


HAT_FORMAT_VERSION = "1.0"


@dataclass
class HatStats:
    """Statistics about a hat's contents."""
    memories: int = 0
    relationships: int = 0
    patterns: int = 0
    contexts: List[str] = field(default_factory=list)
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HatStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HatManifest:
    """
    Manifest for a specialist memory hat.

    Stored server-side in the hat registry. Describes what the hat contains,
    its quality, pricing, and sample queries for discovery.
    """
    id: str
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    domain: str = ""
    tags: List[str] = field(default_factory=list)

    # Embedding config (must match to be compatible)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Contents
    stats: HatStats = field(default_factory=HatStats)

    # Quality
    quality_score: int = 0  # 0-100, computed by validator

    # Discovery — sample queries bots can try before activating
    sample_queries: List[str] = field(default_factory=list)

    # Metadata
    hat_format_version: str = HAT_FORMAT_VERSION
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    requires: Dict[str, str] = field(default_factory=lambda: {"memory_nexus_version": ">=0.2.0"})

    # Commercial
    tier: str = "standard"  # "standard" ($14.99/mo) or "premium" ($29-49/mo)
    price_cents: Optional[int] = None
    license: str = "proprietary"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Write manifest to hat.json file."""
        path = Path(path)
        if path.is_dir():
            path = path / "hat.json"
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HatManifest":
        """Create from dictionary."""
        data = data.copy()
        if "stats" in data and isinstance(data["stats"], dict):
            data["stats"] = HatStats.from_dict(data["stats"])
        # Filter to known fields
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "HatManifest":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> "HatManifest":
        """Load manifest from hat.json file."""
        path = Path(path)
        if path.is_dir():
            path = path / "hat.json"
        if not path.exists():
            raise FileNotFoundError(f"No hat.json found at {path}")
        return cls.from_json(path.read_text())

    def validate(self) -> List[str]:
        """
        Validate manifest fields. Returns list of errors (empty = valid).
        """
        errors = []
        if not self.id:
            errors.append("id is required")
        if not self.name:
            errors.append("name is required")
        if not self.version:
            errors.append("version is required")
        if self.hat_format_version != HAT_FORMAT_VERSION:
            errors.append(f"Unsupported hat format version: {self.hat_format_version} (expected {HAT_FORMAT_VERSION})")
        if self.embedding_dimension not in (384, 768):
            errors.append(f"Unsupported embedding dimension: {self.embedding_dimension} (expected 384 or 768)")
        if self.quality_score < 0 or self.quality_score > 100:
            errors.append(f"quality_score must be 0-100, got {self.quality_score}")
        return errors


def generate_manifest_from_store(
    hat_id: str,
    hat_name: str,
    data_dir: Path,
    store,
    **kwargs
) -> HatManifest:
    """
    Generate a HatManifest by inspecting a MemoryStore's contents.

    Args:
        hat_id: Unique identifier for the hat
        hat_name: Display name
        data_dir: Path to the hat's data directory
        store: An initialized MemoryStore instance
        **kwargs: Additional manifest fields (author, description, domain, tags, etc.)
    """
    # Get stats from store
    store_stats = store.get_stats()
    contexts = store.list_contexts() if hasattr(store, 'list_contexts') else {}

    # Calculate directory size
    data_path = Path(data_dir)
    size_bytes = sum(f.stat().st_size for f in data_path.rglob("*") if f.is_file())

    stats = HatStats(
        memories=store_stats.get("memories", store_stats.get("total_memories", 0)),
        relationships=store_stats.get("relationships", store_stats.get("total_relationships", 0)),
        patterns=store_stats.get("patterns_learned", store_stats.get("patterns", 0)),
        contexts=list(contexts.keys()) if isinstance(contexts, dict) else [],
        size_bytes=size_bytes,
    )

    manifest = HatManifest(
        id=hat_id,
        name=hat_name,
        stats=stats,
        **{k: v for k, v in kwargs.items() if k in HatManifest.__dataclass_fields__}
    )

    return manifest
