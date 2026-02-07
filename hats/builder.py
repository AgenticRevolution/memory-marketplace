"""
Hat Builder - Create specialist memory hats from seed scripts.

The builder takes a seed script (Python file that defines domain knowledge),
runs it against a fresh MemoryStore, and outputs a complete hat data directory
ready for server-side deployment.
"""

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Add memory-nexus to path (bundled in sibling directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "memory-nexus"))


class HatBuilder:
    """
    Builds hat data directories from seed scripts or programmatic knowledge.

    A seed script is a Python file that defines a `seed(store)` function.
    The builder creates a fresh MemoryStore, calls seed(store), then
    generates the manifest from the resulting data.

    Usage:
        builder = HatBuilder(
            hat_id="devops-expert",
            hat_name="DevOps Deployment Expert",
            output_dir="./build/devops-expert",
        )

        # From a seed script
        builder.build_from_seed("hats/seeds/devops/seed.py")

        # Or programmatically
        builder.build(knowledge=[
            {"content": "...", "context": "docker", "importance": 0.9},
            ...
        ])
    """

    def __init__(
        self,
        hat_id: str,
        hat_name: str,
        output_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        author: str = "Mobius",
        description: str = "",
        domain: str = "",
        tags: Optional[List[str]] = None,
        tier: str = "standard",
        price_cents: Optional[int] = None,
        sample_queries: Optional[List[str]] = None,
    ):
        self.hat_id = hat_id
        self.hat_name = hat_name
        self.output_dir = Path(output_dir)
        self.embedding_model = embedding_model
        self.author = author
        self.description = description
        self.domain = domain
        self.tags = tags or []
        self.tier = tier
        self.price_cents = price_cents
        self.sample_queries = sample_queries or []
        self._store = None

    def _create_store(self):
        """Create a fresh MemoryStore at the output directory."""
        from core.store import MemoryStore

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._store = MemoryStore(
            data_dir=str(self.output_dir),
            embedding_model=self.embedding_model,
            enable_cache=False,
            enable_patterns=True,
        )
        return self._store

    def build_from_seed(self, seed_script: str) -> Dict[str, Any]:
        """
        Build a hat from a seed script.

        The seed script must define a `seed(store)` function that adds
        memories and relationships to the provided MemoryStore.

        Args:
            seed_script: Path to the seed Python file

        Returns:
            Build results dict with stats
        """
        seed_path = Path(seed_script)
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed script not found: {seed_path}")

        store = self._create_store()

        # Load and run the seed script
        spec = importlib.util.spec_from_file_location("hat_seed", str(seed_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "seed"):
            raise ValueError(f"Seed script must define a seed(store) function: {seed_path}")

        logger.info(f"Running seed script: {seed_path}")
        module.seed(store)

        return self._finalize(store)

    def build(
        self,
        knowledge: List[Dict[str, Any]],
        relationships: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Build a hat from structured knowledge data.

        Args:
            knowledge: List of memory dicts with keys:
                - content (required): The knowledge text
                - context (required): Category/domain
                - importance (optional): 0.0-1.0, default 0.7
                - metadata (optional): Additional data
            relationships: List of relationship dicts with keys:
                - source_id: Source memory ID (or index into knowledge list)
                - target_id: Target memory ID (or index)
                - type: Relationship type (RELATED_TO, TREATS, LEADS_TO, etc.)

        Returns:
            Build results dict with stats
        """
        store = self._create_store()

        # Add memories
        memory_ids = []
        for item in knowledge:
            mem = store.add(
                content=item["content"],
                context=item.get("context", "general"),
                importance=item.get("importance", 0.7),
                metadata=item.get("metadata", {}),
            )
            memory_ids.append(mem.id)
            logger.debug(f"Added memory: {item['content'][:60]}...")

        # Add relationships
        if relationships:
            for rel in relationships:
                source = rel["source_id"]
                target = rel["target_id"]
                # Allow referencing by index
                if isinstance(source, int):
                    source = memory_ids[source]
                if isinstance(target, int):
                    target = memory_ids[target]
                store.relate(source, target, rel.get("type", "RELATED_TO"))

        return self._finalize(store)

    def build_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Build a hat from a JSON knowledge file.

        Expected format:
        {
            "knowledge": [
                {"content": "...", "context": "...", "importance": 0.9},
                ...
            ],
            "relationships": [
                {"source_id": 0, "target_id": 1, "type": "RELATED_TO"},
                ...
            ]
        }
        """
        data = json.loads(Path(json_path).read_text())
        return self.build(
            knowledge=data["knowledge"],
            relationships=data.get("relationships"),
        )

    def _finalize(self, store) -> Dict[str, Any]:
        """Generate manifest, run pattern extraction, return build stats."""
        # Run pattern analysis if available
        if hasattr(store, "analyze_patterns"):
            try:
                store.analyze_patterns()
                logger.info("Pattern analysis complete")
            except Exception as e:
                logger.warning(f"Pattern analysis failed (non-fatal): {e}")

        # Generate stats
        stats = store.get_stats()

        # Generate manifest
        from .manifest import HatManifest, HatStats, generate_manifest_from_store

        manifest = generate_manifest_from_store(
            hat_id=self.hat_id,
            hat_name=self.hat_name,
            data_dir=self.output_dir,
            store=store,
            author=self.author,
            description=self.description,
            domain=self.domain,
            tags=self.tags,
            tier=self.tier,
            price_cents=self.price_cents,
            sample_queries=self.sample_queries,
        )
        manifest.save(self.output_dir)

        store.close()

        result = {
            "hat_id": self.hat_id,
            "output_dir": str(self.output_dir),
            "memories": stats.get("memories", 0),
            "relationships": stats.get("relationships", 0),
            "patterns": stats.get("patterns_learned", 0),
            "manifest_path": str(self.output_dir / "hat.json"),
        }

        logger.info(
            f"Built hat '{self.hat_name}': "
            f"{result['memories']} memories, "
            f"{result['relationships']} relationships, "
            f"{result['patterns']} patterns"
        )
        return result
