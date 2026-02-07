"""
Hat Manager - Server-side hat registry and activation.

Hats are server-side only. Data never leaves our infrastructure.
This manages the hat catalog and handles activation/deactivation
into tenant memory stores via context prefixing.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add memory-nexus to path (bundled in sibling directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "memory-nexus"))

from .manifest import HatManifest

logger = logging.getLogger(__name__)


REGISTRY_FILENAME = "_registry.json"


class HatManager:
    """
    Server-side hat registry and activation manager.

    Hat data lives in a base directory (e.g., /data/hats/) on the server.
    Each hat is a subdirectory containing pre-seeded Memory Nexus data files.
    The registry (_registry.json) tracks the catalog of available hats.
    """

    def __init__(self, hats_dir: Optional[str] = None):
        """
        Args:
            hats_dir: Directory containing hat data subdirectories.
                      Defaults to /data/hats/ (server) or ~/.memory-nexus/hats/ (local dev)
        """
        if hats_dir is None:
            hats_dir = os.environ.get(
                "HATS_DATA_DIR",
                os.path.expanduser("~/.memory-nexus/hats"),
            )
        self.hats_dir = Path(hats_dir)
        self.hats_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, HatManifest] = {}
        self._load_registry()

    def _registry_path(self) -> Path:
        return self.hats_dir / REGISTRY_FILENAME

    def _load_registry(self) -> None:
        """Load hat catalog from registry file and/or scan directories."""
        reg_path = self._registry_path()
        if reg_path.exists():
            try:
                data = json.loads(reg_path.read_text())
                for hat_data in data.get("hats", []):
                    manifest = HatManifest.from_dict(hat_data)
                    self._registry[manifest.id] = manifest
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        # Also scan for hat.json files in subdirectories
        for item in sorted(self.hats_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("_"):
                manifest_path = item / "hat.json"
                if manifest_path.exists() and item.name not in self._registry:
                    try:
                        manifest = HatManifest.load(manifest_path)
                        self._registry[manifest.id] = manifest
                    except Exception as e:
                        logger.warning(f"Failed to load hat {item.name}: {e}")

    def _save_registry(self) -> None:
        """Persist the registry to disk."""
        data = {
            "hats": [m.to_dict() for m in self._registry.values()]
        }
        self._registry_path().write_text(json.dumps(data, indent=2))

    def register_hat(self, manifest: HatManifest) -> None:
        """Add or update a hat in the registry."""
        self._registry[manifest.id] = manifest
        self._save_registry()

    def list_hats(self) -> List[HatManifest]:
        """List all available hats."""
        return list(self._registry.values())

    def get_hat(self, hat_id: str) -> Optional[HatManifest]:
        """Get manifest for a specific hat."""
        return self._registry.get(hat_id)

    def get_hat_data_dir(self, hat_id: str) -> Optional[Path]:
        """Get the data directory for a hat. Returns None if not found."""
        hat_dir = self.hats_dir / hat_id
        if hat_dir.exists() and hat_dir.is_dir():
            return hat_dir
        return None

    def hat_exists(self, hat_id: str) -> bool:
        """Check if a hat exists (has both registry entry and data dir)."""
        return hat_id in self._registry and self.get_hat_data_dir(hat_id) is not None

    def activate_hat(self, hat_id: str, tenant_store, prefix: Optional[str] = None) -> int:
        """
        Activate a hat by importing its memories into a tenant's store.

        Hat memories are imported with context prefixing so they coexist
        with the tenant's own memories and other activated hats.

        Args:
            hat_id: ID of the hat to activate
            tenant_store: The tenant's MemoryStore instance
            prefix: Context prefix (default: "hat:{hat_id}")

        Returns:
            Number of memories imported

        Raises:
            FileNotFoundError: If hat data directory doesn't exist
            ValueError: If hat is not in the registry
        """
        if hat_id not in self._registry:
            raise ValueError(f"Unknown hat: {hat_id}")

        hat_dir = self.get_hat_data_dir(hat_id)
        if hat_dir is None:
            raise FileNotFoundError(f"Hat data not found: {hat_id}")

        if prefix is None:
            prefix = f"hat:{hat_id}"

        # Create a read-only store pointing at the hat data
        from core.store import MemoryStore

        manifest = self._registry[hat_id]
        hat_store = MemoryStore(
            data_dir=str(hat_dir),
            embedding_model=manifest.embedding_model,
            enable_cache=False,
            enable_patterns=False,
        )

        imported = 0
        try:
            contexts = hat_store.list_contexts() if hasattr(hat_store, "list_contexts") else {}

            for context_name in contexts:
                results = hat_store.query("", context=context_name, limit=10000)
                for result in results:
                    mem = result.memory
                    prefixed_context = f"{prefix}:{mem.context}"
                    tenant_store.add(
                        content=mem.content,
                        context=prefixed_context,
                        importance=mem.importance,
                        metadata={
                            **mem.metadata,
                            "_hat_id": hat_id,
                            "_hat_version": manifest.version,
                            "_original_context": mem.context,
                        },
                    )
                    imported += 1
        finally:
            if hasattr(hat_store, "close"):
                hat_store.close()

        logger.info(f"Activated hat '{hat_id}': imported {imported} memories with prefix '{prefix}'")
        return imported

    def deactivate_hat(self, hat_id: str, tenant_store) -> int:
        """
        Deactivate a hat by removing its memories from a tenant's store.

        Finds and removes all memories with context prefix "hat:{hat_id}:".

        Args:
            hat_id: ID of the hat to deactivate
            tenant_store: The tenant's MemoryStore instance

        Returns:
            Number of memories removed
        """
        prefix = f"hat:{hat_id}:"
        removed = 0

        if hasattr(tenant_store, "list_contexts"):
            contexts = tenant_store.list_contexts()
            for context_name in contexts:
                if context_name.startswith(prefix):
                    # Get all memories in this hat context
                    results = tenant_store.query("", context=context_name, limit=10000)
                    for result in results:
                        if hasattr(tenant_store, "delete"):
                            tenant_store.delete(result.memory.id)
                            removed += 1

        logger.info(f"Deactivated hat '{hat_id}': removed {removed} memories")
        return removed

    def get_tenant_active_hats(self, tenant_store) -> List[str]:
        """
        Get list of hat IDs currently activated for a tenant.

        Scans the tenant's contexts for "hat:*:" prefixes.
        """
        active = set()
        if hasattr(tenant_store, "list_contexts"):
            contexts = tenant_store.list_contexts()
            for context_name in contexts:
                if context_name.startswith("hat:"):
                    # Extract hat_id from "hat:{hat_id}:{original_context}"
                    parts = context_name.split(":", 2)
                    if len(parts) >= 2:
                        active.add(parts[1])
        return sorted(active)
