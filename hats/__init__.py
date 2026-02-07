"""
Memory Nexus Hats - Specialist Memory Packs

Server-side pre-seeded memory stores that give any LLM instant domain expertise.
Same brain + different memories = different expert.

Hats are hosted-only — data never leaves our servers.
Requires active Memory API subscription.
"""

from .manifest import HatManifest
from .manager import HatManager

__all__ = ["HatManifest", "HatManager"]
