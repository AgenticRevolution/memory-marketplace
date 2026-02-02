"""
Pattern Intelligence System for Memory Nexus Lite.

Extracts, learns, and applies patterns from your knowledge base.
Works locally without any LLM dependencies, with optional LLM enhancement.

Core Concepts:
- Patterns are recurring structures in your knowledge
- Evidence supports patterns from multiple sources
- Patterns evolve: they're validated, strengthened, or retired
- Usage patterns improve retrieval over time

The system learns:
1. What content patterns exist in your knowledge
2. What query patterns users follow
3. What makes knowledge valuable (access patterns)
4. How knowledge relates (structural patterns)
"""

from .types import (
    Pattern,
    PatternType,
    PatternEvidence,
    PatternRelationship,
    UsageEvent,
    PatternMatch,
)
from .extractor import PatternExtractor
from .analyzer import PatternAnalyzer
from .learner import UsageLearner
from .integration import PatternIntelligence

__all__ = [
    # Types
    "Pattern",
    "PatternType",
    "PatternEvidence",
    "PatternRelationship",
    "UsageEvent",
    "PatternMatch",
    # Core classes
    "PatternExtractor",
    "PatternAnalyzer",
    "UsageLearner",
    "PatternIntelligence",
]
