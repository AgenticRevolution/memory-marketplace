"""
Pattern Type Definitions.

These structures represent patterns, their evidence, and relationships.
Designed for flexibility and integration with Memory Nexus.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PatternType(Enum):
    """
    Categories of patterns that can be extracted.

    Based on cognitive science and the original Distillation Engine taxonomy.
    """
    # Content patterns
    LINGUISTIC = "linguistic"      # Word choice, phrasing, terminology
    STRUCTURAL = "structural"      # Organization, hierarchy, format
    SEMANTIC = "semantic"          # Meaning, concepts, topics
    EMOTIONAL = "emotional"        # Tone, sentiment, affect

    # Behavioral patterns
    TEMPORAL = "temporal"          # Time-based patterns, sequences
    CAUSAL = "causal"              # Cause-effect relationships
    BEHAVIORAL = "behavioral"      # User behavior, access patterns

    # Meta patterns
    QUERY = "query"                # Search/query patterns
    RETRIEVAL = "retrieval"        # What gets retrieved together
    RELATIONSHIP = "relationship"  # How knowledge connects
    QUALITY = "quality"            # What makes content valuable


class ConfidenceLevel(Enum):
    """How confident we are in a pattern."""
    SPECULATIVE = "speculative"    # Weak signal, needs validation
    LOW = "low"                    # Some evidence, emerging
    MEDIUM = "medium"              # Moderate evidence, stable
    HIGH = "high"                  # Strong evidence, validated


class RelationshipType(Enum):
    """How patterns relate to each other."""
    SUPPORTS = "supports"          # Evidence reinforces
    CONTRADICTS = "contradicts"    # Evidence conflicts
    EXTENDS = "extends"            # Builds upon
    SPECIALIZES = "specializes"    # More specific version
    GENERALIZES = "generalizes"    # More abstract version
    CORRELATES = "correlates"      # Co-occurs
    PRECEDES = "precedes"          # Comes before
    FOLLOWS = "follows"            # Comes after
    CAUSES = "causes"              # Causal relationship
    SIMILAR = "similar"            # High similarity


@dataclass
class PatternEvidence:
    """
    Evidence supporting a pattern from a specific source.

    Patterns are supported by evidence from memories, queries, or usage.
    """
    source_id: str                           # Memory ID or event ID
    source_type: str                         # "memory", "query", "usage"
    confidence: float                        # 0.0 to 1.0
    relevance: float                         # How relevant to the pattern
    detected_at: datetime = field(default_factory=datetime.now)
    detection_method: str = "statistical"    # How it was detected
    snippets: List[str] = field(default_factory=list)  # Supporting text
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "detected_at": self.detected_at.isoformat(),
            "detection_method": self.detection_method,
            "snippets": self.snippets,
            "metadata": self.metadata,
        }


@dataclass
class PatternRelationship:
    """Relationship between two patterns."""
    source_pattern_id: str
    target_pattern_id: str
    relationship_type: RelationshipType
    strength: float                          # 0.0 to 1.0
    description: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pattern_id": self.source_pattern_id,
            "target_pattern_id": self.target_pattern_id,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Pattern:
    """
    A pattern extracted from knowledge or usage.

    Patterns are the core learning unit - they capture recurring structures,
    behaviors, and relationships that can improve retrieval and understanding.
    """
    id: str
    name: str
    description: str
    pattern_type: PatternType

    # Evidence and confidence
    evidence: List[PatternEvidence] = field(default_factory=list)
    confidence: float = 0.5                  # Aggregate confidence
    stability: float = 0.5                   # How stable over time

    # Relationships
    relationships: List[PatternRelationship] = field(default_factory=list)

    # Classification
    tags: List[str] = field(default_factory=list)
    context: Optional[str] = None            # Associated context

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    application_count: int = 0               # How often applied
    success_rate: float = 0.0                # When applied, how successful

    # Synthesis
    is_synthesized: bool = False             # Created from other patterns
    synthesis_method: Optional[str] = None
    source_pattern_ids: List[str] = field(default_factory=list)

    # Embeddings
    embedding_fingerprint: Optional[List[float]] = None  # 64-dim fingerprint
    keywords: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, evidence: PatternEvidence):
        """Add evidence and recalculate confidence."""
        self.evidence.append(evidence)
        self._recalculate_confidence()
        self.updated_at = datetime.now()

    def _recalculate_confidence(self):
        """Recalculate aggregate confidence from evidence."""
        if not self.evidence:
            self.confidence = 0.5
            return

        # Weighted average by relevance
        total_weight = sum(e.relevance for e in self.evidence)
        if total_weight == 0:
            self.confidence = 0.5
            return

        weighted_sum = sum(e.confidence * e.relevance for e in self.evidence)
        self.confidence = weighted_sum / total_weight

        # Stability increases with more evidence
        self.stability = min(1.0, 0.3 + (len(self.evidence) * 0.1))

    def record_application(self, success: bool):
        """Record that this pattern was applied."""
        self.application_count += 1
        # Running average of success rate
        self.success_rate = (
            (self.success_rate * (self.application_count - 1) + (1.0 if success else 0.0))
            / self.application_count
        )
        self.updated_at = datetime.now()

    def get_confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level."""
        if self.confidence >= 0.8 and self.stability >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.SPECULATIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "stability": self.stability,
            "relationships": [r.to_dict() for r in self.relationships],
            "tags": self.tags,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "application_count": self.application_count,
            "success_rate": self.success_rate,
            "is_synthesized": self.is_synthesized,
            "synthesis_method": self.synthesis_method,
            "source_pattern_ids": self.source_pattern_ids,
            "embedding_fingerprint": self.embedding_fingerprint,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }


@dataclass
class UsageEvent:
    """
    A usage event to learn from.

    Tracks queries, retrievals, and user actions to learn behavioral patterns.
    """
    id: str
    event_type: str                          # "query", "retrieval", "click", "feedback"
    timestamp: datetime = field(default_factory=datetime.now)

    # Query info
    query: Optional[str] = None
    query_context: Optional[str] = None

    # Result info
    result_ids: List[str] = field(default_factory=list)
    result_scores: List[float] = field(default_factory=list)

    # User action
    selected_id: Optional[str] = None        # Which result was used
    feedback_score: Optional[float] = None   # User rating if provided

    # Session tracking
    session_id: Optional[str] = None
    previous_event_id: Optional[str] = None  # For sequence learning

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "query_context": self.query_context,
            "result_ids": self.result_ids,
            "result_scores": self.result_scores,
            "selected_id": self.selected_id,
            "feedback_score": self.feedback_score,
            "session_id": self.session_id,
            "previous_event_id": self.previous_event_id,
            "metadata": self.metadata,
        }


@dataclass
class PatternMatch:
    """
    A match between content and a pattern.

    Used when applying patterns to new content.
    """
    pattern: Pattern
    match_score: float                       # 0.0 to 1.0
    matched_elements: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
