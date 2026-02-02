"""
Pattern Intelligence - Unified Integration Layer.

This is the main interface that ties together:
- Pattern extraction from content
- Multi-source analysis and validation
- Usage learning and behavioral patterns
- Integration with Memory Nexus

The PatternIntelligence class provides a single interface for
all pattern-related operations.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analyzer import PatternAnalyzer
from .extractor import ExtractionConfig, PatternExtractor
from .learner import UsageLearner
from .types import (
    Pattern,
    PatternEvidence,
    PatternMatch,
    PatternType,
    UsageEvent,
)

logger = logging.getLogger(__name__)


class PatternIntelligence:
    """
    Unified Pattern Intelligence System.

    Integrates pattern extraction, analysis, and learning into
    a single cohesive system that works with Memory Nexus.

    Features:
    - Extract patterns from your knowledge base
    - Learn from how the system is used
    - Cross-validate patterns across sources
    - Synthesize new patterns from existing ones
    - Persist patterns for long-term learning
    - Improve retrieval with pattern matching

    Example:
        # Initialize with a store
        from core.store import MemoryStore
        from patterns import PatternIntelligence

        store = MemoryStore()
        intelligence = PatternIntelligence(data_dir="./pattern_data")

        # Extract patterns from stored memories
        patterns = intelligence.analyze_knowledge_base(store)

        # Learn from a query
        results = store.query("fatigue treatments")
        intelligence.learn_from_query("fatigue treatments", results)

        # Get pattern-enhanced retrieval suggestions
        suggestions = intelligence.suggest_for_query("fatigue")

        # Record user selection
        intelligence.record_selection("fatigue treatments", selected_id="mem_123")
    """

    def __init__(
        self,
        data_dir: str = "./pattern_data",
        extraction_config: ExtractionConfig = None,
        auto_analyze_threshold: int = 10,
    ):
        """
        Initialize Pattern Intelligence.

        Args:
            data_dir: Directory for pattern persistence
            extraction_config: Configuration for pattern extraction
            auto_analyze_threshold: Auto-analyze after this many events
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.extractor = PatternExtractor(extraction_config)
        self.analyzer = PatternAnalyzer()
        self.learner = UsageLearner()

        # Pattern storage
        self.patterns: Dict[str, Pattern] = {}
        self._lock = threading.RLock()

        # Auto-analysis settings
        self.auto_analyze_threshold = auto_analyze_threshold
        self._events_since_analysis = 0

        # Initialize SQLite storage
        self._init_storage()

        # Load existing patterns
        self._load_patterns()

    def _init_storage(self):
        """Initialize SQLite storage for patterns."""
        db_path = self.data_dir / "patterns.db"
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                pattern_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                stability REAL DEFAULT 0.5,
                data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS usage_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                data JSON NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_type
            ON patterns(pattern_type)
        """)

        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON usage_events(event_type)
        """)

        self.db.commit()

    def _load_patterns(self):
        """Load patterns from storage."""
        cursor = self.db.execute("SELECT id, data FROM patterns")
        for row in cursor:
            try:
                data = json.loads(row[1])
                pattern = self._dict_to_pattern(data)
                self.patterns[pattern.id] = pattern
            except Exception as e:
                logger.warning(f"Failed to load pattern {row[0]}: {e}")

    def _save_pattern(self, pattern: Pattern):
        """Save pattern to storage."""
        self.db.execute("""
            INSERT OR REPLACE INTO patterns (id, name, description, pattern_type,
                confidence, stability, data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.id,
            pattern.name,
            pattern.description,
            pattern.pattern_type.value,
            pattern.confidence,
            pattern.stability,
            json.dumps(pattern.to_dict()),
            datetime.now().isoformat(),
        ))
        self.db.commit()

    def _dict_to_pattern(self, data: Dict) -> Pattern:
        """Convert dict back to Pattern object."""
        # Convert pattern_type string to enum
        data["pattern_type"] = PatternType(data["pattern_type"])

        # Convert evidence dicts to objects
        data["evidence"] = [
            PatternEvidence(
                source_id=e["source_id"],
                source_type=e["source_type"],
                confidence=e["confidence"],
                relevance=e["relevance"],
                detected_at=datetime.fromisoformat(e["detected_at"]),
                detection_method=e["detection_method"],
                snippets=e.get("snippets", []),
                metadata=e.get("metadata", {}),
            )
            for e in data.get("evidence", [])
        ]

        # Convert relationship dicts (simplified - store as metadata)
        data["relationships"] = []

        # Convert datetime strings
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return Pattern(**{
            k: v for k, v in data.items()
            if k in Pattern.__dataclass_fields__
        })

    # =========================================================================
    # Knowledge Base Analysis
    # =========================================================================

    def analyze_knowledge_base(
        self,
        store,  # MemoryStore
        context: str = None,
        force: bool = False,
    ) -> List[Pattern]:
        """
        Analyze knowledge base and extract patterns.

        Args:
            store: MemoryStore instance
            context: Optional context to filter by
            force: Force re-analysis even if patterns exist

        Returns:
            List of extracted patterns
        """
        with self._lock:
            # Get all memories
            results = store.query("", limit=10000, threshold=0.0)
            memories = [r.memory for r in results]

            if context:
                memories = [m for m in memories if m.context == context]

            if len(memories) < 2:
                logger.info("Not enough memories for pattern extraction")
                return []

            # Extract patterns
            logger.info(f"Analyzing {len(memories)} memories for patterns...")
            extracted = self.extractor.extract_from_memories(memories, context)

            # Cross-validate
            validated, invalidated = self.analyzer.cross_validate(extracted)
            logger.info(
                f"Extracted {len(extracted)} patterns, "
                f"{len(validated)} validated, {len(invalidated)} invalidated"
            )

            # Detect relationships
            relationships = self.analyzer.detect_relationships(validated)
            logger.info(f"Detected {len(relationships)} relationships")

            # Store patterns
            for pattern in validated:
                self.patterns[pattern.id] = pattern
                self._save_pattern(pattern)

            return validated

    def get_patterns(
        self,
        pattern_type: PatternType = None,
        min_confidence: float = 0.0,
        context: str = None,
        limit: int = 100,
    ) -> List[Pattern]:
        """
        Get stored patterns with optional filtering.

        Args:
            pattern_type: Filter by type
            min_confidence: Minimum confidence threshold
            context: Filter by context
            limit: Maximum patterns to return

        Returns:
            List of matching patterns
        """
        patterns = list(self.patterns.values())

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if min_confidence > 0:
            patterns = [p for p in patterns if p.confidence >= min_confidence]

        if context:
            patterns = [p for p in patterns if p.context == context]

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns[:limit]

    def find_gaps(self) -> List[Dict[str, Any]]:
        """
        Find gaps in pattern coverage.

        Returns:
            List of identified gaps with suggestions
        """
        return self.analyzer.find_gaps(list(self.patterns.values()))

    # =========================================================================
    # Usage Learning
    # =========================================================================

    def learn_from_query(
        self,
        query: str,
        results: List[Any],  # List of MemoryResult
        context: str = None,
        session_id: str = None,
    ):
        """
        Learn from a query and its results.

        Call this after every query to build behavioral patterns.

        Args:
            query: The query string
            results: List of MemoryResult objects
            context: Optional context
            session_id: Optional session ID for sequence learning
        """
        result_tuples = [
            (r.memory.id, r.score)
            for r in results[:20]  # Top 20
        ]

        self.learner.record_query(
            query=query,
            results=result_tuples,
            context=context,
            session_id=session_id,
        )

        self._events_since_analysis += 1
        self._maybe_auto_analyze()

    def record_selection(
        self,
        query: str,
        selected_id: str,
        session_id: str = None,
        feedback_score: float = None,
    ):
        """
        Record that a user selected a specific result.

        Args:
            query: The query that produced results
            selected_id: ID of the selected memory
            session_id: Optional session ID
            feedback_score: Optional explicit feedback (0-1)
        """
        self.learner.record_selection(
            query=query,
            selected_id=selected_id,
            session_id=session_id,
            feedback_score=feedback_score,
        )

        self._events_since_analysis += 1
        self._maybe_auto_analyze()

    def record_feedback(
        self,
        memory_id: str,
        feedback_score: float,
        query: str = None,
    ):
        """
        Record explicit user feedback on a memory.

        Args:
            memory_id: ID of the rated memory
            feedback_score: Score from 0 (bad) to 1 (good)
            query: Optional query context
        """
        self.learner.record_feedback(
            memory_id=memory_id,
            feedback_score=feedback_score,
            query=query,
        )

    def _maybe_auto_analyze(self):
        """Check if we should auto-analyze usage patterns."""
        if self._events_since_analysis >= self.auto_analyze_threshold:
            self._extract_usage_patterns()
            self._events_since_analysis = 0

    def _extract_usage_patterns(self):
        """Extract patterns from usage data."""
        usage_patterns = self.learner.extract_patterns()

        for pattern in usage_patterns:
            # Merge with existing if similar
            existing = self._find_similar_pattern(pattern)
            if existing:
                self._merge_patterns(existing, pattern)
            else:
                self.patterns[pattern.id] = pattern
                self._save_pattern(pattern)

    def _find_similar_pattern(self, pattern: Pattern) -> Optional[Pattern]:
        """Find an existing pattern similar to the given one."""
        for existing in self.patterns.values():
            if existing.pattern_type != pattern.pattern_type:
                continue
            comparison = self.analyzer.compare_patterns(existing, pattern)
            if comparison["similarity"] > 0.8:
                return existing
        return None

    def _merge_patterns(self, existing: Pattern, new: Pattern):
        """Merge new pattern evidence into existing pattern."""
        existing.evidence.extend(new.evidence)
        existing.keywords = list(set(existing.keywords + new.keywords))
        existing._recalculate_confidence()
        existing.updated_at = datetime.now()
        self._save_pattern(existing)

    # =========================================================================
    # Query Enhancement
    # =========================================================================

    def suggest_for_query(
        self,
        query: str,
        context: str = None,
    ) -> Dict[str, Any]:
        """
        Get pattern-based suggestions for a query.

        Returns predictions, preferred results, and related queries.

        Args:
            query: The query string
            context: Optional context

        Returns:
            Dict with suggestions
        """
        return {
            "next_queries": self.learner.predict_next(query),
            "preferred_results": self.learner.get_preferred_results(query),
            "related_queries": self.learner.get_related_queries(query),
            "matching_patterns": self.match_patterns_to_query(query, context),
        }

    def match_patterns_to_query(
        self,
        query: str,
        context: str = None,
        limit: int = 5,
    ) -> List[PatternMatch]:
        """
        Find patterns that match a query.

        Args:
            query: The query to match
            context: Optional context filter
            limit: Maximum matches to return

        Returns:
            List of PatternMatch objects
        """
        matches = []
        query_words = set(query.lower().split())

        for pattern in self.patterns.values():
            if context and pattern.context and pattern.context != context:
                continue

            # Calculate match score
            keyword_overlap = 0
            if pattern.keywords:
                keyword_set = set(kw.lower() for kw in pattern.keywords)
                overlap = len(query_words & keyword_set)
                keyword_overlap = overlap / len(query_words) if query_words else 0

            if keyword_overlap > 0.2:
                match = PatternMatch(
                    pattern=pattern,
                    match_score=keyword_overlap,
                    matched_elements=list(query_words & set(pattern.keywords)),
                )
                matches.append(match)

        # Sort by match score
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches[:limit]

    # =========================================================================
    # Pattern Synthesis
    # =========================================================================

    def synthesize_pattern(
        self,
        pattern_ids: List[str],
        name: str,
        description: str = None,
    ) -> Pattern:
        """
        Synthesize a new pattern from existing ones.

        Args:
            pattern_ids: IDs of patterns to synthesize from
            name: Name for the new pattern
            description: Optional description

        Returns:
            New synthesized pattern
        """
        patterns = [
            self.patterns[pid]
            for pid in pattern_ids
            if pid in self.patterns
        ]

        if len(patterns) < 2:
            raise ValueError("Need at least 2 valid patterns to synthesize")

        synthesized = self.analyzer.synthesize_patterns(
            patterns, name, description
        )

        self.patterns[synthesized.id] = synthesized
        self._save_pattern(synthesized)

        return synthesized

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get pattern intelligence statistics."""
        usage_stats = self.learner.get_usage_stats()

        pattern_by_type = {}
        for p in self.patterns.values():
            ptype = p.pattern_type.value
            pattern_by_type[ptype] = pattern_by_type.get(ptype, 0) + 1

        return {
            "total_patterns": len(self.patterns),
            "patterns_by_type": pattern_by_type,
            "avg_confidence": (
                sum(p.confidence for p in self.patterns.values()) /
                len(self.patterns) if self.patterns else 0
            ),
            **usage_stats,
        }

    def export_patterns(
        self,
        output_path: str,
        min_confidence: float = 0.5,
    ) -> int:
        """
        Export patterns to JSON file.

        Args:
            output_path: Path for output file
            min_confidence: Minimum confidence to export

        Returns:
            Number of patterns exported
        """
        patterns = [
            p.to_dict()
            for p in self.patterns.values()
            if p.confidence >= min_confidence
        ]

        with open(output_path, 'w') as f:
            json.dump({
                "patterns": patterns,
                "exported_at": datetime.now().isoformat(),
                "count": len(patterns),
            }, f, indent=2)

        return len(patterns)

    def import_patterns(self, input_path: str) -> int:
        """
        Import patterns from JSON file.

        Args:
            input_path: Path to input file

        Returns:
            Number of patterns imported
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        imported = 0
        for pdata in data.get("patterns", []):
            try:
                pattern = self._dict_to_pattern(pdata)
                if pattern.id not in self.patterns:
                    self.patterns[pattern.id] = pattern
                    self._save_pattern(pattern)
                    imported += 1
            except Exception as e:
                logger.warning(f"Failed to import pattern: {e}")

        return imported

    def close(self):
        """Close database connection."""
        self.db.close()
