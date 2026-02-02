"""
Usage Learner - Learn from System Usage.

Tracks and learns from how the system is used:
- Query patterns (what do users search for?)
- Retrieval patterns (what gets retrieved together?)
- Selection patterns (what do users choose?)
- Session patterns (what sequences do users follow?)

This creates behavioral patterns that improve retrieval over time.
"""

import hashlib
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .types import (
    Pattern,
    PatternEvidence,
    PatternType,
    UsageEvent,
)


@dataclass
class QuerySequence:
    """A sequence of related queries in a session."""
    session_id: str
    queries: List[str]
    timestamps: List[datetime]
    contexts: List[Optional[str]]
    selected_results: List[Optional[str]]


@dataclass
class RetrievalCluster:
    """A cluster of memories that get retrieved together."""
    memory_ids: Set[str]
    query_patterns: List[str]
    frequency: int
    avg_score: float


class UsageLearner:
    """
    Learn behavioral patterns from system usage.

    Tracks queries, retrievals, and user actions to learn:
    - What queries lead to what results
    - What sequences users follow
    - What makes results satisfying
    - How to pre-fetch relevant content

    Example:
        learner = UsageLearner()

        # Record a query event
        learner.record_query(
            query="fatigue treatments",
            results=[(mem_id, score), ...],
            context="tcm"
        )

        # Record user selection
        learner.record_selection(
            query="fatigue treatments",
            selected_id="mem_123"
        )

        # Get learned patterns
        patterns = learner.extract_patterns()

        # Get predictions for next query
        predictions = learner.predict_next("fatigue treatments")
    """

    def __init__(
        self,
        session_timeout: int = 1800,  # 30 minutes
        min_sequence_length: int = 2,
        min_cluster_frequency: int = 2,
    ):
        """
        Initialize usage learner.

        Args:
            session_timeout: Seconds before session expires
            min_sequence_length: Minimum queries for sequence pattern
            min_cluster_frequency: Minimum frequency for retrieval cluster
        """
        self.session_timeout = session_timeout
        self.min_sequence_length = min_sequence_length
        self.min_cluster_frequency = min_cluster_frequency

        # Event storage
        self.events: List[UsageEvent] = []

        # Learned structures
        self.query_sequences: Dict[str, QuerySequence] = {}
        self.retrieval_clusters: List[RetrievalCluster] = []
        self.query_to_results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.query_transitions: Dict[str, Counter] = defaultdict(Counter)
        self.selection_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_query(
        self,
        query: str,
        results: List[Tuple[str, float]],  # (memory_id, score) pairs
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> UsageEvent:
        """
        Record a query event.

        Args:
            query: The query string
            results: List of (memory_id, score) tuples
            context: Optional context/category
            session_id: Optional session ID for sequence tracking

        Returns:
            The created UsageEvent
        """
        event = UsageEvent(
            id=self._generate_id("query"),
            event_type="query",
            query=query,
            query_context=context,
            result_ids=[r[0] for r in results],
            result_scores=[r[1] for r in results],
            session_id=session_id,
        )

        # Link to previous event in session
        if session_id:
            prev_events = [
                e for e in reversed(self.events)
                if e.session_id == session_id
            ]
            if prev_events:
                event.previous_event_id = prev_events[0].id

        self.events.append(event)

        # Update query-results mapping
        query_normalized = self._normalize_query(query)
        self.query_to_results[query_normalized].extend(results)

        # Update query transitions
        if event.previous_event_id:
            prev_event = next(
                (e for e in self.events if e.id == event.previous_event_id),
                None
            )
            if prev_event and prev_event.query:
                prev_normalized = self._normalize_query(prev_event.query)
                self.query_transitions[prev_normalized][query_normalized] += 1

        # Update session sequence
        if session_id:
            if session_id not in self.query_sequences:
                self.query_sequences[session_id] = QuerySequence(
                    session_id=session_id,
                    queries=[],
                    timestamps=[],
                    contexts=[],
                    selected_results=[],
                )
            seq = self.query_sequences[session_id]
            seq.queries.append(query)
            seq.timestamps.append(event.timestamp)
            seq.contexts.append(context)
            seq.selected_results.append(None)

        return event

    def record_selection(
        self,
        query: str,
        selected_id: str,
        session_id: Optional[str] = None,
        feedback_score: Optional[float] = None,
    ) -> UsageEvent:
        """
        Record that a user selected a specific result.

        Args:
            query: The query that produced results
            selected_id: ID of the selected memory
            session_id: Optional session ID
            feedback_score: Optional explicit feedback (0-1)

        Returns:
            The created UsageEvent
        """
        event = UsageEvent(
            id=self._generate_id("selection"),
            event_type="selection",
            query=query,
            selected_id=selected_id,
            session_id=session_id,
            feedback_score=feedback_score,
        )

        self.events.append(event)

        # Update selection statistics
        query_normalized = self._normalize_query(query)
        self.selection_stats[query_normalized][selected_id] += 1

        # Update session sequence
        if session_id and session_id in self.query_sequences:
            seq = self.query_sequences[session_id]
            if seq.selected_results:
                seq.selected_results[-1] = selected_id

        return event

    def record_feedback(
        self,
        memory_id: str,
        feedback_score: float,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> UsageEvent:
        """
        Record explicit user feedback on a memory.

        Args:
            memory_id: ID of the rated memory
            feedback_score: Score from 0 (bad) to 1 (good)
            query: Optional query that led to this memory
            session_id: Optional session ID

        Returns:
            The created UsageEvent
        """
        event = UsageEvent(
            id=self._generate_id("feedback"),
            event_type="feedback",
            query=query,
            selected_id=memory_id,
            feedback_score=feedback_score,
            session_id=session_id,
        )

        self.events.append(event)
        return event

    def extract_patterns(self) -> List[Pattern]:
        """
        Extract learned patterns from accumulated usage data.

        Returns:
            List of behavioral patterns
        """
        patterns = []

        # 1. Query sequence patterns
        sequence_patterns = self._extract_sequence_patterns()
        patterns.extend(sequence_patterns)

        # 2. Retrieval cluster patterns
        cluster_patterns = self._extract_cluster_patterns()
        patterns.extend(cluster_patterns)

        # 3. Selection preference patterns
        selection_patterns = self._extract_selection_patterns()
        patterns.extend(selection_patterns)

        # 4. Query transition patterns
        transition_patterns = self._extract_transition_patterns()
        patterns.extend(transition_patterns)

        return patterns

    def _extract_sequence_patterns(self) -> List[Pattern]:
        """Extract patterns from query sequences."""
        patterns = []

        # Filter active sessions
        active_sequences = [
            seq for seq in self.query_sequences.values()
            if len(seq.queries) >= self.min_sequence_length
        ]

        if not active_sequences:
            return patterns

        # Find common subsequences
        subsequence_counts = Counter()
        for seq in active_sequences:
            # Extract all 2-3 query subsequences
            for length in range(2, min(4, len(seq.queries) + 1)):
                for i in range(len(seq.queries) - length + 1):
                    subseq = tuple(
                        self._normalize_query(q)
                        for q in seq.queries[i:i+length]
                    )
                    subsequence_counts[subseq] += 1

        # Create patterns for common subsequences
        for subseq, count in subsequence_counts.most_common(10):
            if count >= self.min_sequence_length:
                evidence = [
                    PatternEvidence(
                        source_id=f"sequence_{i}",
                        source_type="usage",
                        confidence=min(1.0, count / 10),
                        relevance=1.0,
                        detection_method="sequence_analysis",
                        snippets=list(subseq),
                        metadata={"count": count},
                    )
                    for i in range(min(count, 5))
                ]

                pattern = Pattern(
                    id=self._generate_id("sequence"),
                    name=f"Query Sequence: {' → '.join(subseq[:2])}{'...' if len(subseq) > 2 else ''}",
                    description=f"Users often query: {' → '.join(subseq)}",
                    pattern_type=PatternType.BEHAVIORAL,
                    evidence=evidence,
                    keywords=list(subseq),
                    tags=["query_sequence", "behavioral"],
                )
                pattern._recalculate_confidence()
                patterns.append(pattern)

        return patterns

    def _extract_cluster_patterns(self) -> List[Pattern]:
        """Extract patterns from retrieval clusters."""
        patterns = []

        # Build co-retrieval matrix
        coretrieval = defaultdict(Counter)
        for query, results in self.query_to_results.items():
            result_ids = [r[0] for r in results[:10]]  # Top 10
            for i, id_a in enumerate(result_ids):
                for id_b in result_ids[i+1:]:
                    coretrieval[id_a][id_b] += 1
                    coretrieval[id_b][id_a] += 1

        # Find clusters
        processed = set()
        clusters = []

        for memory_id, co_memories in coretrieval.items():
            if memory_id in processed:
                continue

            cluster = {memory_id}
            for co_id, count in co_memories.most_common(5):
                if count >= self.min_cluster_frequency:
                    cluster.add(co_id)

            if len(cluster) >= 2:
                clusters.append(cluster)
                processed.update(cluster)

        # Create patterns for clusters
        for cluster in clusters[:10]:
            # Find queries that retrieved this cluster
            cluster_queries = []
            for query, results in self.query_to_results.items():
                result_ids = set(r[0] for r in results[:10])
                overlap = len(cluster & result_ids)
                if overlap >= len(cluster) * 0.5:
                    cluster_queries.append(query)

            evidence = [
                PatternEvidence(
                    source_id=mem_id,
                    source_type="memory",
                    confidence=0.7,
                    relevance=1.0,
                    detection_method="cluster_analysis",
                    metadata={"cluster_size": len(cluster)},
                )
                for mem_id in list(cluster)[:5]
            ]

            pattern = Pattern(
                id=self._generate_id("cluster"),
                name=f"Retrieval Cluster ({len(cluster)} items)",
                description=f"These memories are frequently retrieved together",
                pattern_type=PatternType.RETRIEVAL,
                evidence=evidence,
                keywords=cluster_queries[:5],
                tags=["retrieval_cluster", "co-occurrence"],
                metadata={"memory_ids": list(cluster)},
            )
            pattern._recalculate_confidence()
            patterns.append(pattern)

        return patterns

    def _extract_selection_patterns(self) -> List[Pattern]:
        """Extract patterns from user selections."""
        patterns = []

        # Find queries with clear selection preferences
        for query, selections in self.selection_stats.items():
            total_selections = sum(selections.values())
            if total_selections < 3:
                continue

            # Find dominant selection
            top_id, top_count = max(selections.items(), key=lambda x: x[1])
            selection_rate = top_count / total_selections

            if selection_rate >= 0.5:
                evidence = [
                    PatternEvidence(
                        source_id=top_id,
                        source_type="usage",
                        confidence=selection_rate,
                        relevance=1.0,
                        detection_method="selection_analysis",
                        metadata={
                            "selection_count": top_count,
                            "total_selections": total_selections,
                        },
                    )
                ]

                pattern = Pattern(
                    id=self._generate_id("selection"),
                    name=f"Preferred Result for '{query[:30]}...'",
                    description=f"Users prefer memory {top_id} ({selection_rate:.0%} of selections)",
                    pattern_type=PatternType.BEHAVIORAL,
                    evidence=evidence,
                    keywords=[query],
                    tags=["selection_preference", "user_choice"],
                    metadata={
                        "query": query,
                        "preferred_id": top_id,
                        "selection_rate": selection_rate,
                    },
                )
                pattern._recalculate_confidence()
                patterns.append(pattern)

        return patterns

    def _extract_transition_patterns(self) -> List[Pattern]:
        """Extract patterns from query transitions."""
        patterns = []

        for from_query, to_counts in self.query_transitions.items():
            total = sum(to_counts.values())
            if total < 2:
                continue

            for to_query, count in to_counts.most_common(3):
                transition_rate = count / total
                if transition_rate >= 0.3 and count >= 2:
                    evidence = [
                        PatternEvidence(
                            source_id=f"transition_{from_query}_{to_query}",
                            source_type="usage",
                            confidence=transition_rate,
                            relevance=1.0,
                            detection_method="transition_analysis",
                            metadata={
                                "from_query": from_query,
                                "to_query": to_query,
                                "count": count,
                            },
                        )
                    ]

                    pattern = Pattern(
                        id=self._generate_id("transition"),
                        name=f"Query Flow: {from_query[:20]} → {to_query[:20]}",
                        description=f"After searching '{from_query}', users often search '{to_query}'",
                        pattern_type=PatternType.TEMPORAL,
                        evidence=evidence,
                        keywords=[from_query, to_query],
                        tags=["query_transition", "temporal"],
                        metadata={
                            "from_query": from_query,
                            "to_query": to_query,
                            "transition_rate": transition_rate,
                        },
                    )
                    pattern._recalculate_confidence()
                    patterns.append(pattern)

        return patterns

    def predict_next(
        self,
        current_query: str,
        n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Predict what the user might query next.

        Args:
            current_query: Current/most recent query
            n: Number of predictions to return

        Returns:
            List of predicted next queries with confidence
        """
        predictions = []
        query_normalized = self._normalize_query(current_query)

        if query_normalized in self.query_transitions:
            transitions = self.query_transitions[query_normalized]
            total = sum(transitions.values())

            for next_query, count in transitions.most_common(n):
                predictions.append({
                    "query": next_query,
                    "confidence": count / total,
                    "count": count,
                })

        return predictions

    def get_preferred_results(
        self,
        query: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get preferred results for a query based on past selections.

        Args:
            query: The query
            n: Number of results to return

        Returns:
            List of preferred memory IDs with selection rates
        """
        results = []
        query_normalized = self._normalize_query(query)

        if query_normalized in self.selection_stats:
            selections = self.selection_stats[query_normalized]
            total = sum(selections.values())

            for mem_id, count in sorted(
                selections.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]:
                results.append({
                    "memory_id": mem_id,
                    "selection_rate": count / total,
                    "selection_count": count,
                })

        return results

    def get_related_queries(
        self,
        query: str,
    ) -> List[str]:
        """
        Get queries related to the given query.

        Args:
            query: The query to find related queries for

        Returns:
            List of related query strings
        """
        related = set()
        query_normalized = self._normalize_query(query)

        # Queries that lead to this one
        for from_q, transitions in self.query_transitions.items():
            if query_normalized in transitions:
                related.add(from_q)

        # Queries that follow this one
        if query_normalized in self.query_transitions:
            related.update(self.query_transitions[query_normalized].keys())

        return list(related)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get overall usage statistics."""
        query_events = [e for e in self.events if e.event_type == "query"]
        selection_events = [e for e in self.events if e.event_type == "selection"]

        return {
            "total_events": len(self.events),
            "total_queries": len(query_events),
            "total_selections": len(selection_events),
            "unique_queries": len(self.query_to_results),
            "active_sessions": len(self.query_sequences),
            "transition_patterns": sum(
                len(t) for t in self.query_transitions.values()
            ),
        }

    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison."""
        return query.lower().strip()

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        unique = f"{prefix}_{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique.encode()).hexdigest()[:16]
