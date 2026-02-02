"""
Pattern Connection Weaver - Discovers hidden connections between patterns.

Part of the SIMBA (Self-Improving Meta-learning Brain Architecture) system.
Adapted from Distillation Engine's PatternConnectionWeaver.

Features:
- Multi-type connection detection (semantic, temporal, causal, complementary, synergistic)
- Network metrics calculation (density, connectivity score)
- Cluster identification
- Hub pattern detection
- Strong connection chain finding
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class Connection:
    """A connection between two patterns."""
    to_pattern_id: str
    connection_type: str
    strength: float
    bidirectional: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionInsight:
    """An insight derived from connection analysis."""
    insight_type: str
    description: str
    pattern_ids: List[str]
    recommendation: str
    confidence: float = 0.8


@dataclass
class NetworkMetrics:
    """Metrics describing the pattern connection network."""
    nodes: int
    edges: int
    density: float
    average_connection_strength: float
    connectivity_score: float


@dataclass
class Cluster:
    """A cluster of related patterns."""
    patterns: List[str]
    size: int
    cohesion: float


@dataclass
class ConnectionResult:
    """Result of connection weaving analysis."""
    connections: Dict[str, List[Connection]]
    score: float
    insights: List[ConnectionInsight]
    network_metrics: NetworkMetrics
    clusters: List[Cluster]


class PatternConnectionWeaver:
    """
    Discovers hidden connections between patterns.

    Analyzes patterns to find:
    - Semantic connections (similar meaning/purpose)
    - Temporal connections (occur together in time)
    - Causal connections (one leads to another)
    - Complementary connections (work well together)
    - Synergistic connections (multiply value when combined)

    Example:
        weaver = PatternConnectionWeaver()
        result = weaver.weave_connections(patterns, context)

        # Get hub patterns (highly connected)
        hubs = [i for i in result.insights if i.insight_type == 'hub_patterns']

        # Get pattern clusters
        for cluster in result.clusters:
            print(f"Cluster: {cluster.patterns}, cohesion: {cluster.cohesion}")
    """

    CONNECTION_TYPES = ['semantic', 'temporal', 'causal', 'complementary', 'synergistic']

    def __init__(
        self,
        connection_threshold: float = 0.5,
        cluster_threshold: float = 0.6,
    ):
        """
        Initialize the connection weaver.

        Args:
            connection_threshold: Min strength to consider a connection
            cluster_threshold: Min strength for cluster expansion
        """
        self.connection_threshold = connection_threshold
        self.cluster_threshold = cluster_threshold
        self.connection_graph: Dict[str, List[Dict]] = defaultdict(list)

    def weave_connections(
        self,
        patterns: List[Any],
        context: Dict[str, Any] = None,
    ) -> ConnectionResult:
        """
        Discover connections between patterns.

        Args:
            patterns: List of Pattern objects
            context: Optional context for causal analysis

        Returns:
            ConnectionResult with connections, insights, clusters
        """
        context = context or {}
        connections: Dict[str, List[Connection]] = defaultdict(list)
        total_strength = 0.0
        connection_count = 0

        # Analyze all pattern pairs
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                connection = self._find_connection(pattern1, pattern2, context)

                if connection["strength"] > self.connection_threshold:
                    p1_id = pattern1.id
                    p2_id = pattern2.id

                    # Add connection for pattern1
                    connections[p1_id].append(Connection(
                        to_pattern_id=p2_id,
                        connection_type=connection["type"],
                        strength=connection["strength"],
                        bidirectional=connection["bidirectional"],
                        metadata=connection["metadata"],
                    ))

                    # Add reverse connection if bidirectional
                    if connection["bidirectional"]:
                        connections[p2_id].append(Connection(
                            to_pattern_id=p1_id,
                            connection_type=connection["type"],
                            strength=connection["strength"],
                            bidirectional=True,
                            metadata=connection["metadata"],
                        ))

                    total_strength += connection["strength"]
                    connection_count += 1

                    # Update graph
                    self._update_graph(p1_id, p2_id, connection)

        # Calculate score
        score = total_strength / connection_count if connection_count > 0 else 0.0

        # Extract insights
        insights = self._extract_insights(connections, patterns)

        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(connections)

        # Identify clusters
        clusters = self._identify_clusters(connections)

        return ConnectionResult(
            connections=dict(connections),
            score=score,
            insights=insights,
            network_metrics=network_metrics,
            clusters=clusters,
        )

    def _find_connection(
        self,
        pattern1: Any,
        pattern2: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Find the strongest connection between two patterns."""
        # Calculate all connection types
        connections = [
            ("semantic", self._calc_semantic_similarity(pattern1, pattern2)),
            ("temporal", self._calc_temporal_correlation(pattern1, pattern2)),
            ("causal", self._calc_causal_relationship(pattern1, pattern2, context)),
            ("complementary", self._calc_complementarity(pattern1, pattern2)),
            ("synergistic", self._calc_synergy(pattern1, pattern2)),
        ]

        # Find strongest
        strongest = max(connections, key=lambda x: x[1])

        return {
            "type": strongest[0],
            "strength": strongest[1],
            "bidirectional": strongest[0] != "causal",
            "metadata": {
                "all_connections": {c[0]: c[1] for c in connections},
                "context_relevance": self._calc_context_relevance(pattern1, pattern2, context),
            },
        }

    def _calc_semantic_similarity(self, p1: Any, p2: Any) -> float:
        """Calculate semantic similarity between patterns."""
        similarity = 0.0

        # Pattern type match
        if hasattr(p1, 'pattern_type') and hasattr(p2, 'pattern_type'):
            if p1.pattern_type == p2.pattern_type:
                similarity += 0.25

        # Context match
        if hasattr(p1, 'context') and hasattr(p2, 'context'):
            if p1.context and p2.context and p1.context == p2.context:
                similarity += 0.25

        # Keyword overlap (Jaccard similarity)
        kw1 = set(getattr(p1, 'keywords', []) or [])
        kw2 = set(getattr(p2, 'keywords', []) or [])
        if kw1 and kw2:
            intersection = len(kw1 & kw2)
            union = len(kw1 | kw2)
            if union > 0:
                similarity += 0.3 * (intersection / union)

        # Tag overlap
        tags1 = set(getattr(p1, 'tags', []) or [])
        tags2 = set(getattr(p2, 'tags', []) or [])
        if tags1 and tags2:
            intersection = len(tags1 & tags2)
            union = len(tags1 | tags2)
            if union > 0:
                similarity += 0.2 * (intersection / union)

        return min(1.0, similarity)

    def _calc_temporal_correlation(self, p1: Any, p2: Any) -> float:
        """Calculate temporal correlation between patterns."""
        t1 = getattr(p1, 'created_at', None)
        t2 = getattr(p2, 'created_at', None)

        if not t1 or not t2:
            return 0.0

        if isinstance(t1, str):
            t1 = datetime.fromisoformat(t1)
        if isinstance(t2, str):
            t2 = datetime.fromisoformat(t2)

        time_diff = abs((t1 - t2).total_seconds())
        hours_diff = time_diff / 3600

        # Strong correlation if within same hour
        if hours_diff < 1:
            return 0.9
        # Good correlation within same day
        if hours_diff < 24:
            return 0.7
        # Moderate correlation within same week
        if hours_diff < 168:
            return 0.5
        # Weak correlation within same month
        if hours_diff < 720:
            return 0.3

        return 0.1

    def _calc_causal_relationship(
        self,
        p1: Any,
        p2: Any,
        context: Dict[str, Any],
    ) -> float:
        """Detect causal relationship between patterns."""
        causal_strength = 0.0

        # Check if in workflow context
        workflow = context.get("workflow", {})
        steps = workflow.get("steps", [])

        if steps:
            p1_id = p1.id
            p2_id = p2.id

            idx1 = next((i for i, s in enumerate(steps) if s.get("pattern_id") == p1_id), -1)
            idx2 = next((i for i, s in enumerate(steps) if s.get("pattern_id") == p2_id), -1)

            if idx1 != -1 and idx2 != -1 and idx1 < idx2:
                distance = idx2 - idx1
                if distance == 1:
                    causal_strength += 0.5  # Direct sequence
                elif distance <= 3:
                    causal_strength += 0.3  # Close in sequence

        # Check source pattern relationships
        relationships = getattr(p1, 'relationships', [])
        for rel in relationships:
            if hasattr(rel, 'target_pattern_id') and rel.target_pattern_id == p2.id:
                if hasattr(rel, 'relationship_type'):
                    rel_type = str(rel.relationship_type).lower()
                    if 'cause' in rel_type or 'lead' in rel_type:
                        causal_strength += 0.4

        return min(1.0, causal_strength)

    def _calc_complementarity(self, p1: Any, p2: Any) -> float:
        """Assess if patterns are complementary."""
        complementarity = 0.0

        # Complementary pattern types
        complementary_pairs = {
            "linguistic": ["semantic", "structural"],
            "structural": ["behavioral", "temporal"],
            "semantic": ["emotional", "quality"],
            "behavioral": ["query", "retrieval"],
        }

        p1_type = str(getattr(p1, 'pattern_type', '')).lower()
        p2_type = str(getattr(p2, 'pattern_type', '')).lower()

        if p1_type in complementary_pairs:
            if p2_type in complementary_pairs.get(p1_type, []):
                complementarity += 0.5

        # Keyword complementarity (different but related)
        kw1 = set(getattr(p1, 'keywords', []) or [])
        kw2 = set(getattr(p2, 'keywords', []) or [])

        if kw1 and kw2:
            # Low overlap but some connection = complementary
            intersection = len(kw1 & kw2)
            total = len(kw1) + len(kw2)
            if total > 0:
                overlap_ratio = intersection * 2 / total
                # Sweet spot: some overlap (0.1-0.4) indicates complementarity
                if 0.1 <= overlap_ratio <= 0.4:
                    complementarity += 0.3

        return min(1.0, complementarity)

    def _calc_synergy(self, p1: Any, p2: Any) -> float:
        """Calculate synergistic potential when patterns combine."""
        synergy = 0.0

        # High confidence patterns together = more synergy
        c1 = getattr(p1, 'confidence', 0.5)
        c2 = getattr(p2, 'confidence', 0.5)

        if c1 > 0.7 and c2 > 0.7:
            synergy += 0.3

        # Patterns that cover different evidence sources
        e1_sources = {e.source_id for e in getattr(p1, 'evidence', [])}
        e2_sources = {e.source_id for e in getattr(p2, 'evidence', [])}

        if e1_sources and e2_sources:
            # Low source overlap = different perspectives = synergy
            overlap = len(e1_sources & e2_sources)
            total = len(e1_sources | e2_sources)
            if total > 0 and overlap / total < 0.3:
                synergy += 0.4

        # Related but different keywords
        kw1 = set(getattr(p1, 'keywords', []) or [])
        kw2 = set(getattr(p2, 'keywords', []) or [])

        if kw1 and kw2:
            union = len(kw1 | kw2)
            intersection = len(kw1 & kw2)
            if union > 0:
                # Moderate overlap is good for synergy
                overlap_ratio = intersection / union
                if 0.2 <= overlap_ratio <= 0.5:
                    synergy += 0.3

        return min(1.0, synergy)

    def _calc_context_relevance(
        self,
        p1: Any,
        p2: Any,
        context: Dict[str, Any],
    ) -> float:
        """Calculate how relevant both patterns are to the context."""
        if not context:
            return 0.5

        relevance = 0.0
        ctx_domain = context.get("domain")

        if ctx_domain:
            if getattr(p1, 'context', None) == ctx_domain:
                relevance += 0.25
            if getattr(p2, 'context', None) == ctx_domain:
                relevance += 0.25

        return relevance

    def _update_graph(
        self,
        p1_id: str,
        p2_id: str,
        connection: Dict[str, Any],
    ) -> None:
        """Update the internal connection graph."""
        key = f"{p1_id}-{p2_id}"
        self.connection_graph[key].append({
            "type": connection["type"],
            "strength": connection["strength"],
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last 10 for memory efficiency
        if len(self.connection_graph[key]) > 10:
            self.connection_graph[key] = self.connection_graph[key][-10:]

    def _extract_insights(
        self,
        connections: Dict[str, List[Connection]],
        patterns: List[Any],
    ) -> List[ConnectionInsight]:
        """Extract insights from connection analysis."""
        insights = []

        # Find hub patterns (3+ connections)
        connection_counts = {
            pid: len(conns) for pid, conns in connections.items()
        }

        hubs = [
            (pid, count) for pid, count in connection_counts.items()
            if count >= 3
        ]
        hubs.sort(key=lambda x: x[1], reverse=True)

        if hubs:
            insights.append(ConnectionInsight(
                insight_type="hub_patterns",
                description=f"Found {len(hubs)} hub patterns with 3+ connections",
                pattern_ids=[h[0] for h in hubs],
                recommendation="Prioritize these patterns as they are central to the system",
                confidence=0.9,
            ))

        # Find isolated patterns (no connections)
        all_pattern_ids = {p.id for p in patterns}
        connected_ids = set(connections.keys())
        isolated = all_pattern_ids - connected_ids

        if isolated:
            insights.append(ConnectionInsight(
                insight_type="isolated_patterns",
                description=f"Found {len(isolated)} isolated patterns with no connections",
                pattern_ids=list(isolated),
                recommendation="Consider removing or better integrating these patterns",
                confidence=0.8,
            ))

        # Find strong connection chains
        chains = self._find_strong_chains(connections)
        if chains:
            insights.append(ConnectionInsight(
                insight_type="connection_chains",
                description=f"Found {len(chains)} strong pattern chains",
                pattern_ids=[p for chain in chains for p in chain],
                recommendation="Package these chains as composite patterns",
                confidence=0.85,
            ))

        return insights

    def _calculate_network_metrics(
        self,
        connections: Dict[str, List[Connection]],
    ) -> NetworkMetrics:
        """Calculate network metrics for the connection graph."""
        node_count = len(connections)
        edge_count = 0
        total_strength = 0.0

        for conns in connections.values():
            edge_count += len(conns)
            total_strength += sum(c.strength for c in conns)

        # Adjust for bidirectional counting
        edge_count = edge_count // 2

        # Density: actual edges / possible edges
        density = 0.0
        if node_count > 1:
            possible_edges = node_count * (node_count - 1) / 2
            density = edge_count / possible_edges if possible_edges > 0 else 0

        avg_strength = total_strength / (edge_count * 2) if edge_count > 0 else 0

        return NetworkMetrics(
            nodes=node_count,
            edges=edge_count,
            density=density,
            average_connection_strength=avg_strength,
            connectivity_score=density * avg_strength,
        )

    def _identify_clusters(
        self,
        connections: Dict[str, List[Connection]],
    ) -> List[Cluster]:
        """Identify clusters of related patterns."""
        clusters = []
        visited: Set[str] = set()

        for pattern_id in connections.keys():
            if pattern_id not in visited:
                cluster_patterns = self._expand_cluster(
                    pattern_id, connections, visited
                )

                if len(cluster_patterns) > 1:
                    cohesion = self._calc_cluster_cohesion(
                        cluster_patterns, connections
                    )
                    clusters.append(Cluster(
                        patterns=cluster_patterns,
                        size=len(cluster_patterns),
                        cohesion=cohesion,
                    ))

        # Sort by cohesion
        clusters.sort(key=lambda c: c.cohesion, reverse=True)
        return clusters

    def _expand_cluster(
        self,
        start_id: str,
        connections: Dict[str, List[Connection]],
        visited: Set[str],
    ) -> List[str]:
        """Expand a cluster from a starting pattern using BFS."""
        cluster = [start_id]
        visited.add(start_id)
        queue = [start_id]

        while queue:
            current_id = queue.pop(0)
            current_conns = connections.get(current_id, [])

            for conn in current_conns:
                if conn.to_pattern_id not in visited:
                    if conn.strength >= self.cluster_threshold:
                        visited.add(conn.to_pattern_id)
                        cluster.append(conn.to_pattern_id)
                        queue.append(conn.to_pattern_id)

        return cluster

    def _calc_cluster_cohesion(
        self,
        cluster: List[str],
        connections: Dict[str, List[Connection]],
    ) -> float:
        """Calculate internal cohesion of a cluster."""
        if len(cluster) < 2:
            return 0.0

        internal_connections = 0
        total_strength = 0.0
        cluster_set = set(cluster)

        for pattern_id in cluster:
            conns = connections.get(pattern_id, [])
            for conn in conns:
                if conn.to_pattern_id in cluster_set:
                    internal_connections += 1
                    total_strength += conn.strength

        possible_connections = len(cluster) * (len(cluster) - 1)
        density = internal_connections / possible_connections if possible_connections > 0 else 0
        avg_strength = total_strength / internal_connections if internal_connections > 0 else 0

        return density * avg_strength

    def _find_strong_chains(
        self,
        connections: Dict[str, List[Connection]],
        min_length: int = 3,
        min_strength: float = 0.7,
    ) -> List[List[str]]:
        """Find strong connection chains."""
        chains = []

        for start_id in connections.keys():
            for conn in connections[start_id]:
                if conn.strength >= min_strength:
                    chain = self._follow_chain(
                        start_id, connections, min_strength, set()
                    )
                    if len(chain) >= min_length:
                        chains.append(chain)

        # Remove duplicates
        unique_chains = []
        seen_keys = set()

        for chain in chains:
            key = "-".join(sorted(chain))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_chains.append(chain)

        return unique_chains

    def _follow_chain(
        self,
        current_id: str,
        connections: Dict[str, List[Connection]],
        min_strength: float,
        visited: Set[str],
    ) -> List[str]:
        """Follow a chain of strong connections."""
        if current_id in visited:
            return []

        visited.add(current_id)
        chain = [current_id]
        conns = connections.get(current_id, [])

        for conn in conns:
            if conn.strength >= min_strength and conn.to_pattern_id not in visited:
                sub_chain = self._follow_chain(
                    conn.to_pattern_id, connections, min_strength, visited
                )
                if sub_chain:
                    chain.extend(sub_chain)
                    break  # Follow one path

        return chain
