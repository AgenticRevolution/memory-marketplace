"""
Pattern Optimizer - Decay tracking, freshness, and re-ranking.

Part of the SIMBA system. Adapted from Distillation Engine.

Features:
- Pattern decay tracking (time-sensitive relevance)
- Freshness scoring
- Re-ranking metadata enrichment
- Query affinity calculation
- Impact magnitude calculation
- Synergy detection with execution order
- LRU caching for queries
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import math


@dataclass
class DecayInfo:
    """Decay information for a pattern."""
    original_confidence: float
    current_confidence: float
    decay_rate: float
    last_updated: str


@dataclass
class ReRankingMetadata:
    """Metadata for re-ranking patterns."""
    decay_factor: float
    current_relevance: float
    correlation_count: int
    strong_correlations: List[str]
    performance_score: float
    usage_frequency: int
    success_rate: float
    impact_magnitude: float
    query_affinity: float
    freshness_score: float


@dataclass
class PatternSynergy:
    """Synergy information between patterns."""
    pattern_id: str
    synergy_score: float
    synergy_type: str
    combined_impact: float
    suggested_order: List[str]


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get value, moving to end (most recently used)."""
        if key not in self.cache:
            return None

        # Move to end
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set value, evicting oldest if needed."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Evict oldest if over capacity
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.cache

    def values(self) -> List[Any]:
        """Get all values."""
        return list(self.cache.values())


class PatternOptimizer:
    """
    Optimizes patterns for retrieval and re-ranking.

    Provides:
    - Pattern decay tracking (relevance decreases over time)
    - Freshness scoring (newer patterns score higher)
    - Re-ranking metadata (comprehensive scoring for retrieval)
    - Query affinity (how well pattern matches common queries)
    - Synergy detection (patterns that work well together)

    Example:
        optimizer = PatternOptimizer()

        # Enrich pattern for re-ranking
        enriched = optimizer.enrich_for_reranking(pattern)

        # Find synergetic patterns
        synergies = optimizer.find_synergetic_patterns(base_pattern, candidates)

        # Optimize query with patterns
        optimization = optimizer.optimize_query(query, context)
    """

    # Configuration
    DECAY_HALF_LIFE_DAYS = 7  # Pattern relevance halves every 7 days
    CORRELATION_THRESHOLD = 0.7

    def __init__(self, query_cache_size: int = 1000):
        """Initialize the pattern optimizer."""
        self.decay_tracker: Dict[str, DecayInfo] = {}
        self.correlation_graph: Dict[str, Set[Tuple[str, float]]] = {}
        self.performance_metrics: Dict[str, Dict] = {}
        self.query_cache = LRUCache(query_cache_size)

    def calculate_decay(self, pattern: Any) -> float:
        """
        Calculate pattern decay factor.

        Args:
            pattern: Pattern object

        Returns:
            Decay factor (0-1, where 1 is no decay)
        """
        created_at = getattr(pattern, 'created_at', None)
        if not created_at:
            return 1.0

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        age_seconds = (datetime.now() - created_at).total_seconds()
        half_life_seconds = self.DECAY_HALF_LIFE_DAYS * 24 * 60 * 60

        # Exponential decay formula
        decay_factor = math.exp(-0.693 * age_seconds / half_life_seconds)

        # Store decay info
        confidence = getattr(pattern, 'confidence', 0.5)
        self.decay_tracker[pattern.id] = DecayInfo(
            original_confidence=confidence,
            current_confidence=confidence * decay_factor,
            decay_rate=0.693 / half_life_seconds,
            last_updated=datetime.now().isoformat(),
        )

        return decay_factor

    def calculate_freshness(self, pattern: Any) -> float:
        """
        Calculate pattern freshness score.

        Args:
            pattern: Pattern object

        Returns:
            Freshness score (0-1, where 1 is newest)
        """
        created_at = getattr(pattern, 'created_at', None)
        if not created_at:
            return 0.5

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        age_days = (datetime.now() - created_at).days

        # Freshness scoring
        if age_days < 1:
            return 1.0
        elif age_days < 7:
            return 0.8
        elif age_days < 30:
            return 0.6
        elif age_days < 90:
            return 0.4
        return 0.2

    def calculate_impact_magnitude(self, pattern: Any) -> float:
        """
        Calculate pattern impact magnitude.

        Args:
            pattern: Pattern object

        Returns:
            Impact magnitude (0-1+, can exceed 1 with multipliers)
        """
        # Impact multipliers for different domains
        impact_multipliers = {
            'revenue': 1.5,
            'efficiency': 1.2,
            'user_experience': 1.3,
            'system_stability': 1.4,
            'cost_reduction': 1.3,
            'performance': 1.2,
        }

        # Base impact from confidence and success rate
        confidence = getattr(pattern, 'confidence', 0.5)
        success_rate = getattr(pattern, 'success_rate', 0.5)
        base_impact = (confidence + success_rate) / 2

        # Apply multipliers from keywords/tags
        keywords = set(kw.lower() for kw in (getattr(pattern, 'keywords', []) or []))
        tags = set(t.lower() for t in (getattr(pattern, 'tags', []) or []))
        all_terms = keywords | tags

        magnitude = base_impact
        for term, multiplier in impact_multipliers.items():
            if term in all_terms:
                magnitude *= multiplier

        return min(magnitude, 2.0)  # Cap at 2x

    def calculate_query_affinity(
        self,
        pattern: Any,
        recent_queries: List[str] = None,
    ) -> float:
        """
        Calculate how well pattern matches common queries.

        Args:
            pattern: Pattern object
            recent_queries: Optional list of recent queries

        Returns:
            Query affinity score (0-1)
        """
        if recent_queries is None:
            recent_queries = self.query_cache.values()

        if not recent_queries:
            return 0.5

        keywords = set(kw.lower() for kw in (getattr(pattern, 'keywords', []) or []))
        if not keywords:
            return 0.3

        total_affinity = 0.0
        matched_queries = 0

        for query in recent_queries:
            if isinstance(query, str):
                query_words = set(query.lower().split())
                overlap = len(keywords & query_words)
                if overlap > 0:
                    affinity = overlap / len(query_words) if query_words else 0
                    total_affinity += affinity
                    matched_queries += 1

        return total_affinity / matched_queries if matched_queries > 0 else 0.3

    def enrich_for_reranking(
        self,
        pattern: Any,
        recent_queries: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Enrich pattern with re-ranking metadata.

        Args:
            pattern: Pattern object
            recent_queries: Optional recent queries for affinity

        Returns:
            Dict with pattern and reranking_metadata
        """
        decay = self.calculate_decay(pattern)
        confidence = getattr(pattern, 'confidence', 0.5)

        # Get correlations
        correlations = self.correlation_graph.get(pattern.id, set())
        strong_correlations = [
            pid for pid, score in correlations
            if score > 0.8
        ]

        # Get performance
        perf = self.performance_metrics.get(pattern.id, {
            'score': 0.5,
            'usage_count': 0,
            'success_rate': 0.5,
        })

        metadata = ReRankingMetadata(
            decay_factor=decay,
            current_relevance=confidence * decay,
            correlation_count=len(correlations),
            strong_correlations=strong_correlations,
            performance_score=perf['score'],
            usage_frequency=perf['usage_count'],
            success_rate=perf['success_rate'],
            impact_magnitude=self.calculate_impact_magnitude(pattern),
            query_affinity=self.calculate_query_affinity(pattern, recent_queries),
            freshness_score=self.calculate_freshness(pattern),
        )

        return {
            "pattern": pattern,
            "reranking_metadata": metadata,
        }

    def batch_optimize(
        self,
        patterns: List[Any],
        recent_queries: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch optimize patterns for re-ranking.

        Args:
            patterns: List of Pattern objects
            recent_queries: Optional recent queries

        Returns:
            List of enriched patterns with metadata
        """
        return [
            self.enrich_for_reranking(p, recent_queries)
            for p in patterns
        ]

    def find_synergetic_patterns(
        self,
        base_pattern: Any,
        candidates: List[Any],
        min_synergy: float = 0.5,
    ) -> List[PatternSynergy]:
        """
        Find patterns that work well with the base pattern.

        Args:
            base_pattern: Pattern to find synergies for
            candidates: Candidate patterns
            min_synergy: Minimum synergy score

        Returns:
            List of PatternSynergy objects, sorted by combined impact
        """
        synergies = []

        for candidate in candidates:
            if candidate.id == base_pattern.id:
                continue

            synergy = self._calculate_synergy(base_pattern, candidate)
            if synergy['score'] >= min_synergy:
                synergies.append(PatternSynergy(
                    pattern_id=candidate.id,
                    synergy_score=synergy['score'],
                    synergy_type=synergy['type'],
                    combined_impact=synergy['combined_impact'],
                    suggested_order=synergy['order'],
                ))

        # Sort by combined impact
        synergies.sort(key=lambda s: s.combined_impact, reverse=True)
        return synergies

    def _calculate_synergy(
        self,
        pattern1: Any,
        pattern2: Any,
    ) -> Dict[str, Any]:
        """Calculate detailed synergy between two patterns."""
        # Category match
        ctx1 = getattr(pattern1, 'context', '')
        ctx2 = getattr(pattern2, 'context', '')
        category_match = 0.2 if ctx1 == ctx2 else 0.0

        # Type complementarity
        type1 = str(getattr(pattern1, 'pattern_type', '')).lower()
        type2 = str(getattr(pattern2, 'pattern_type', '')).lower()

        complementary_types = [
            ('linguistic', 'semantic'),
            ('structural', 'behavioral'),
            ('query', 'retrieval'),
        ]

        type_complement = 0.0
        for pair in complementary_types:
            if (type1 in pair and type2 in pair and type1 != type2):
                type_complement = 0.3
                break

        # Keyword synergy
        kw1 = set(getattr(pattern1, 'keywords', []) or [])
        kw2 = set(getattr(pattern2, 'keywords', []) or [])

        keyword_synergy = 0.0
        if kw1 and kw2:
            overlap = len(kw1 & kw2) / len(kw1 | kw2)
            # Sweet spot: 20-50% overlap
            if 0.2 <= overlap <= 0.5:
                keyword_synergy = 0.3

        # Calculate scores
        base_score = category_match + type_complement + keyword_synergy

        # Determine type
        if type_complement > 0:
            synergy_type = "complementary"
        elif category_match > 0:
            synergy_type = "same_domain"
        else:
            synergy_type = "general"

        # Combined impact
        impact1 = self.calculate_impact_magnitude(pattern1)
        impact2 = self.calculate_impact_magnitude(pattern2)
        combined_impact = (impact1 + impact2) * (1 + base_score)

        # Suggested order (higher confidence first)
        conf1 = getattr(pattern1, 'confidence', 0.5)
        conf2 = getattr(pattern2, 'confidence', 0.5)
        order = [pattern1.id, pattern2.id] if conf1 >= conf2 else [pattern2.id, pattern1.id]

        return {
            'score': min(1.0, base_score),
            'type': synergy_type,
            'combined_impact': combined_impact,
            'order': order,
        }

    def optimize_query(
        self,
        query: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a query using pattern knowledge.

        Args:
            query: Query string
            context: Optional context

        Returns:
            Dict with optimization suggestions
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Check cache
        cached = self.query_cache.get(query_hash)
        if cached:
            return cached

        # Store query for affinity calculation
        self.query_cache.set(query_hash, query)

        # Extract query features
        query_words = set(query.lower().split())

        # Generate optimization
        optimization = {
            "query": query,
            "query_words": list(query_words),
            "suggested_expansions": self._suggest_expansions(query_words),
            "context_recommendations": self._recommend_context(query_words, context),
            "optimization_applied": True,
        }

        return optimization

    def _suggest_expansions(self, query_words: Set[str]) -> List[str]:
        """Suggest query expansions."""
        # Synonym/related term suggestions
        expansions_map = {
            'fatigue': ['tiredness', 'exhaustion', 'weakness'],
            'treatment': ['remedy', 'therapy', 'cure'],
            'pattern': ['structure', 'template', 'model'],
            'learn': ['study', 'understand', 'discover'],
        }

        suggestions = []
        for word in query_words:
            if word in expansions_map:
                suggestions.extend(expansions_map[word])

        return suggestions[:5]

    def _recommend_context(
        self,
        query_words: Set[str],
        context: Dict[str, Any] = None,
    ) -> List[str]:
        """Recommend contexts for the query."""
        # Context keywords
        context_indicators = {
            'tcm': ['herb', 'qi', 'yang', 'yin', 'acupuncture', 'formula'],
            'programming': ['code', 'function', 'api', 'database', 'algorithm'],
            'general': ['learn', 'understand', 'help', 'what', 'how'],
        }

        recommended = []
        for ctx, keywords in context_indicators.items():
            if query_words & set(keywords):
                recommended.append(ctx)

        return recommended if recommended else ['general']

    def record_performance(
        self,
        pattern_id: str,
        success: bool,
        score: float = None,
    ) -> None:
        """
        Record pattern performance for optimization.

        Args:
            pattern_id: Pattern ID
            success: Whether application was successful
            score: Optional performance score
        """
        if pattern_id not in self.performance_metrics:
            self.performance_metrics[pattern_id] = {
                'score': 0.5,
                'usage_count': 0,
                'success_count': 0,
                'success_rate': 0.5,
            }

        metrics = self.performance_metrics[pattern_id]
        metrics['usage_count'] += 1

        if success:
            metrics['success_count'] += 1

        metrics['success_rate'] = metrics['success_count'] / metrics['usage_count']

        if score is not None:
            # Running average
            old_score = metrics['score']
            metrics['score'] = (old_score * (metrics['usage_count'] - 1) + score) / metrics['usage_count']

    def add_correlation(
        self,
        pattern1_id: str,
        pattern2_id: str,
        correlation: float,
    ) -> None:
        """
        Add correlation between patterns.

        Args:
            pattern1_id: First pattern ID
            pattern2_id: Second pattern ID
            correlation: Correlation strength (0-1)
        """
        if correlation < self.CORRELATION_THRESHOLD:
            return

        if pattern1_id not in self.correlation_graph:
            self.correlation_graph[pattern1_id] = set()
        if pattern2_id not in self.correlation_graph:
            self.correlation_graph[pattern2_id] = set()

        self.correlation_graph[pattern1_id].add((pattern2_id, correlation))
        self.correlation_graph[pattern2_id].add((pattern1_id, correlation))
