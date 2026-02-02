"""
SIMBA Enhancer - Unified interface for all SIMBA dimensions.

Self-Improving Meta-learning Brain Architecture.
Adapted from Distillation Engine's DistillationSimba.

Combines all 5 enhancement dimensions:
1. Context Weaving (PatternConnectionWeaver)
2. Pattern Prophecy (PatternEvolutionPredictor)
3. Insight Crystallization (PatternWisdomExtractor)
4. Value Creation (PatternValueAssessor)
5. Emergence Guardian (PatternBreakthroughDetector)

Plus PatternOptimizer for decay and re-ranking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import logging

from .connection_weaver import PatternConnectionWeaver, ConnectionResult
from .evolution_predictor import PatternEvolutionPredictor, EvolutionResult
from .wisdom_extractor import PatternWisdomExtractor, WisdomResult
from .value_assessor import PatternValueAssessor, ValueResult
from .breakthrough_detector import PatternBreakthroughDetector, BreakthroughResult, Breakthrough
from .optimizer import PatternOptimizer

logger = logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    """Result of full SIMBA enhancement."""
    pattern_connections: ConnectionResult
    evolution_predictions: EvolutionResult
    crystallized_insights: WisdomResult
    pattern_valuations: ValueResult
    emergent_patterns: BreakthroughResult
    enhancement_score: float
    processing_time_ms: float


@dataclass
class SimbaMetrics:
    """SIMBA system metrics."""
    patterns_enhanced: int
    pattern_connections_discovered: int
    evolution_predictions_made: int
    insights_crystallized: int
    pattern_valuations: int
    breakthroughs_detected: int
    intelligence_level: float


class SimbaEnhancer:
    """
    Unified SIMBA Enhancement System.

    Integrates all 5 SIMBA dimensions to enhance pattern intelligence.

    Features:
    - Full pattern enhancement with all dimensions
    - Breakthrough detection and tracking
    - Self-improving intelligence level
    - Pattern optimization for retrieval
    - Metrics tracking

    Example:
        simba = SimbaEnhancer()

        # Enhance patterns with all dimensions
        result = simba.enhance_patterns(patterns, context)

        # Check for breakthroughs
        if result.enhancement_score > 0.8:
            print("High-quality enhancement achieved!")

        # Get breakthroughs
        breakthroughs = simba.get_breakthroughs()

        # Get metrics
        metrics = simba.get_metrics()
    """

    def __init__(self):
        """Initialize the SIMBA enhancer."""
        # Initialize all dimensions
        self.dimensions = {
            'context_weaving': PatternConnectionWeaver(),
            'pattern_prophecy': PatternEvolutionPredictor(),
            'insight_crystallization': PatternWisdomExtractor(),
            'value_creation': PatternValueAssessor(),
            'emergence_guardian': PatternBreakthroughDetector(),
        }

        # Pattern optimizer
        self.optimizer = PatternOptimizer()

        # Breakthrough storage
        self.breakthroughs: List[Breakthrough] = []

        # Metrics
        self.metrics = SimbaMetrics(
            patterns_enhanced=0,
            pattern_connections_discovered=0,
            evolution_predictions_made=0,
            insights_crystallized=0,
            pattern_valuations=0,
            breakthroughs_detected=0,
            intelligence_level=1.0,
        )

    def enhance_patterns(
        self,
        patterns: List[Any],
        context: Dict[str, Any] = None,
    ) -> EnhancementResult:
        """
        Enhance patterns using all 5 SIMBA dimensions.

        Args:
            patterns: List of Pattern objects
            context: Optional context for analysis

        Returns:
            EnhancementResult with all dimension results
        """
        start_time = datetime.now()
        context = context or {}

        # Apply all 5 dimensions
        connections = self.dimensions['context_weaving'].weave_connections(patterns, context)
        evolution = self.dimensions['pattern_prophecy'].predict_evolution(patterns)
        wisdom = self.dimensions['insight_crystallization'].extract_wisdom(patterns)
        valuations = self.dimensions['value_creation'].assess_value(patterns)
        emergence = self.dimensions['emergence_guardian'].detect_breakthroughs(patterns)

        # Calculate enhancement score
        enhancement_score = self._calculate_enhancement_score(
            connections, evolution, wisdom, valuations, emergence
        )

        # Track breakthroughs
        if enhancement_score > 0.8:
            self._handle_high_enhancement(patterns, enhancement_score, context)

        for breakthrough in emergence.breakthrough_candidates:
            self.breakthroughs.append(breakthrough)
            self.metrics.breakthroughs_detected += 1

        # Update metrics
        self.metrics.patterns_enhanced += len(patterns)
        self.metrics.pattern_connections_discovered += connections.network_metrics.edges
        self.metrics.evolution_predictions_made += len(evolution.predictions)
        self.metrics.insights_crystallized += sum(len(w) for w in wisdom.insights.values())
        self.metrics.pattern_valuations += len(valuations.valuations)
        self.metrics.intelligence_level *= 1.001  # Gradual growth

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return EnhancementResult(
            pattern_connections=connections,
            evolution_predictions=evolution,
            crystallized_insights=wisdom,
            pattern_valuations=valuations,
            emergent_patterns=emergence,
            enhancement_score=enhancement_score,
            processing_time_ms=processing_time,
        )

    def _calculate_enhancement_score(
        self,
        connections: ConnectionResult,
        evolution: EvolutionResult,
        wisdom: WisdomResult,
        valuations: ValueResult,
        emergence: BreakthroughResult,
    ) -> float:
        """Calculate overall enhancement score from all dimensions."""
        weights = {
            'connections': 0.2,
            'evolution': 0.2,
            'wisdom': 0.2,
            'valuations': 0.2,
            'emergence': 0.2,
        }

        score = (
            connections.score * weights['connections'] +
            evolution.score * weights['evolution'] +
            wisdom.score * weights['wisdom'] +
            valuations.score * weights['valuations'] +
            emergence.score * weights['emergence']
        )

        return score

    def _handle_high_enhancement(
        self,
        patterns: List[Any],
        score: float,
        context: Dict[str, Any],
    ) -> None:
        """Handle high-quality enhancement (breakthrough level)."""
        logger.info(f"High enhancement detected: score={score:.3f}")

        # Create breakthrough record
        breakthrough_id = hashlib.md5(
            f"enhancement_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        breakthrough = Breakthrough(
            breakthrough_id=breakthrough_id,
            pattern_id=f"batch_{len(patterns)}_patterns",
            emergence_score=score,
            key_factors=['high_enhancement'],
            timestamp=datetime.now().isoformat(),
            description=f"High-quality enhancement batch with {len(patterns)} patterns",
        )

        self.breakthroughs.append(breakthrough)

    def optimize_for_retrieval(
        self,
        patterns: List[Any],
        recent_queries: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Optimize patterns for retrieval/re-ranking.

        Args:
            patterns: List of Pattern objects
            recent_queries: Optional recent queries

        Returns:
            List of patterns enriched with re-ranking metadata
        """
        return self.optimizer.batch_optimize(patterns, recent_queries)

    def find_synergies(
        self,
        base_pattern: Any,
        candidates: List[Any],
        min_synergy: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find synergetic patterns.

        Args:
            base_pattern: Pattern to find synergies for
            candidates: Candidate patterns
            min_synergy: Minimum synergy score

        Returns:
            List of synergy information dicts
        """
        synergies = self.optimizer.find_synergetic_patterns(
            base_pattern, candidates, min_synergy
        )

        return [
            {
                'pattern_id': s.pattern_id,
                'synergy_score': s.synergy_score,
                'synergy_type': s.synergy_type,
                'combined_impact': s.combined_impact,
                'suggested_order': s.suggested_order,
            }
            for s in synergies
        ]

    def get_breakthroughs(
        self,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get recent breakthroughs.

        Args:
            limit: Max breakthroughs to return
            min_score: Minimum emergence score

        Returns:
            List of breakthrough dicts
        """
        filtered = [
            b for b in self.breakthroughs
            if b.emergence_score >= min_score
        ]

        filtered.sort(key=lambda b: b.timestamp, reverse=True)

        return [
            {
                'breakthrough_id': b.breakthrough_id,
                'pattern_id': b.pattern_id,
                'emergence_score': b.emergence_score,
                'key_factors': b.key_factors,
                'timestamp': b.timestamp,
                'description': b.description,
            }
            for b in filtered[:limit]
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get SIMBA metrics."""
        return {
            'patterns_enhanced': self.metrics.patterns_enhanced,
            'pattern_connections_discovered': self.metrics.pattern_connections_discovered,
            'evolution_predictions_made': self.metrics.evolution_predictions_made,
            'insights_crystallized': self.metrics.insights_crystallized,
            'pattern_valuations': self.metrics.pattern_valuations,
            'breakthroughs_detected': self.metrics.breakthroughs_detected,
            'intelligence_level': self.metrics.intelligence_level,
            'total_breakthroughs_stored': len(self.breakthroughs),
        }

    def record_pattern_performance(
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
        self.optimizer.record_performance(pattern_id, success, score)

    def quick_enhance(
        self,
        patterns: List[Any],
    ) -> Dict[str, Any]:
        """
        Quick enhancement with just essential dimensions.

        Runs: connections, valuations, breakthroughs only.

        Args:
            patterns: List of Pattern objects

        Returns:
            Dict with essential enhancement results
        """
        connections = self.dimensions['context_weaving'].weave_connections(patterns)
        valuations = self.dimensions['value_creation'].assess_value(patterns)
        emergence = self.dimensions['emergence_guardian'].detect_breakthroughs(patterns)

        return {
            'connections': connections.score,
            'valuations': valuations.score,
            'emergence': emergence.score,
            'breakthroughs': len(emergence.breakthrough_candidates),
            'high_value_patterns': [
                pid for pid, v in valuations.valuations.items()
                if v.value_category in ('exceptional', 'high')
            ],
        }

    def get_pattern_decay(self, pattern: Any) -> float:
        """Get current decay factor for a pattern."""
        return self.optimizer.calculate_decay(pattern)

    def get_pattern_freshness(self, pattern: Any) -> float:
        """Get freshness score for a pattern."""
        return self.optimizer.calculate_freshness(pattern)
