"""
Pattern Breakthrough Detector - Identifies emergent patterns.

Part of the SIMBA system. Adapted from Distillation Engine.

Features:
- Emergence detection
- Novelty assessment
- Unexpected effectiveness detection
- Cross-domain applicability assessment
- Synergistic effects detection
- Breakthrough potential scoring
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import hashlib


@dataclass
class EmergenceFactors:
    """Factors contributing to pattern emergence."""
    novelty: float
    unexpected_effectiveness: float
    cross_domain_applicability: float
    synergistic_effects: float


@dataclass
class EmergenceAnalysis:
    """Analysis of a pattern's emergence."""
    is_emergent: bool
    emergence_score: float
    factors: EmergenceFactors
    breakthrough_potential: bool


@dataclass
class Breakthrough:
    """A detected breakthrough pattern."""
    breakthrough_id: str
    pattern_id: str
    emergence_score: float
    key_factors: List[str]
    timestamp: str
    description: str


@dataclass
class BreakthroughResult:
    """Result of breakthrough detection."""
    emergence: Dict[str, EmergenceAnalysis]
    score: float
    breakthrough_candidates: List[Breakthrough]


class PatternBreakthroughDetector:
    """
    Detects emergent and breakthrough patterns.

    Identifies patterns that represent:
    - Novel discoveries
    - Unexpected high performance
    - Cross-domain breakthroughs
    - Synergistic combinations

    Example:
        detector = PatternBreakthroughDetector()
        result = detector.detect_breakthroughs(patterns)

        for breakthrough in result.breakthrough_candidates:
            print(f"Breakthrough: {breakthrough.pattern_id}")
            print(f"  Score: {breakthrough.emergence_score}")
            print(f"  Factors: {', '.join(breakthrough.key_factors)}")
    """

    EMERGENCE_THRESHOLD = 0.7
    BREAKTHROUGH_THRESHOLD = 0.8

    def __init__(self):
        """Initialize the breakthrough detector."""
        self.detected_breakthroughs: List[Breakthrough] = []

    def detect_breakthroughs(
        self,
        patterns: List[Any],
    ) -> BreakthroughResult:
        """
        Detect breakthroughs among patterns.

        Args:
            patterns: List of Pattern objects

        Returns:
            BreakthroughResult with emergence analysis and breakthroughs
        """
        emergence: Dict[str, EmergenceAnalysis] = {}
        breakthrough_candidates: List[Breakthrough] = []

        for pattern in patterns:
            analysis = self._analyze_emergence(pattern)
            emergence[pattern.id] = analysis

            if analysis.is_emergent:
                # Check for breakthrough
                if analysis.breakthrough_potential:
                    breakthrough = self._create_breakthrough(pattern, analysis)
                    breakthrough_candidates.append(breakthrough)
                    self.detected_breakthroughs.append(breakthrough)

        # Calculate overall score
        scores = [a.emergence_score for a in emergence.values()]
        score = sum(scores) / len(scores) if scores else 0.0

        return BreakthroughResult(
            emergence=emergence,
            score=score,
            breakthrough_candidates=breakthrough_candidates,
        )

    def _analyze_emergence(self, pattern: Any) -> EmergenceAnalysis:
        """Analyze emergence potential of a pattern."""
        factors = EmergenceFactors(
            novelty=self._assess_novelty(pattern),
            unexpected_effectiveness=self._check_unexpected_effectiveness(pattern),
            cross_domain_applicability=self._assess_cross_domain(pattern),
            synergistic_effects=self._detect_synergies(pattern),
        )

        # Calculate emergence score
        emergence_score = self._calc_emergence_score(factors)

        return EmergenceAnalysis(
            is_emergent=emergence_score > self.EMERGENCE_THRESHOLD,
            emergence_score=emergence_score,
            factors=factors,
            breakthrough_potential=emergence_score > self.BREAKTHROUGH_THRESHOLD,
        )

    def _assess_novelty(self, pattern: Any) -> float:
        """Assess pattern novelty."""
        novelty = 0.0

        # Age factor - newer patterns are more novel
        created_at = getattr(pattern, 'created_at', None)
        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            age_days = (datetime.now() - created_at).days

            if age_days < 1:
                novelty += 0.5
            elif age_days < 7:
                novelty += 0.3
            elif age_days < 30:
                novelty += 0.1

        # Uniqueness - fewer similar patterns = more novel
        relationships = getattr(pattern, 'relationships', [])
        similar_count = sum(
            1 for r in relationships
            if hasattr(r, 'relationship_type') and 'similar' in str(r.relationship_type).lower()
        )

        if similar_count == 0:
            novelty += 0.3
        elif similar_count == 1:
            novelty += 0.1

        # Synthesized patterns have novelty
        if getattr(pattern, 'is_synthesized', False):
            novelty += 0.2

        return min(1.0, novelty)

    def _check_unexpected_effectiveness(self, pattern: Any) -> float:
        """Check for unexpected effectiveness."""
        effectiveness = 0.0

        # High success rate with low initial confidence = unexpected
        success_rate = getattr(pattern, 'success_rate', None)
        confidence = getattr(pattern, 'confidence', 0.5)

        if success_rate is not None:
            if success_rate > 0.8 and confidence < 0.7:
                effectiveness += 0.5
            elif success_rate > confidence:
                effectiveness += (success_rate - confidence) * 0.5

        # Many applications with good results = surprisingly effective
        app_count = getattr(pattern, 'application_count', 0)
        if app_count > 5 and (success_rate or 0) > 0.7:
            effectiveness += 0.3

        # High stability despite being new = unexpected
        stability = getattr(pattern, 'stability', 0.5)
        created_at = getattr(pattern, 'created_at', None)

        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            age_days = (datetime.now() - created_at).days

            if age_days < 14 and stability > 0.7:
                effectiveness += 0.2

        return min(1.0, effectiveness)

    def _assess_cross_domain(self, pattern: Any) -> float:
        """Assess cross-domain applicability."""
        cross_domain = 0.0

        # Evidence from multiple source types
        evidence = getattr(pattern, 'evidence', [])
        source_types = {e.source_type for e in evidence if hasattr(e, 'source_type')}

        if len(source_types) >= 3:
            cross_domain += 0.4
        elif len(source_types) >= 2:
            cross_domain += 0.2

        # Multiple contexts/domains mentioned
        keywords = getattr(pattern, 'keywords', []) or []
        domain_keywords = {'general', 'universal', 'cross-domain', 'multi', 'various'}
        domain_matches = sum(1 for kw in keywords if any(d in kw.lower() for d in domain_keywords))

        if domain_matches > 0:
            cross_domain += min(0.3, domain_matches * 0.1)

        # Relationships to patterns in different contexts
        relationships = getattr(pattern, 'relationships', [])
        # Assume relationships to different contexts exist if we have many relationships
        if len(relationships) >= 5:
            cross_domain += 0.3

        return min(1.0, cross_domain)

    def _detect_synergies(self, pattern: Any) -> float:
        """Detect synergistic effects."""
        synergy = 0.0

        # Patterns that are part of synthesis have synergy
        if getattr(pattern, 'is_synthesized', False):
            synergy += 0.4

        # Many supporting relationships = synergistic
        relationships = getattr(pattern, 'relationships', [])
        supports_count = sum(
            1 for r in relationships
            if hasattr(r, 'relationship_type') and 'support' in str(r.relationship_type).lower()
        )

        if supports_count >= 3:
            synergy += 0.3
        elif supports_count >= 1:
            synergy += 0.1

        # High confidence with multiple evidence sources = synergistic validation
        evidence = getattr(pattern, 'evidence', [])
        sources = {e.source_id for e in evidence if hasattr(e, 'source_id')}
        confidence = getattr(pattern, 'confidence', 0.5)

        if len(sources) >= 2 and confidence > 0.7:
            synergy += 0.3

        return min(1.0, synergy)

    def _calc_emergence_score(self, factors: EmergenceFactors) -> float:
        """Calculate overall emergence score from factors."""
        weights = {
            'novelty': 0.3,
            'unexpected_effectiveness': 0.3,
            'cross_domain_applicability': 0.2,
            'synergistic_effects': 0.2,
        }

        score = (
            factors.novelty * weights['novelty'] +
            factors.unexpected_effectiveness * weights['unexpected_effectiveness'] +
            factors.cross_domain_applicability * weights['cross_domain_applicability'] +
            factors.synergistic_effects * weights['synergistic_effects']
        )

        return score

    def _create_breakthrough(
        self,
        pattern: Any,
        analysis: EmergenceAnalysis,
    ) -> Breakthrough:
        """Create a breakthrough record."""
        # Identify key factors
        key_factors = []
        if analysis.factors.novelty > 0.7:
            key_factors.append("novelty")
        if analysis.factors.unexpected_effectiveness > 0.7:
            key_factors.append("unexpected_effectiveness")
        if analysis.factors.cross_domain_applicability > 0.7:
            key_factors.append("cross_domain")
        if analysis.factors.synergistic_effects > 0.7:
            key_factors.append("synergy")

        # Generate breakthrough ID
        breakthrough_id = hashlib.md5(
            f"{pattern.id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Create description
        pattern_name = getattr(pattern, 'name', pattern.id[:20])
        description = f"Breakthrough detected in '{pattern_name}' with factors: {', '.join(key_factors)}"

        return Breakthrough(
            breakthrough_id=breakthrough_id,
            pattern_id=pattern.id,
            emergence_score=analysis.emergence_score,
            key_factors=key_factors,
            timestamp=datetime.now().isoformat(),
            description=description,
        )

    def get_recent_breakthroughs(
        self,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Breakthrough]:
        """
        Get recent breakthroughs.

        Args:
            limit: Max breakthroughs to return
            min_score: Minimum emergence score

        Returns:
            List of recent Breakthrough objects
        """
        filtered = [
            b for b in self.detected_breakthroughs
            if b.emergence_score >= min_score
        ]

        # Sort by timestamp descending
        filtered.sort(key=lambda b: b.timestamp, reverse=True)

        return filtered[:limit]

    def detect_emergent_patterns(
        self,
        patterns: List[Any],
    ) -> Dict[str, EmergenceAnalysis]:
        """
        Simplified interface - just detect emergent patterns.

        Args:
            patterns: List of Pattern objects

        Returns:
            Dict mapping pattern_id to EmergenceAnalysis for emergent patterns only
        """
        result = self.detect_breakthroughs(patterns)

        return {
            pid: analysis
            for pid, analysis in result.emergence.items()
            if analysis.is_emergent
        }
