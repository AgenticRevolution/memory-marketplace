"""
Pattern Wisdom Extractor - Crystallizes deep insights from patterns.

Part of the SIMBA system. Adapted from Distillation Engine.

Features:
- Application wisdom extraction
- Optimization wisdom extraction
- Contextual wisdom extraction
- Meta-wisdom across patterns
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Wisdom:
    """A piece of wisdom extracted from patterns."""
    wisdom_type: str  # 'application', 'optimization', 'contextual', 'pitfall', 'meta'
    insight: str
    confidence: float
    source_pattern_id: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class WisdomResult:
    """Result of wisdom extraction."""
    insights: Dict[str, List[Wisdom]]  # pattern_id -> list of wisdom
    score: float
    meta_insights: List[Wisdom]


class PatternWisdomExtractor:
    """
    Crystallizes deep insights and wisdom from patterns.

    Extracts:
    - Application wisdom (how to use effectively)
    - Optimization wisdom (how to improve results)
    - Contextual wisdom (when/where works best)
    - Pitfall wisdom (what to avoid)
    - Meta-wisdom (patterns across patterns)

    Example:
        extractor = PatternWisdomExtractor()
        result = extractor.extract_wisdom(patterns)

        for pattern_id, wisdoms in result.insights.items():
            for w in wisdoms:
                print(f"[{w.wisdom_type}] {w.insight}")
    """

    def __init__(self):
        """Initialize the wisdom extractor."""
        pass

    def extract_wisdom(
        self,
        patterns: List[Any],
    ) -> WisdomResult:
        """
        Extract wisdom from all patterns.

        Args:
            patterns: List of Pattern objects

        Returns:
            WisdomResult with insights and meta-insights
        """
        insights: Dict[str, List[Wisdom]] = {}

        for pattern in patterns:
            pattern_wisdom = self._extract_pattern_wisdom(pattern)
            insights[pattern.id] = pattern_wisdom

        # Calculate score
        score = self._calc_wisdom_score(insights)

        # Extract meta-wisdom
        meta_insights = self._extract_meta_wisdom(insights)

        return WisdomResult(
            insights=insights,
            score=score,
            meta_insights=meta_insights,
        )

    def _extract_pattern_wisdom(self, pattern: Any) -> List[Wisdom]:
        """Extract all wisdom from a single pattern."""
        wisdom_list = []

        wisdom_list.extend(self._extract_application_wisdom(pattern))
        wisdom_list.extend(self._extract_optimization_wisdom(pattern))
        wisdom_list.extend(self._extract_contextual_wisdom(pattern))
        wisdom_list.extend(self._extract_pitfall_wisdom(pattern))

        return wisdom_list

    def _extract_application_wisdom(self, pattern: Any) -> List[Wisdom]:
        """Extract wisdom about how to apply the pattern."""
        wisdom = []

        # Success rate wisdom
        success_rate = getattr(pattern, 'success_rate', None)
        if success_rate is not None:
            if success_rate > 0.8:
                wisdom.append(Wisdom(
                    wisdom_type="application",
                    insight=f"High reliability pattern with {success_rate:.0%} success rate",
                    confidence=0.9,
                    source_pattern_id=pattern.id,
                ))
            elif success_rate < 0.4:
                wisdom.append(Wisdom(
                    wisdom_type="pitfall",
                    insight=f"Low success rate ({success_rate:.0%}) - use with caution",
                    confidence=0.85,
                    source_pattern_id=pattern.id,
                ))

        # Application count wisdom
        app_count = getattr(pattern, 'application_count', 0)
        if app_count > 10:
            wisdom.append(Wisdom(
                wisdom_type="application",
                insight=f"Well-tested pattern with {app_count} applications",
                confidence=0.8,
                source_pattern_id=pattern.id,
            ))

        # Evidence-based application wisdom
        evidence = getattr(pattern, 'evidence', [])
        if len(evidence) >= 3:
            sources = {e.source_id for e in evidence if hasattr(e, 'source_id')}
            if len(sources) >= 2:
                wisdom.append(Wisdom(
                    wisdom_type="application",
                    insight=f"Cross-validated across {len(sources)} sources - high reliability",
                    confidence=0.85,
                    source_pattern_id=pattern.id,
                ))

        return wisdom

    def _extract_optimization_wisdom(self, pattern: Any) -> List[Wisdom]:
        """Extract wisdom about optimizing pattern usage."""
        wisdom = []

        # Confidence-based optimization
        confidence = getattr(pattern, 'confidence', 0.5)
        if confidence > 0.8:
            wisdom.append(Wisdom(
                wisdom_type="optimization",
                insight="High confidence pattern - suitable for critical applications",
                confidence=0.85,
                source_pattern_id=pattern.id,
            ))
        elif confidence < 0.4:
            wisdom.append(Wisdom(
                wisdom_type="optimization",
                insight="Low confidence - consider gathering more evidence before relying on this pattern",
                confidence=0.75,
                source_pattern_id=pattern.id,
            ))

        # Stability-based optimization
        stability = getattr(pattern, 'stability', 0.5)
        if stability > 0.7:
            wisdom.append(Wisdom(
                wisdom_type="optimization",
                insight="Stable pattern - results are consistent across applications",
                confidence=0.8,
                source_pattern_id=pattern.id,
            ))

        # Keyword-based optimization hints
        keywords = getattr(pattern, 'keywords', []) or []
        optimization_keywords = {'performance', 'efficiency', 'speed', 'optimization'}
        matching_keywords = [kw for kw in keywords if kw.lower() in optimization_keywords]

        if matching_keywords:
            wisdom.append(Wisdom(
                wisdom_type="optimization",
                insight=f"Pattern focuses on {', '.join(matching_keywords)} - good for improving metrics",
                confidence=0.7,
                source_pattern_id=pattern.id,
            ))

        return wisdom

    def _extract_contextual_wisdom(self, pattern: Any) -> List[Wisdom]:
        """Extract wisdom about context and situational usage."""
        wisdom = []

        # Context-specific wisdom
        context = getattr(pattern, 'context', None)
        if context:
            wisdom.append(Wisdom(
                wisdom_type="contextual",
                insight=f"Most effective in '{context}' context",
                confidence=0.75,
                source_pattern_id=pattern.id,
            ))

        # Pattern type contextual wisdom
        pattern_type = getattr(pattern, 'pattern_type', None)
        if pattern_type:
            type_contexts = {
                'linguistic': "Works best with text-heavy content",
                'structural': "Best for organizing and categorizing",
                'semantic': "Effective for meaning-based matching",
                'behavioral': "Useful for predicting user actions",
                'temporal': "Best for time-series or sequence data",
            }

            type_str = str(pattern_type.value if hasattr(pattern_type, 'value') else pattern_type).lower()
            if type_str in type_contexts:
                wisdom.append(Wisdom(
                    wisdom_type="contextual",
                    insight=type_contexts[type_str],
                    confidence=0.7,
                    source_pattern_id=pattern.id,
                ))

        # Evidence source contextual wisdom
        evidence = getattr(pattern, 'evidence', [])
        source_types = {e.source_type for e in evidence if hasattr(e, 'source_type')}

        if 'query' in source_types:
            wisdom.append(Wisdom(
                wisdom_type="contextual",
                insight="Pattern learned from query behavior - effective for search optimization",
                confidence=0.7,
                source_pattern_id=pattern.id,
            ))

        return wisdom

    def _extract_pitfall_wisdom(self, pattern: Any) -> List[Wisdom]:
        """Extract wisdom about potential pitfalls and warnings."""
        wisdom = []

        # Low stability warning
        stability = getattr(pattern, 'stability', 0.5)
        if stability < 0.3:
            wisdom.append(Wisdom(
                wisdom_type="pitfall",
                insight="Unstable pattern - results may vary significantly",
                confidence=0.8,
                source_pattern_id=pattern.id,
            ))

        # Limited evidence warning
        evidence = getattr(pattern, 'evidence', [])
        if len(evidence) < 2:
            wisdom.append(Wisdom(
                wisdom_type="pitfall",
                insight="Limited evidence - pattern may not generalize well",
                confidence=0.75,
                source_pattern_id=pattern.id,
            ))

        # Synthesized pattern warning
        is_synthesized = getattr(pattern, 'is_synthesized', False)
        if is_synthesized:
            wisdom.append(Wisdom(
                wisdom_type="pitfall",
                insight="Synthesized pattern - verify accuracy against source patterns",
                confidence=0.7,
                source_pattern_id=pattern.id,
            ))

        return wisdom

    def _calc_wisdom_score(
        self,
        insights: Dict[str, List[Wisdom]],
    ) -> float:
        """Calculate overall wisdom extraction score."""
        total_wisdom = 0
        pattern_count = 0

        for wisdoms in insights.values():
            total_wisdom += len(wisdoms)
            pattern_count += 1

        if pattern_count == 0:
            return 0.0

        # Normalize by expected insights per pattern (3-5)
        expected_per_pattern = 4
        avg_wisdom = total_wisdom / pattern_count

        return min(1.0, avg_wisdom / expected_per_pattern)

    def _extract_meta_wisdom(
        self,
        insights: Dict[str, List[Wisdom]],
    ) -> List[Wisdom]:
        """Extract meta-wisdom from patterns across all patterns."""
        meta_wisdom = []

        # Collect all wisdom
        all_wisdom = [w for wisdoms in insights.values() for w in wisdoms]

        # Find common themes
        type_counts: Dict[str, int] = {}
        for w in all_wisdom:
            type_counts[w.wisdom_type] = type_counts.get(w.wisdom_type, 0) + 1

        # Meta-insight about prevalent themes
        total = len(all_wisdom)
        for wisdom_type, count in type_counts.items():
            if count >= 3 and total > 0:
                percentage = count / total * 100
                meta_wisdom.append(Wisdom(
                    wisdom_type="meta",
                    insight=f"{wisdom_type.title()} insights are prevalent ({count} instances, {percentage:.0f}%)",
                    confidence=0.8,
                ))

        # Meta-insight about pattern health
        pitfall_count = type_counts.get('pitfall', 0)
        if pitfall_count > len(insights) * 0.5:
            meta_wisdom.append(Wisdom(
                wisdom_type="meta",
                insight="Many patterns have warnings - consider knowledge base cleanup",
                confidence=0.85,
            ))
        elif pitfall_count < len(insights) * 0.1:
            meta_wisdom.append(Wisdom(
                wisdom_type="meta",
                insight="Knowledge base is healthy - few warning patterns",
                confidence=0.8,
            ))

        # Meta-insight about confidence
        high_conf_count = sum(
            1 for wisdoms in insights.values()
            for w in wisdoms
            if w.confidence > 0.8
        )

        if high_conf_count > total * 0.6:
            meta_wisdom.append(Wisdom(
                wisdom_type="meta",
                insight="High overall confidence in extracted wisdom",
                confidence=0.9,
            ))

        return meta_wisdom

    def crystallize_meta_patterns(
        self,
        meta_patterns: List[Dict[str, Any]],
    ) -> List[Wisdom]:
        """
        Crystallize wisdom from meta-pattern analysis.

        Args:
            meta_patterns: Results from meta-pattern extraction

        Returns:
            List of wisdom about meta-patterns
        """
        wisdom = []

        for mp in meta_patterns:
            # Cross-pattern insights
            cross_patterns = mp.get('cross_patterns', [])
            if len(cross_patterns) > 2:
                wisdom.append(Wisdom(
                    wisdom_type="meta",
                    insight=f"Collection has {len(cross_patterns)} cross-cutting patterns",
                    confidence=0.75,
                ))

            # Higher-order pattern insights
            higher_order = mp.get('higher_order', [])
            if higher_order:
                wisdom.append(Wisdom(
                    wisdom_type="meta",
                    insight="Abstract patterns detected - system is developing deep understanding",
                    confidence=0.8,
                ))

            # Emergence score insight
            emergence = mp.get('emergence_score', 0)
            if emergence > 0.7:
                wisdom.append(Wisdom(
                    wisdom_type="meta",
                    insight="High emergence score - novel patterns forming",
                    confidence=0.85,
                ))

        return wisdom
