"""
Pattern Value Assessor - Evaluates economic and strategic value.

Part of the SIMBA system. Adapted from Distillation Engine.

Features:
- Multi-component value calculation
- Efficiency value (performance gains, resource savings)
- Reusability value (cross-domain applicability)
- Innovation value (novelty, breakthrough potential)
- Strategic value (alignment, competitive advantage)
- Risk mitigation value (reliability, safety)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ValueComponents:
    """Breakdown of value components."""
    efficiency_value: float
    reusability_value: float
    innovation_value: float
    strategic_value: float
    risk_mitigation_value: float


@dataclass
class PatternValuation:
    """Complete valuation for a pattern."""
    pattern_id: str
    total_value: float
    components: ValueComponents
    value_category: str  # exceptional, high, moderate, low, minimal


@dataclass
class ValueResult:
    """Result of value assessment."""
    valuations: Dict[str, PatternValuation]
    score: float  # Average value score
    value_distribution: Dict[str, int]  # Category counts


class PatternValueAssessor:
    """
    Evaluates the economic and strategic value of patterns.

    Calculates value across 5 dimensions:
    - Efficiency: Performance gains and resource savings
    - Reusability: How broadly the pattern can be applied
    - Innovation: Novelty and breakthrough potential
    - Strategic: Alignment with goals and competitive advantage
    - Risk Mitigation: Reliability and safety improvements

    Example:
        assessor = PatternValueAssessor()
        result = assessor.assess_value(patterns)

        for pattern_id, valuation in result.valuations.items():
            if valuation.value_category == 'exceptional':
                print(f"High-value pattern: {pattern_id}")
    """

    # Value category thresholds
    CATEGORY_THRESHOLDS = {
        'exceptional': 4.0,
        'high': 3.0,
        'moderate': 2.0,
        'low': 1.0,
        'minimal': 0.0,
    }

    def __init__(self):
        """Initialize the value assessor."""
        pass

    def assess_value(
        self,
        patterns: List[Any],
    ) -> ValueResult:
        """
        Assess value of all patterns.

        Args:
            patterns: List of Pattern objects

        Returns:
            ValueResult with valuations and distribution
        """
        valuations = {}

        for pattern in patterns:
            valuation = self._calculate_value(pattern)
            valuations[pattern.id] = valuation

        # Calculate average score
        values = [v.total_value for v in valuations.values()]
        score = sum(values) / len(values) if values else 0.0

        # Calculate distribution
        distribution = self._analyze_distribution(valuations)

        return ValueResult(
            valuations=valuations,
            score=score,
            value_distribution=distribution,
        )

    def _calculate_value(self, pattern: Any) -> PatternValuation:
        """Calculate complete value for a pattern."""
        components = ValueComponents(
            efficiency_value=self._calc_efficiency_value(pattern),
            reusability_value=self._calc_reusability_value(pattern),
            innovation_value=self._calc_innovation_value(pattern),
            strategic_value=self._calc_strategic_value(pattern),
            risk_mitigation_value=self._calc_risk_mitigation_value(pattern),
        )

        # Sum all components
        total_value = (
            components.efficiency_value +
            components.reusability_value +
            components.innovation_value +
            components.strategic_value +
            components.risk_mitigation_value
        )

        # Determine category
        category = self._categorize_value(total_value)

        return PatternValuation(
            pattern_id=pattern.id,
            total_value=total_value,
            components=components,
            value_category=category,
        )

    def _calc_efficiency_value(self, pattern: Any) -> float:
        """Calculate efficiency value (performance/resource savings)."""
        value = 0.0

        # Success rate indicates efficiency
        success_rate = getattr(pattern, 'success_rate', 0.5)
        value += success_rate * 0.4

        # High confidence = efficient predictions
        confidence = getattr(pattern, 'confidence', 0.5)
        if confidence > 0.7:
            value += 0.3

        # Stability indicates consistent efficiency
        stability = getattr(pattern, 'stability', 0.5)
        value += stability * 0.3

        return min(1.0, value)

    def _calc_reusability_value(self, pattern: Any) -> float:
        """Calculate reusability value (cross-domain applicability)."""
        value = 0.0

        # Application count shows reusability
        app_count = getattr(pattern, 'application_count', 0)
        value += min(0.4, app_count * 0.04)  # Cap at 10 applications

        # Multiple evidence sources = more reusable
        evidence = getattr(pattern, 'evidence', [])
        source_types = {e.source_type for e in evidence if hasattr(e, 'source_type')}
        value += min(0.3, len(source_types) * 0.1)

        # Keyword richness indicates broad applicability
        keywords = getattr(pattern, 'keywords', []) or []
        value += min(0.3, len(keywords) * 0.02)

        return min(1.0, value)

    def _calc_innovation_value(self, pattern: Any) -> float:
        """Calculate innovation value (novelty and breakthrough potential)."""
        value = 0.0

        # Synthesized patterns are innovative
        is_synthesized = getattr(pattern, 'is_synthesized', False)
        if is_synthesized:
            value += 0.3

        # Newer patterns have novelty
        created_at = getattr(pattern, 'created_at', None)
        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            age_days = (datetime.now() - created_at).days
            if age_days < 7:
                value += 0.4
            elif age_days < 30:
                value += 0.2

        # Few relationships = potentially novel territory
        relationships = getattr(pattern, 'relationships', [])
        if len(relationships) <= 2:
            value += 0.2

        # High confidence + low evidence = breakthrough potential
        confidence = getattr(pattern, 'confidence', 0.5)
        evidence = getattr(pattern, 'evidence', [])
        if confidence > 0.7 and len(evidence) < 3:
            value += 0.1

        return min(1.0, value)

    def _calc_strategic_value(self, pattern: Any) -> float:
        """Calculate strategic value (alignment and competitive advantage)."""
        value = 0.0

        # High importance = strategic
        importance = getattr(pattern, 'importance', 0.5)
        if hasattr(pattern, 'evidence') and pattern.evidence:
            # Get max relevance from evidence
            relevances = [e.relevance for e in pattern.evidence if hasattr(e, 'relevance')]
            if relevances:
                importance = max(importance, max(relevances))

        value += importance * 0.5

        # Many relationships = central to system = strategic
        relationships = getattr(pattern, 'relationships', [])
        value += min(0.3, len(relationships) * 0.05)

        # High confidence = reliable strategic asset
        confidence = getattr(pattern, 'confidence', 0.5)
        if confidence > 0.7:
            value += 0.2

        return min(1.0, value)

    def _calc_risk_mitigation_value(self, pattern: Any) -> float:
        """Calculate risk mitigation value (reliability and safety)."""
        value = 0.0

        # High stability = reliable = low risk
        stability = getattr(pattern, 'stability', 0.5)
        value += stability * 0.4

        # Multiple evidence sources = validated = lower risk
        evidence = getattr(pattern, 'evidence', [])
        sources = {e.source_id for e in evidence if hasattr(e, 'source_id')}
        value += min(0.3, len(sources) * 0.1)

        # High success rate = proven = low risk
        success_rate = getattr(pattern, 'success_rate', 0.5)
        value += success_rate * 0.3

        return min(1.0, value)

    def _categorize_value(self, total_value: float) -> str:
        """Categorize the total value score."""
        if total_value >= self.CATEGORY_THRESHOLDS['exceptional']:
            return 'exceptional'
        elif total_value >= self.CATEGORY_THRESHOLDS['high']:
            return 'high'
        elif total_value >= self.CATEGORY_THRESHOLDS['moderate']:
            return 'moderate'
        elif total_value >= self.CATEGORY_THRESHOLDS['low']:
            return 'low'
        return 'minimal'

    def _analyze_distribution(
        self,
        valuations: Dict[str, PatternValuation],
    ) -> Dict[str, int]:
        """Analyze the distribution of value categories."""
        distribution = {
            'exceptional': 0,
            'high': 0,
            'moderate': 0,
            'low': 0,
            'minimal': 0,
        }

        for valuation in valuations.values():
            category = valuation.value_category
            distribution[category] = distribution.get(category, 0) + 1

        return distribution

    def predict_value_impact(
        self,
        pattern: Any,
        target_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict value impact when applying pattern in a context.

        Args:
            pattern: Pattern to evaluate
            target_context: Context where pattern will be applied

        Returns:
            Dict with predicted impacts
        """
        base_valuation = self._calculate_value(pattern)

        # Adjust for context fit
        context_fit = self._assess_context_fit(pattern, target_context)

        predicted_value = base_valuation.total_value * context_fit

        return {
            "base_value": base_valuation.total_value,
            "context_fit": context_fit,
            "predicted_value": predicted_value,
            "value_category": self._categorize_value(predicted_value),
            "risk_level": self._assess_risk_level(pattern, target_context),
            "recommended_action": self._recommend_action(predicted_value, context_fit),
        }

    def _assess_context_fit(
        self,
        pattern: Any,
        context: Dict[str, Any],
    ) -> float:
        """Assess how well a pattern fits a context."""
        fit = 0.5  # Base fit

        # Context domain match
        pattern_context = getattr(pattern, 'context', None)
        target_domain = context.get('domain')

        if pattern_context and target_domain:
            if pattern_context == target_domain:
                fit += 0.3
            elif pattern_context in str(target_domain):
                fit += 0.1

        # Keyword match with context requirements
        keywords = set(getattr(pattern, 'keywords', []) or [])
        requirements = set(context.get('requirements', []))

        if keywords and requirements:
            overlap = len(keywords & requirements)
            if overlap > 0:
                fit += min(0.2, overlap * 0.05)

        return min(1.0, fit)

    def _assess_risk_level(
        self,
        pattern: Any,
        context: Dict[str, Any],
    ) -> str:
        """Assess risk level of applying pattern in context."""
        stability = getattr(pattern, 'stability', 0.5)
        confidence = getattr(pattern, 'confidence', 0.5)
        context_fit = self._assess_context_fit(pattern, context)

        risk_score = (
            (1 - stability) * 0.4 +
            (1 - confidence) * 0.3 +
            (1 - context_fit) * 0.3
        )

        if risk_score > 0.7:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        return "low"

    def _recommend_action(
        self,
        predicted_value: float,
        context_fit: float,
    ) -> str:
        """Recommend action based on value and fit."""
        if predicted_value >= 3.0 and context_fit >= 0.7:
            return "apply_immediately"
        elif predicted_value >= 2.0 and context_fit >= 0.5:
            return "apply_with_monitoring"
        elif predicted_value >= 1.0:
            return "test_first"
        return "gather_more_evidence"
