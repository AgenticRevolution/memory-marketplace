"""
Pattern Evolution Predictor - Predicts how patterns will evolve.

Part of the SIMBA system. Adapted from Distillation Engine.

Features:
- Next iteration prediction
- Optimization potential calculation
- Stability assessment
- Evolution trajectory projection
- Maturity timeline estimation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EvolutionPrediction:
    """Prediction for a pattern's evolution."""
    pattern_id: str
    next_iteration: Dict[str, Any]
    optimization_potential: float
    stability_score: float
    evolution_trajectory: Dict[str, Any]


@dataclass
class EvolutionResult:
    """Result of evolution prediction analysis."""
    predictions: Dict[str, EvolutionPrediction]
    score: float  # Overall prediction confidence
    recommendations: List[Dict[str, Any]]


class PatternEvolutionPredictor:
    """
    Predicts how patterns will evolve over time.

    Provides:
    - Next iteration predictions (confidence, applicability, complexity changes)
    - Optimization potential assessment
    - Stability scoring based on history
    - Evolution trajectory (maturity timeline, specialization path)

    Example:
        predictor = PatternEvolutionPredictor()
        result = predictor.predict_evolution(patterns)

        for pattern_id, prediction in result.predictions.items():
            if prediction.optimization_potential > 0.7:
                print(f"Pattern {pattern_id} has high optimization potential")
    """

    def __init__(self):
        """Initialize the evolution predictor."""
        self.prediction_history: Dict[str, List[Dict]] = {}

    def predict_evolution(
        self,
        patterns: List[Any],
    ) -> EvolutionResult:
        """
        Predict evolution for all patterns.

        Args:
            patterns: List of Pattern objects

        Returns:
            EvolutionResult with predictions and recommendations
        """
        predictions = {}

        for pattern in patterns:
            prediction = EvolutionPrediction(
                pattern_id=pattern.id,
                next_iteration=self._predict_next_iteration(pattern),
                optimization_potential=self._calc_optimization_potential(pattern),
                stability_score=self._assess_stability(pattern),
                evolution_trajectory=self._project_trajectory(pattern),
            )
            predictions[pattern.id] = prediction

        # Calculate overall score
        score = self._calc_prediction_confidence(predictions)

        # Generate recommendations
        recommendations = self._generate_recommendations(predictions)

        return EvolutionResult(
            predictions=predictions,
            score=score,
            recommendations=recommendations,
        )

    def _predict_next_iteration(self, pattern: Any) -> Dict[str, Any]:
        """Predict the next iteration of a pattern."""
        return {
            "confidence_change": self._predict_confidence_change(pattern),
            "applicability_expansion": self._predict_applicability_expansion(pattern),
            "complexity_evolution": self._predict_complexity_evolution(pattern),
            "estimated_timeframe": self._estimate_timeframe(pattern),
        }

    def _predict_confidence_change(self, pattern: Any) -> float:
        """Predict how confidence will change."""
        current = getattr(pattern, 'confidence', 0.5)

        # Check for evidence growth trend
        evidence = getattr(pattern, 'evidence', [])
        if len(evidence) > 2:
            # More evidence = confidence likely to increase
            trend = min(0.1, len(evidence) * 0.01)
        else:
            trend = 0.0

        # Patterns with high application success trend up
        success_rate = getattr(pattern, 'success_rate', 0.5)
        if success_rate > 0.7:
            trend += 0.05
        elif success_rate < 0.3:
            trend -= 0.05

        return min(1.0, max(0.0, current + trend))

    def _predict_applicability_expansion(self, pattern: Any) -> Dict[str, Any]:
        """Predict how broadly the pattern might be applied."""
        current_contexts = []

        # Get current application contexts
        if hasattr(pattern, 'context') and pattern.context:
            current_contexts = [pattern.context] if isinstance(pattern.context, str) else list(pattern.context)

        # Potential new domains based on pattern type
        potential_domains = self._suggest_new_domains(pattern)

        expansion_score = 1 - (len(current_contexts) / max(10, len(current_contexts) + 5))

        return {
            "current_contexts": current_contexts,
            "potential_domains": potential_domains,
            "expansion_score": expansion_score,
        }

    def _predict_complexity_evolution(self, pattern: Any) -> float:
        """Predict how complexity will evolve."""
        # Estimate current complexity from keywords and evidence
        keywords = getattr(pattern, 'keywords', []) or []
        evidence = getattr(pattern, 'evidence', []) or []

        current_complexity = min(1.0, (len(keywords) + len(evidence)) / 20)

        # Patterns tend toward optimal complexity (0.4-0.6)
        if current_complexity < 0.3:
            return current_complexity + 0.1
        elif current_complexity > 0.7:
            return current_complexity - 0.1

        return current_complexity

    def _estimate_timeframe(self, pattern: Any) -> str:
        """Estimate when pattern will reach next maturity level."""
        created_at = getattr(pattern, 'created_at', None)

        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            age_days = (datetime.now() - created_at).days

            if age_days < 7:
                return "2-3 weeks"
            elif age_days < 30:
                return "1-2 weeks"
            else:
                return "< 1 week"

        return "unknown"

    def _calc_optimization_potential(self, pattern: Any) -> float:
        """Calculate how much the pattern can be improved."""
        potential = 0.0

        # Low confidence = high optimization potential
        confidence = getattr(pattern, 'confidence', 0.5)
        potential += (1 - confidence) * 0.4

        # Low stability = room for improvement
        stability = getattr(pattern, 'stability', 0.5)
        potential += (1 - stability) * 0.3

        # Few evidence sources = can gather more
        evidence = getattr(pattern, 'evidence', [])
        if len(evidence) < 5:
            potential += 0.3 * (1 - len(evidence) / 5)

        return min(1.0, potential)

    def _assess_stability(self, pattern: Any) -> float:
        """Assess pattern stability over time."""
        # Check for confidence/stability attributes
        stability = getattr(pattern, 'stability', None)
        if stability is not None:
            return stability

        # Estimate from evidence
        evidence = getattr(pattern, 'evidence', [])
        if not evidence:
            return 0.5

        # More evidence from different sources = more stable
        sources = {e.source_id for e in evidence if hasattr(e, 'source_id')}
        source_diversity = min(1.0, len(sources) / 3)

        # Higher average confidence in evidence = more stable
        confidences = [e.confidence for e in evidence if hasattr(e, 'confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return 0.5 * source_diversity + 0.5 * avg_confidence

    def _project_trajectory(self, pattern: Any) -> Dict[str, Any]:
        """Project the future evolution path of a pattern."""
        return {
            "maturity_timeline": self._estimate_maturity_timeline(pattern),
            "specialization_path": self._predict_specialization(pattern),
            "generalization_potential": self._assess_generalization(pattern),
        }

    def _estimate_maturity_timeline(self, pattern: Any) -> Dict[str, Any]:
        """Estimate timeline to pattern maturity."""
        stability = self._assess_stability(pattern)
        confidence = getattr(pattern, 'confidence', 0.5)

        maturity_score = (stability + confidence) / 2

        if maturity_score > 0.8:
            stage = "mature"
            time_to_next = "stable"
        elif maturity_score > 0.6:
            stage = "established"
            time_to_next = "1-2 weeks"
        elif maturity_score > 0.4:
            stage = "developing"
            time_to_next = "2-4 weeks"
        else:
            stage = "emerging"
            time_to_next = "1-2 months"

        return {
            "current_stage": stage,
            "maturity_score": maturity_score,
            "time_to_next_stage": time_to_next,
        }

    def _predict_specialization(self, pattern: Any) -> Dict[str, Any]:
        """Predict pattern specialization trajectory."""
        keywords = getattr(pattern, 'keywords', []) or []

        # Patterns with many specific keywords tend to specialize
        specialization_tendency = min(1.0, len(keywords) / 15)

        likely_specializations = []
        if specialization_tendency > 0.5:
            # Extract likely specialization areas from keywords
            keyword_domains = {
                'accuracy', 'performance', 'efficiency', 'reliability',
                'scalability', 'security', 'usability'
            }
            for kw in keywords:
                if kw.lower() in keyword_domains:
                    likely_specializations.append(kw.lower())

        return {
            "specialization_score": specialization_tendency,
            "likely_specializations": likely_specializations or ["general"],
        }

    def _assess_generalization(self, pattern: Any) -> Dict[str, Any]:
        """Assess potential for pattern generalization."""
        # Patterns with high confidence and broad evidence can generalize
        confidence = getattr(pattern, 'confidence', 0.5)
        evidence = getattr(pattern, 'evidence', [])

        # Diverse source types = can generalize
        source_types = {e.source_type for e in evidence if hasattr(e, 'source_type')}

        generalization_score = 0.0
        if confidence > 0.7:
            generalization_score += 0.4
        if len(source_types) >= 2:
            generalization_score += 0.3
        if len(evidence) >= 3:
            generalization_score += 0.3

        abstraction_level = "high" if generalization_score > 0.7 else \
                           "medium" if generalization_score > 0.4 else "low"

        return {
            "generalization_potential": generalization_score,
            "abstraction_level": abstraction_level,
        }

    def _suggest_new_domains(self, pattern: Any) -> List[str]:
        """Suggest new domains where pattern might apply."""
        current_context = getattr(pattern, 'context', 'general')

        # Domain expansion map
        domain_expansions = {
            'tcm': ['wellness', 'integrative_medicine', 'nutrition'],
            'programming': ['automation', 'devops', 'data_science'],
            'general': ['specific_domain', 'cross_domain'],
        }

        suggestions = domain_expansions.get(current_context, ['cross_domain'])

        # Filter already covered
        return suggestions[:3]

    def _calc_prediction_confidence(
        self,
        predictions: Dict[str, EvolutionPrediction],
    ) -> float:
        """Calculate overall confidence in predictions."""
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions.values():
            # Higher stability = more confident prediction
            stability = pred.stability_score
            # Lower optimization potential = pattern is settled = more confident
            settled = 1 - pred.optimization_potential

            scores.append((stability + settled) / 2)

        return sum(scores) / len(scores)

    def _generate_recommendations(
        self,
        predictions: Dict[str, EvolutionPrediction],
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from predictions."""
        recommendations = []

        for pattern_id, pred in predictions.items():
            # High optimization potential
            if pred.optimization_potential > 0.7:
                recommendations.append({
                    "pattern_id": pattern_id,
                    "action": "optimize",
                    "reason": "High optimization potential detected",
                    "priority": "high",
                })

            # Low stability
            if pred.stability_score < 0.3:
                recommendations.append({
                    "pattern_id": pattern_id,
                    "action": "stabilize",
                    "reason": "Pattern showing high variability",
                    "priority": "medium",
                })

            # Good expansion potential
            expansion = pred.next_iteration.get("applicability_expansion", {})
            if expansion.get("expansion_score", 0) > 0.6:
                recommendations.append({
                    "pattern_id": pattern_id,
                    "action": "expand",
                    "reason": "Pattern can apply to more domains",
                    "priority": "low",
                })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))

        return recommendations
