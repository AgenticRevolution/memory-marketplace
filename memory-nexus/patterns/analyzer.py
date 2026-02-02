"""
Pattern Analyzer - Multi-Source Analysis & Cross-Validation.

Adapted from the Distillation Engine's MultiSourcePatternAnalyzer.

Provides:
- Pattern comparison and similarity
- Cross-validation across sources
- Pattern synthesis (combining patterns)
- Pattern relationship detection
- Quality scoring
"""

import hashlib
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .types import (
    ConfidenceLevel,
    Pattern,
    PatternEvidence,
    PatternRelationship,
    PatternType,
    RelationshipType,
)


class PatternAnalyzer:
    """
    Analyze, compare, and synthesize patterns.

    This is the intelligence layer that:
    - Compares patterns for similarity
    - Cross-validates patterns across sources
    - Synthesizes new patterns from existing ones
    - Detects relationships between patterns
    - Scores pattern quality and stability

    Example:
        analyzer = PatternAnalyzer()

        # Compare two patterns
        similarity = analyzer.compare_patterns(pattern_a, pattern_b)

        # Cross-validate all patterns
        validated, invalidated = analyzer.cross_validate(patterns)

        # Synthesize a new pattern from related ones
        synthesized = analyzer.synthesize_patterns([p1, p2, p3], "Combined Topic")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        validation_threshold: float = 0.5,
    ):
        """
        Initialize analyzer.

        Args:
            similarity_threshold: Threshold for considering patterns similar
            validation_threshold: Threshold for cross-validation
        """
        self.similarity_threshold = similarity_threshold
        self.validation_threshold = validation_threshold

    def compare_patterns(
        self,
        pattern_a: Pattern,
        pattern_b: Pattern,
    ) -> Dict[str, Any]:
        """
        Compare two patterns for similarity.

        Returns detailed comparison with overall similarity score
        and reasons for the score.

        Args:
            pattern_a: First pattern
            pattern_b: Second pattern

        Returns:
            Dict with similarity score and reasons
        """
        similarity = 0.0
        reasons = []

        # 1. Compare pattern type (30% weight)
        if pattern_a.pattern_type == pattern_b.pattern_type:
            similarity += 0.3
            reasons.append(f"Same type: {pattern_a.pattern_type.value}")

        # 2. Compare keywords (30% weight)
        if pattern_a.keywords and pattern_b.keywords:
            keywords_a = set(pattern_a.keywords)
            keywords_b = set(pattern_b.keywords)
            if keywords_a and keywords_b:
                overlap = len(keywords_a & keywords_b)
                union = len(keywords_a | keywords_b)
                keyword_sim = overlap / union if union > 0 else 0
                similarity += 0.3 * keyword_sim
                if keyword_sim > 0:
                    reasons.append(f"Keyword overlap: {keyword_sim:.0%}")

        # 3. Compare embedding fingerprints (25% weight)
        if pattern_a.embedding_fingerprint and pattern_b.embedding_fingerprint:
            cosine_sim = self._cosine_similarity(
                pattern_a.embedding_fingerprint,
                pattern_b.embedding_fingerprint
            )
            similarity += 0.25 * cosine_sim
            if cosine_sim > 0.5:
                reasons.append(f"Embedding similarity: {cosine_sim:.0%}")

        # 4. Compare tags (10% weight)
        if pattern_a.tags and pattern_b.tags:
            tags_a = set(pattern_a.tags)
            tags_b = set(pattern_b.tags)
            if tags_a and tags_b:
                tag_overlap = len(tags_a & tags_b) / len(tags_a | tags_b)
                similarity += 0.1 * tag_overlap
                if tag_overlap > 0:
                    reasons.append(f"Tag overlap: {tag_overlap:.0%}")

        # 5. Compare confidence levels (5% weight)
        conf_diff = abs(pattern_a.confidence - pattern_b.confidence)
        if conf_diff < 0.2:
            conf_score = 0.05 * (1 - conf_diff / 0.2)
            similarity += conf_score
            reasons.append(f"Similar confidence")

        return {
            "similarity": similarity,
            "reasons": reasons,
            "is_similar": similarity >= self.similarity_threshold,
            "pattern_a_id": pattern_a.id,
            "pattern_b_id": pattern_b.id,
        }

    def cross_validate(
        self,
        patterns: List[Pattern],
    ) -> Tuple[List[Pattern], List[Pattern]]:
        """
        Cross-validate patterns against each other.

        Patterns supported by multiple sources gain confidence.
        Patterns that contradict each other are flagged.

        Args:
            patterns: List of patterns to validate

        Returns:
            Tuple of (validated_patterns, invalidated_patterns)
        """
        if len(patterns) < 2:
            return patterns, []

        validated = []
        invalidated = []

        # Group patterns by type
        by_type: Dict[PatternType, List[Pattern]] = defaultdict(list)
        for p in patterns:
            by_type[p.pattern_type].append(p)

        # Compare patterns within each type
        for pattern_type, type_patterns in by_type.items():
            if len(type_patterns) < 2:
                validated.extend(type_patterns)
                continue

            for pattern in type_patterns:
                support_score = 0.0
                contradiction_score = 0.0

                for other in type_patterns:
                    if pattern.id == other.id:
                        continue

                    comparison = self.compare_patterns(pattern, other)

                    # Check if they have distinct sources
                    pattern_sources = {e.source_id for e in pattern.evidence}
                    other_sources = {e.source_id for e in other.evidence}
                    has_distinct = bool(other_sources - pattern_sources)

                    if not has_distinct:
                        continue

                    # High similarity = supporting evidence
                    if comparison["similarity"] > 0.7:
                        support_score += comparison["similarity"]
                        # Add relationship
                        pattern.relationships.append(PatternRelationship(
                            source_pattern_id=pattern.id,
                            target_pattern_id=other.id,
                            relationship_type=RelationshipType.SUPPORTS,
                            strength=comparison["similarity"],
                            description="; ".join(comparison["reasons"]),
                        ))
                    # Low similarity with same keywords = potential contradiction
                    elif comparison["similarity"] < 0.3:
                        # Check for keyword overlap (might be contradicting)
                        if pattern.keywords and other.keywords:
                            kw_overlap = len(
                                set(pattern.keywords) & set(other.keywords)
                            ) / max(len(pattern.keywords), 1)
                            if kw_overlap > 0.3:
                                contradiction_score += 0.5

                # Calculate validation score
                total = support_score + contradiction_score
                if total > 0:
                    validation_score = (support_score - contradiction_score) / total
                else:
                    validation_score = 0

                # Update pattern based on validation
                if validation_score > 0:
                    # Increase stability
                    pattern.stability = min(1.0, pattern.stability + 0.1)
                    # Slightly boost confidence
                    pattern.confidence = min(1.0, pattern.confidence + 0.05)
                    validated.append(pattern)
                elif validation_score < -0.3:
                    # Decrease stability
                    pattern.stability = max(0.1, pattern.stability - 0.2)
                    invalidated.append(pattern)
                else:
                    # Neutral - keep as is
                    validated.append(pattern)

        return validated, invalidated

    def synthesize_patterns(
        self,
        patterns: List[Pattern],
        name: str,
        description: str = None,
        synthesis_method: str = "aggregation",
    ) -> Pattern:
        """
        Synthesize a new pattern from multiple existing patterns.

        Combines evidence, keywords, and relationships from source patterns.

        Args:
            patterns: Patterns to synthesize from (minimum 2)
            name: Name for the synthesized pattern
            description: Optional description
            synthesis_method: Method used for synthesis

        Returns:
            New synthesized pattern
        """
        if len(patterns) < 2:
            raise ValueError("Synthesis requires at least 2 patterns")

        # Determine pattern type (use most common)
        type_counts = defaultdict(int)
        for p in patterns:
            type_counts[p.pattern_type] += 1
        pattern_type = max(type_counts, key=type_counts.get)

        # Aggregate evidence
        all_evidence = []
        for p in patterns:
            for e in p.evidence:
                # Adjust relevance based on source pattern confidence
                adjusted = PatternEvidence(
                    source_id=e.source_id,
                    source_type=e.source_type,
                    confidence=e.confidence * p.confidence,
                    relevance=e.relevance * 0.8,  # Slightly reduce for synthesis
                    detected_at=e.detected_at,
                    detection_method="synthesized_from_" + e.detection_method,
                    snippets=e.snippets,
                    metadata={**e.metadata, "synthesized": True},
                )
                all_evidence.append(adjusted)

        # Combine keywords (deduplicated)
        all_keywords = []
        seen_keywords = set()
        for p in patterns:
            for kw in p.keywords:
                if kw not in seen_keywords:
                    all_keywords.append(kw)
                    seen_keywords.add(kw)

        # Combine tags
        all_tags = list(set(tag for p in patterns for tag in p.tags))
        all_tags.append("synthesized")

        # Calculate aggregate confidence
        avg_confidence = np.mean([p.confidence for p in patterns])

        # Combine fingerprints if available
        fingerprints = [p.embedding_fingerprint for p in patterns if p.embedding_fingerprint]
        combined_fingerprint = None
        if fingerprints:
            # Average the fingerprints
            combined_fingerprint = np.mean(fingerprints, axis=0).tolist()

        # Generate description if not provided
        if not description:
            source_names = [p.name for p in patterns[:3]]
            description = f"Synthesized from: {', '.join(source_names)}"

        # Create synthesized pattern
        synthesized = Pattern(
            id=self._generate_id("synthesized"),
            name=name,
            description=description,
            pattern_type=pattern_type,
            evidence=all_evidence,
            confidence=avg_confidence,
            stability=0.5,  # Start at medium stability
            tags=all_tags,
            context=patterns[0].context,  # Use first pattern's context
            is_synthesized=True,
            synthesis_method=synthesis_method,
            source_pattern_ids=[p.id for p in patterns],
            keywords=all_keywords,
            embedding_fingerprint=combined_fingerprint,
        )

        # Create relationships to source patterns
        for p in patterns:
            synthesized.relationships.append(PatternRelationship(
                source_pattern_id=synthesized.id,
                target_pattern_id=p.id,
                relationship_type=RelationshipType.GENERALIZES,
                strength=p.confidence,
                description=f"Synthesized from {p.name}",
            ))

        synthesized._recalculate_confidence()
        return synthesized

    def find_mergeable_patterns(
        self,
        patterns: List[Pattern],
    ) -> List[List[Pattern]]:
        """
        Find groups of patterns that could be merged.

        Returns groups of highly similar patterns.

        Args:
            patterns: Patterns to analyze

        Returns:
            List of pattern groups that could be merged
        """
        groups = []
        processed = set()

        for pattern in patterns:
            if pattern.id in processed:
                continue

            group = [pattern]
            processed.add(pattern.id)

            for other in patterns:
                if other.id in processed:
                    continue

                comparison = self.compare_patterns(pattern, other)
                if comparison["similarity"] >= 0.8:
                    group.append(other)
                    processed.add(other.id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def detect_relationships(
        self,
        patterns: List[Pattern],
    ) -> List[PatternRelationship]:
        """
        Detect relationships between patterns.

        Analyzes all pattern pairs and creates appropriate relationships.

        Args:
            patterns: Patterns to analyze

        Returns:
            List of detected relationships
        """
        relationships = []

        for i, pattern_a in enumerate(patterns):
            for pattern_b in patterns[i+1:]:
                comparison = self.compare_patterns(pattern_a, pattern_b)

                if comparison["similarity"] < 0.3:
                    continue  # Too dissimilar

                # Determine relationship type
                if comparison["similarity"] >= 0.9:
                    rel_type = RelationshipType.SIMILAR
                elif comparison["similarity"] >= 0.7:
                    # Check if one is more specific
                    if len(pattern_a.keywords) > len(pattern_b.keywords) * 1.5:
                        rel_type = RelationshipType.SPECIALIZES
                    elif len(pattern_b.keywords) > len(pattern_a.keywords) * 1.5:
                        rel_type = RelationshipType.GENERALIZES
                    else:
                        rel_type = RelationshipType.CORRELATES
                else:
                    rel_type = RelationshipType.CORRELATES

                # Create relationship
                rel = PatternRelationship(
                    source_pattern_id=pattern_a.id,
                    target_pattern_id=pattern_b.id,
                    relationship_type=rel_type,
                    strength=comparison["similarity"],
                    description="; ".join(comparison["reasons"]),
                )
                relationships.append(rel)

                # Also add to patterns
                pattern_a.relationships.append(rel)

        return relationships

    def score_pattern_quality(
        self,
        pattern: Pattern,
    ) -> Dict[str, float]:
        """
        Score a pattern's quality across multiple dimensions.

        Args:
            pattern: Pattern to score

        Returns:
            Dict with quality scores
        """
        scores = {}

        # Evidence diversity (0-1)
        source_ids = {e.source_id for e in pattern.evidence}
        source_types = {e.source_type for e in pattern.evidence}
        scores["evidence_diversity"] = min(1.0, (
            len(source_ids) / 5 * 0.5 +
            len(source_types) / 3 * 0.5
        ))

        # Confidence (direct)
        scores["confidence"] = pattern.confidence

        # Stability (direct)
        scores["stability"] = pattern.stability

        # Keyword richness (0-1)
        scores["keyword_richness"] = min(1.0, len(pattern.keywords) / 10)

        # Application success (if any applications)
        if pattern.application_count > 0:
            scores["application_success"] = pattern.success_rate
        else:
            scores["application_success"] = 0.5  # Neutral if untested

        # Relationship connectivity
        scores["connectivity"] = min(1.0, len(pattern.relationships) / 5)

        # Overall quality score
        weights = {
            "evidence_diversity": 0.2,
            "confidence": 0.25,
            "stability": 0.15,
            "keyword_richness": 0.1,
            "application_success": 0.2,
            "connectivity": 0.1,
        }
        scores["overall"] = sum(
            scores[k] * w for k, w in weights.items()
        )

        return scores

    def find_gaps(
        self,
        patterns: List[Pattern],
    ) -> List[Dict[str, Any]]:
        """
        Identify gaps in pattern coverage.

        Finds areas where patterns are weak or missing.

        Args:
            patterns: Current patterns

        Returns:
            List of identified gaps
        """
        gaps = []

        # Group patterns by type
        by_type: Dict[PatternType, List[Pattern]] = defaultdict(list)
        for p in patterns:
            by_type[p.pattern_type].append(p)

        # Check for missing pattern types
        for ptype in PatternType:
            if ptype not in by_type or len(by_type[ptype]) == 0:
                gaps.append({
                    "type": "missing_pattern_type",
                    "pattern_type": ptype.value,
                    "severity": "medium",
                    "suggestion": f"No {ptype.value} patterns detected",
                })

        # Check for low-confidence patterns
        low_conf = [p for p in patterns if p.confidence < 0.4]
        if len(low_conf) > len(patterns) * 0.3:
            gaps.append({
                "type": "low_confidence",
                "count": len(low_conf),
                "severity": "high",
                "suggestion": "Many patterns have low confidence - need more evidence",
            })

        # Check for isolated patterns (no relationships)
        isolated = [p for p in patterns if len(p.relationships) == 0]
        if len(isolated) > len(patterns) * 0.5:
            gaps.append({
                "type": "isolated_patterns",
                "count": len(isolated),
                "severity": "low",
                "suggestion": "Many patterns are not connected - consider cross-validation",
            })

        # Check for context coverage
        contexts = {p.context for p in patterns if p.context}
        if len(contexts) == 1:
            gaps.append({
                "type": "single_context",
                "context": list(contexts)[0],
                "severity": "low",
                "suggestion": "All patterns from single context - consider diversifying",
            })

        return gaps

    def _cosine_similarity(
        self,
        vec_a: List[float],
        vec_b: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _generate_id(self, prefix: str) -> str:
        """Generate unique pattern ID."""
        unique = f"{prefix}_{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique.encode()).hexdigest()[:16]
