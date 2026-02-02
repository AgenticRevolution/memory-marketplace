"""
SIMBA - Self-Improving Meta-learning Brain Architecture.

Adapted from the Distillation Engine's SIMBA integration.

The 5 Universal Enhancement Dimensions:
1. Context Weaving (PatternConnectionWeaver) - Hidden connections
2. Pattern Prophecy (PatternEvolutionPredictor) - Future prediction
3. Insight Crystallization (PatternWisdomExtractor) - Deep wisdom
4. Value Creation (PatternValueAssessor) - Economic/strategic value
5. Emergence Guardian (PatternBreakthroughDetector) - Breakthrough detection

Plus: PatternOptimizer for decay, freshness, and re-ranking
"""

from .connection_weaver import PatternConnectionWeaver
from .evolution_predictor import PatternEvolutionPredictor
from .wisdom_extractor import PatternWisdomExtractor
from .value_assessor import PatternValueAssessor
from .breakthrough_detector import PatternBreakthroughDetector
from .optimizer import PatternOptimizer
from .simba_core import SimbaEnhancer

__all__ = [
    "PatternConnectionWeaver",
    "PatternEvolutionPredictor",
    "PatternWisdomExtractor",
    "PatternValueAssessor",
    "PatternBreakthroughDetector",
    "PatternOptimizer",
    "SimbaEnhancer",
]
