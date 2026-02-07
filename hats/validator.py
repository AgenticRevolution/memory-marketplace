"""
Hat Validator - Quality checks for specialist memory hats.

Validates that a hat meets minimum quality standards before
being registered in the catalog and made available to subscribers.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add memory-nexus to path (bundled in sibling directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "memory-nexus"))


# Quality thresholds
MINIMUM_MEMORIES = 100
MINIMUM_CONTEXTS = 3
MINIMUM_RELATIONSHIPS = 50
MINIMUM_PATTERNS = 0  # Patterns emerge from usage, not seeding
MINIMUM_QUERY_PASS_RATE = 0.80
TARGET_QUERY_PASS_RATE = 0.95


class ValidationResult:
    """Result of a hat validation run."""

    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
        self.score: int = 0
        self.passed: bool = False

    def add_check(self, name: str, passed: bool, value: Any, minimum: Any, weight: int = 10):
        self.checks.append({
            "name": name,
            "passed": passed,
            "value": value,
            "minimum": minimum,
            "weight": weight,
        })

    def compute_score(self) -> int:
        """Compute quality score 0-100 from weighted checks."""
        if not self.checks:
            return 0
        total_weight = sum(c["weight"] for c in self.checks)
        earned = sum(c["weight"] for c in self.checks if c["passed"])
        self.score = int((earned / total_weight) * 100) if total_weight > 0 else 0
        self.passed = all(c["passed"] for c in self.checks if c["weight"] >= 10)
        return self.score

    def summary(self) -> str:
        lines = [f"Quality Score: {self.score}/100 ({'PASS' if self.passed else 'FAIL'})\n"]
        for check in self.checks:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(f"  [{status}] {check['name']}: {check['value']} (min: {check['minimum']})")
        return "\n".join(lines)


class HatValidator:
    """
    Validates hat data directories against quality standards.

    Checks:
    1. Memory count meets minimum
    2. Sufficient context diversity
    3. Relationship density
    4. Pattern extraction count
    5. Query test suite pass rate (if tests provided)
    """

    def __init__(self, data_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self._store = None

    def _get_store(self):
        if self._store is None:
            from core.store import MemoryStore
            self._store = MemoryStore(
                data_dir=str(self.data_dir),
                embedding_model=self.embedding_model,
                enable_cache=False,
                enable_patterns=False,
            )
        return self._store

    def validate(self, test_queries: Optional[List[Dict]] = None) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            test_queries: Optional list of test queries, each with:
                - query: The search query
                - expected_context: Expected context in results (optional)
                - expected_keywords: Keywords that should appear in results (optional)
                - min_score: Minimum similarity score for top result (optional, default 0.3)

        Returns:
            ValidationResult with score and per-check details
        """
        result = ValidationResult()
        store = self._get_store()
        stats = store.get_stats()

        # Check 1: Memory count
        memory_count = stats.get("memories", stats.get("nodes", 0))
        result.add_check(
            name="Memory count",
            passed=memory_count >= MINIMUM_MEMORIES,
            value=memory_count,
            minimum=MINIMUM_MEMORIES,
            weight=20,
        )

        # Check 2: Context diversity
        contexts = self._count_contexts(store)
        context_count = len(contexts)
        result.add_check(
            name="Context diversity",
            passed=context_count >= MINIMUM_CONTEXTS,
            value=context_count,
            minimum=MINIMUM_CONTEXTS,
            weight=15,
        )

        # Check 3: Relationship density
        rel_count = stats.get("relationships", 0)
        result.add_check(
            name="Relationships",
            passed=rel_count >= MINIMUM_RELATIONSHIPS,
            value=rel_count,
            minimum=MINIMUM_RELATIONSHIPS,
            weight=15,
        )

        # Check 4: Pattern count
        pattern_count = stats.get("patterns_learned", 0)
        result.add_check(
            name="Patterns extracted",
            passed=pattern_count >= MINIMUM_PATTERNS,
            value=pattern_count,
            minimum=MINIMUM_PATTERNS,
            weight=10,
        )

        # Check 5: Vector count matches memory count
        vector_count = stats.get("vectors", 0)
        vectors_match = vector_count >= memory_count * 0.9  # Allow 10% tolerance
        result.add_check(
            name="Embeddings coverage",
            passed=vectors_match,
            value=f"{vector_count}/{memory_count}",
            minimum="90%",
            weight=10,
        )

        # Check 6: Query test suite
        if test_queries:
            pass_rate = self._run_query_tests(store, test_queries)
            result.add_check(
                name="Query test pass rate",
                passed=pass_rate >= MINIMUM_QUERY_PASS_RATE,
                value=f"{pass_rate:.0%}",
                minimum=f"{MINIMUM_QUERY_PASS_RATE:.0%}",
                weight=30,
            )

        result.compute_score()
        return result

    def _count_contexts(self, store) -> Dict[str, int]:
        """Count memories per context by scanning the graph."""
        contexts = {}
        try:
            # Try to get contexts from graph labels
            graph_stats = store.graph.get_stats()
            if "labels" in graph_stats:
                for label, count in graph_stats["labels"].items():
                    if label != "Memory":  # Skip the base label
                        contexts[label] = count
        except Exception:
            pass

        # Fallback: broad query
        if not contexts:
            try:
                results = store.query("*", limit=1000, threshold=0.0)
                for r in results:
                    ctx = r.memory.context
                    contexts[ctx] = contexts.get(ctx, 0) + 1
            except Exception:
                pass

        return contexts

    def _run_query_tests(self, store, test_queries: List[Dict]) -> float:
        """
        Run test queries and return pass rate.

        Each test query can specify:
        - query: The search text
        - expected_context: A context that should appear in results
        - expected_keywords: Keywords that should appear in result content
        - min_score: Minimum similarity for top result
        """
        passed = 0
        total = len(test_queries)

        for test in test_queries:
            query = test["query"]
            try:
                results = store.query(query, limit=5, threshold=0.1)

                if not results:
                    logger.debug(f"FAIL: No results for '{query}'")
                    continue

                test_passed = True

                # Check minimum score
                min_score = test.get("min_score", 0.3)
                if results[0].score < min_score:
                    logger.debug(f"FAIL: Top score {results[0].score:.2f} < {min_score} for '{query}'")
                    test_passed = False

                # Check expected context
                expected_ctx = test.get("expected_context")
                if expected_ctx and test_passed:
                    result_contexts = {r.memory.context for r in results}
                    if expected_ctx not in result_contexts:
                        logger.debug(f"FAIL: Context '{expected_ctx}' not in results for '{query}'")
                        test_passed = False

                # Check expected keywords
                expected_kw = test.get("expected_keywords", [])
                if expected_kw and test_passed:
                    all_content = " ".join(r.memory.content.lower() for r in results)
                    for kw in expected_kw:
                        if kw.lower() not in all_content:
                            logger.debug(f"FAIL: Keyword '{kw}' not found in results for '{query}'")
                            test_passed = False
                            break

                if test_passed:
                    passed += 1

            except Exception as e:
                logger.warning(f"Error running test query '{query}': {e}")

        return passed / total if total > 0 else 0.0

    def validate_from_file(self, tests_path: str) -> ValidationResult:
        """
        Validate using test queries from a JSON file.

        Expected format:
        {
            "queries": [
                {"query": "...", "expected_context": "...", "expected_keywords": ["..."]},
                ...
            ]
        }
        """
        tests_path = Path(tests_path)
        if tests_path.exists():
            data = json.loads(tests_path.read_text())
            return self.validate(test_queries=data.get("queries", []))
        return self.validate()

    def close(self):
        if self._store:
            self._store.close()
            self._store = None


def main():
    """CLI entry point for hat validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate a specialist memory hat")
    parser.add_argument("--data", required=True, help="Path to hat data directory")
    parser.add_argument("--tests", help="Path to test queries JSON file")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    validator = HatValidator(args.data, embedding_model=args.model)
    try:
        if args.tests:
            result = validator.validate_from_file(args.tests)
        else:
            result = validator.validate()

        print(result.summary())
        sys.exit(0 if result.passed else 1)
    finally:
        validator.close()


if __name__ == "__main__":
    main()
