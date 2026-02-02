"""
Unified Memory Store - The main interface to Memory Nexus Lite.

Brings together all three tiers (temporal, semantic, graph) with
the intelligence layer (semantic caching, pattern learning, pattern intelligence).

The Pattern Intelligence system provides:
- Automatic pattern extraction from stored knowledge
- Usage learning from queries and selections
- Pattern-enhanced retrieval suggestions
- Cross-validation and synthesis of patterns
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
import hashlib
import uuid

from .models import Memory, MemoryResult, MemoryQuery, Relationship
from .temporal import TemporalStore, create_temporal_store
from .semantic import SemanticStore, create_semantic_store
from .graph import GraphStore, create_graph_store

if TYPE_CHECKING:
    from ..patterns import PatternIntelligence

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Unified interface to Memory Nexus Lite.

    Combines three-tier storage with intelligent caching, pattern learning,
    and full pattern intelligence integration.

    Example:
        store = MemoryStore()  # Zero dependencies mode

        # Store knowledge
        store.add("Bu Zhong Yi Qi Tang treats fatigue", context="tcm")

        # Query semantically
        results = store.query("tiredness remedies")

        # Record user selection (feeds pattern learning)
        store.record_selection("tiredness remedies", results[0].memory.id)

        # Create relationships
        store.relate("memory_1", "memory_2", "TREATS")

        # Get pattern-enhanced suggestions
        suggestions = store.get_pattern_suggestions("fatigue")

        # Analyze knowledge base for patterns
        patterns = store.analyze_patterns()
    """

    def __init__(
        self,
        temporal: str = "sqlite",
        semantic: str = "simple",
        graph: str = "sqlite",
        embedding_provider: str = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        data_dir: str = "~/.memory-nexus-lite",
        enable_cache: bool = True,
        cache_similarity_threshold: float = 0.85,
        enable_patterns: bool = True,
    ):
        """
        Initialize Memory Store.

        Args:
            temporal: "sqlite" or "redis://..." URL
            semantic: "simple", "faiss", or "pinecone://..."
            graph: "sqlite" or "neo4j://..."
            embedding_provider: "local" or "openai"
            embedding_model: Model name for embeddings
            data_dir: Directory for SQLite databases
            enable_cache: Enable semantic caching
            cache_similarity_threshold: Min similarity for cache hits
            enable_patterns: Enable pattern intelligence system
        """
        from pathlib import Path

        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding provider
        try:
            from embeddings import create_embeddings
        except ImportError:
            from ..embeddings import create_embeddings
        self.embeddings = create_embeddings(
            embedding_provider,
            model_name=embedding_model
        )

        # Initialize storage tiers
        self.temporal = create_temporal_store(
            temporal,
            db_path=str(self.data_dir / "temporal.db")
        )

        self.semantic = create_semantic_store(
            semantic,
            dimension=self.embeddings.dimension,
            db_path=str(self.data_dir / "semantic.db")
        )

        self.graph = create_graph_store(
            graph,
            db_path=str(self.data_dir / "graph.db")
        )

        # Semantic cache for intelligent query matching
        self.enable_cache = enable_cache
        self.cache_threshold = cache_similarity_threshold
        self._query_cache: Dict[str, Tuple[List[MemoryResult], datetime]] = {}

        # Pattern learning state (basic)
        self._query_patterns: Dict[str, Dict] = {}
        self._query_sequence: List[str] = []

        # Pattern Intelligence System
        self.enable_patterns = enable_patterns
        self._intelligence: Optional["PatternIntelligence"] = None
        self._session_id: str = uuid.uuid4().hex[:12]

        if enable_patterns:
            self._init_pattern_intelligence()

        logger.info(
            f"MemoryStore initialized: temporal={temporal}, "
            f"semantic={semantic}, graph={graph}, "
            f"embeddings={embedding_model}, patterns={enable_patterns}"
        )

    def _init_pattern_intelligence(self) -> None:
        """Initialize the Pattern Intelligence system."""
        try:
            # Try relative import first (when used as package)
            try:
                from ..patterns import PatternIntelligence
            except ImportError:
                # Fall back to absolute import (when running standalone)
                from patterns import PatternIntelligence

            self._intelligence = PatternIntelligence(
                data_dir=str(self.data_dir / "patterns"),
                auto_analyze_threshold=10,
            )
            logger.info("Pattern Intelligence system initialized")
        except ImportError as e:
            logger.warning(f"Pattern Intelligence not available: {e}")
            self.enable_patterns = False
        except Exception as e:
            logger.warning(f"Failed to initialize Pattern Intelligence: {e}")
            self.enable_patterns = False

    @property
    def intelligence(self) -> Optional["PatternIntelligence"]:
        """Access the Pattern Intelligence system."""
        return self._intelligence

    def add(
        self,
        content: str,
        context: str = "default",
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        memory_id: str = None,
    ) -> Memory:
        """
        Add a memory to the store.

        Args:
            content: The text content to store
            context: Context/category for organization
            metadata: Additional metadata
            importance: 0.0 to 1.0, affects retention
            memory_id: Optional specific ID

        Returns:
            Memory object with generated ID
        """
        # Create memory object
        memory = Memory(
            content=content,
            context=context,
            metadata=metadata or {},
            importance=importance,
        )
        if memory_id:
            memory.id = memory_id

        # Generate embedding
        memory.embedding = self.embeddings.embed(content)

        # Store in temporal tier (fast access)
        self.temporal.set(
            f"memory:{memory.id}",
            memory.to_dict(),
            ttl=None  # No expiry for now
        )

        # Store in semantic tier (similarity search)
        self.semantic.add(
            memory.id,
            memory.embedding,
            metadata={
                "context": context,
                "importance": importance,
                **(metadata or {})
            }
        )

        # Add to graph (for relationships)
        self.graph.add_node(
            memory.id,
            labels=["Memory", context],
            properties={
                "content_hash": hashlib.md5(content.encode()).hexdigest()[:8],
                "importance": importance,
                "created_at": memory.created_at.isoformat(),
            }
        )

        logger.debug(f"Added memory {memory.id} to context '{context}'")
        return memory

    def query(
        self,
        query: str,
        context: str = None,
        limit: int = 10,
        threshold: float = 0.5,
        include_related: bool = False,
    ) -> List[MemoryResult]:
        """
        Query memories semantically.

        Args:
            query: Natural language query
            context: Filter by context (optional)
            limit: Maximum results
            threshold: Minimum similarity score
            include_related: Include related memories via graph

        Returns:
            List of MemoryResult with scores
        """
        # Check semantic cache first
        if self.enable_cache:
            cached = self._check_cache(query, context)
            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                self._learn_pattern(query, context, cached, from_cache=True)
                return cached

        # Generate query embedding
        query_embedding = self.embeddings.embed(query)

        # Search semantic store
        filter_dict = {"context": context} if context else None
        results = self.semantic.search(
            query_embedding,
            limit=limit,
            threshold=threshold,
            filter=filter_dict
        )

        # Enrich with full memory data
        memory_results = []
        for memory_id, score, metadata in results:
            memory_data = self.temporal.get(f"memory:{memory_id}")
            if memory_data:
                memory = Memory.from_dict(memory_data)
                memory.touch()  # Mark as accessed

                result = MemoryResult(
                    memory=memory,
                    score=score,
                    source="search"
                )

                # Get related if requested
                if include_related:
                    related = self._get_related_results(memory_id)
                    result.related = related

                memory_results.append(result)

                # Update temporal store with access info
                self.temporal.set(f"memory:{memory_id}", memory.to_dict())

        # Cache the results
        if self.enable_cache and memory_results:
            self._cache_results(query, context, memory_results)

        # Learn from this query (basic patterns)
        self._learn_pattern(query, context, memory_results, from_cache=False)

        # Feed Pattern Intelligence system
        if self.enable_patterns and self._intelligence:
            try:
                self._intelligence.learn_from_query(
                    query=query,
                    results=memory_results,
                    context=context,
                    session_id=self._session_id,
                )
            except Exception as e:
                logger.debug(f"Pattern learning failed: {e}")

        return memory_results

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        data = self.temporal.get(f"memory:{memory_id}")
        if data:
            memory = Memory.from_dict(data)
            memory.touch()
            self.temporal.set(f"memory:{memory_id}", memory.to_dict())
            return memory
        return None

    def relate(
        self,
        source_id: str,
        target_id: str,
        relationship: str = "RELATED_TO",
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Create a relationship between memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relationship: Relationship type (e.g., "TREATS", "CAUSES", "RELATED_TO")
            metadata: Additional relationship metadata
        """
        return self.graph.add_relationship(
            source_id,
            target_id,
            relationship,
            metadata or {}
        )

    def get_related(
        self,
        memory_id: str,
        relationship: str = None,
        depth: int = 1,
    ) -> List[Memory]:
        """Get memories related to given memory."""
        related_nodes = self.graph.get_related_nodes(
            memory_id,
            rel_type=relationship,
            depth=depth
        )

        memories = []
        for node in related_nodes:
            memory_data = self.temporal.get(f"memory:{node['id']}")
            if memory_data:
                memories.append(Memory.from_dict(memory_data))

        return memories

    def forget(self, memory_id: str, cascade: bool = False) -> bool:
        """
        Remove a memory.

        Args:
            memory_id: Memory to remove
            cascade: Also remove related memories
        """
        # Remove from all tiers
        self.temporal.delete(f"memory:{memory_id}")
        self.semantic.delete(memory_id)
        self.graph.delete_node(memory_id, cascade=cascade)

        logger.debug(f"Forgot memory {memory_id}")
        return True

    # =========================================================================
    # Pattern Intelligence Methods
    # =========================================================================

    def record_selection(
        self,
        query: str,
        selected_id: str,
        feedback_score: float = None,
    ) -> None:
        """
        Record that a user selected a specific result.

        This feeds the pattern learning system to improve future retrieval.

        Args:
            query: The query that produced results
            selected_id: ID of the memory the user selected
            feedback_score: Optional explicit feedback (0-1, higher = better)
        """
        if self.enable_patterns and self._intelligence:
            try:
                self._intelligence.record_selection(
                    query=query,
                    selected_id=selected_id,
                    session_id=self._session_id,
                    feedback_score=feedback_score,
                )
            except Exception as e:
                logger.debug(f"Selection recording failed: {e}")

    def record_feedback(
        self,
        memory_id: str,
        feedback_score: float,
        query: str = None,
    ) -> None:
        """
        Record explicit user feedback on a memory.

        Args:
            memory_id: ID of the rated memory
            feedback_score: Score from 0 (bad) to 1 (good)
            query: Optional query context
        """
        if self.enable_patterns and self._intelligence:
            try:
                self._intelligence.record_feedback(
                    memory_id=memory_id,
                    feedback_score=feedback_score,
                    query=query,
                )
            except Exception as e:
                logger.debug(f"Feedback recording failed: {e}")

    def get_pattern_suggestions(
        self,
        query: str,
        context: str = None,
    ) -> Dict[str, Any]:
        """
        Get pattern-based suggestions for a query.

        Returns predictions, preferred results, and related queries
        based on learned usage patterns.

        Args:
            query: The query string
            context: Optional context filter

        Returns:
            Dict with:
                - next_queries: Predicted follow-up queries
                - preferred_results: Results users typically select
                - related_queries: Semantically related queries
                - matching_patterns: Patterns matching the query
        """
        if not self.enable_patterns or not self._intelligence:
            return {
                "next_queries": [],
                "preferred_results": [],
                "related_queries": [],
                "matching_patterns": [],
            }

        try:
            return self._intelligence.suggest_for_query(query, context)
        except Exception as e:
            logger.debug(f"Pattern suggestions failed: {e}")
            return {
                "next_queries": [],
                "preferred_results": [],
                "related_queries": [],
                "matching_patterns": [],
            }

    def analyze_patterns(
        self,
        context: str = None,
        force: bool = False,
    ) -> List[Any]:
        """
        Analyze the knowledge base and extract patterns.

        This examines all stored memories and extracts:
        - Linguistic patterns (keywords, phrases)
        - Structural patterns (document organization)
        - Semantic patterns (topic clusters)
        - Emotional patterns (sentiment)
        - Relationship patterns (connections)

        Args:
            context: Optional context to filter by
            force: Force re-analysis even if patterns exist

        Returns:
            List of extracted Pattern objects
        """
        if not self.enable_patterns or not self._intelligence:
            logger.warning("Pattern Intelligence not available")
            return []

        try:
            return self._intelligence.analyze_knowledge_base(
                store=self,
                context=context,
                force=force,
            )
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []

    def get_patterns(
        self,
        pattern_type: str = None,
        min_confidence: float = 0.0,
        context: str = None,
        limit: int = 100,
    ) -> List[Any]:
        """
        Get stored patterns with optional filtering.

        Args:
            pattern_type: Filter by type (e.g., "semantic", "behavioral")
            min_confidence: Minimum confidence threshold
            context: Filter by context
            limit: Maximum patterns to return

        Returns:
            List of Pattern objects
        """
        if not self.enable_patterns or not self._intelligence:
            return []

        try:
            # Convert string to PatternType if needed
            ptype = None
            if pattern_type:
                try:
                    from ..patterns import PatternType
                except ImportError:
                    from patterns import PatternType
                try:
                    ptype = PatternType(pattern_type)
                except ValueError:
                    pass

            return self._intelligence.get_patterns(
                pattern_type=ptype,
                min_confidence=min_confidence,
                context=context,
                limit=limit,
            )
        except Exception as e:
            logger.debug(f"Get patterns failed: {e}")
            return []

    def find_pattern_gaps(self) -> List[Dict[str, Any]]:
        """
        Find gaps in pattern coverage.

        Identifies areas where patterns are weak or missing,
        which can guide knowledge expansion.

        Returns:
            List of identified gaps with suggestions
        """
        if not self.enable_patterns or not self._intelligence:
            return []

        try:
            return self._intelligence.find_gaps()
        except Exception as e:
            logger.debug(f"Find gaps failed: {e}")
            return []

    def synthesize_pattern(
        self,
        pattern_ids: List[str],
        name: str,
        description: str = None,
    ) -> Optional[Any]:
        """
        Synthesize a new pattern from existing ones.

        Combines evidence, keywords, and relationships from
        multiple patterns into a new synthesized pattern.

        Args:
            pattern_ids: IDs of patterns to synthesize from
            name: Name for the new pattern
            description: Optional description

        Returns:
            New synthesized Pattern or None if failed
        """
        if not self.enable_patterns or not self._intelligence:
            return None

        try:
            return self._intelligence.synthesize_pattern(
                pattern_ids=pattern_ids,
                name=name,
                description=description,
            )
        except Exception as e:
            logger.error(f"Pattern synthesis failed: {e}")
            return None

    def export_patterns(
        self,
        output_path: str,
        min_confidence: float = 0.5,
    ) -> int:
        """
        Export patterns to JSON file.

        Args:
            output_path: Path for output file
            min_confidence: Minimum confidence to export

        Returns:
            Number of patterns exported
        """
        if not self.enable_patterns or not self._intelligence:
            return 0

        try:
            return self._intelligence.export_patterns(
                output_path=output_path,
                min_confidence=min_confidence,
            )
        except Exception as e:
            logger.error(f"Pattern export failed: {e}")
            return 0

    def import_patterns(self, input_path: str) -> int:
        """
        Import patterns from JSON file.

        Args:
            input_path: Path to input file

        Returns:
            Number of patterns imported
        """
        if not self.enable_patterns or not self._intelligence:
            return 0

        try:
            return self._intelligence.import_patterns(input_path)
        except Exception as e:
            logger.error(f"Pattern import failed: {e}")
            return 0

    def _check_cache(
        self,
        query: str,
        context: str
    ) -> Optional[List[MemoryResult]]:
        """Check semantic cache for similar queries."""
        query_embedding = self.embeddings.embed(query)

        # Simple semantic cache check
        # In production, this would use the full SemanticCache class
        for cached_query, (results, timestamp) in self._query_cache.items():
            if context and cached_query.startswith(f"{context}:"):
                cached_query_text = cached_query[len(context) + 1:]
            else:
                cached_query_text = cached_query

            # Quick embedding comparison
            cached_embedding = self.embeddings.embed(cached_query_text)

            # Cosine similarity
            import numpy as np
            q_vec = np.array(query_embedding)
            c_vec = np.array(cached_embedding)
            similarity = np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))

            if similarity >= self.cache_threshold:
                # Update result sources to indicate cache hit
                for r in results:
                    r.source = "cache"
                return results

        return None

    def _cache_results(
        self,
        query: str,
        context: str,
        results: List[MemoryResult]
    ) -> None:
        """Cache query results."""
        cache_key = f"{context}:{query}" if context else query
        self._query_cache[cache_key] = (results, datetime.utcnow())

        # Limit cache size
        if len(self._query_cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k][1]
            )
            for key in sorted_keys[:100]:
                del self._query_cache[key]

    def _learn_pattern(
        self,
        query: str,
        context: str,
        results: List[MemoryResult],
        from_cache: bool
    ) -> None:
        """Learn from query patterns for self-nourishing."""
        pattern_key = f"{context}:{query[:50]}" if context else query[:50]

        if pattern_key not in self._query_patterns:
            self._query_patterns[pattern_key] = {
                "count": 0,
                "cache_hits": 0,
                "avg_results": 0,
                "follow_ups": []
            }

        pattern = self._query_patterns[pattern_key]
        pattern["count"] += 1
        if from_cache:
            pattern["cache_hits"] += 1
        pattern["avg_results"] = (
            (pattern["avg_results"] * (pattern["count"] - 1) + len(results))
            / pattern["count"]
        )

        # Track query sequences for prediction
        if self._query_sequence:
            last_query = self._query_sequence[-1]
            if last_query in self._query_patterns:
                if pattern_key not in self._query_patterns[last_query]["follow_ups"]:
                    self._query_patterns[last_query]["follow_ups"].append(pattern_key)

        self._query_sequence.append(pattern_key)
        if len(self._query_sequence) > 100:
            self._query_sequence = self._query_sequence[-50:]

    def _get_related_results(self, memory_id: str) -> List[MemoryResult]:
        """Get related memories as results."""
        related = self.get_related(memory_id, depth=1)
        return [
            MemoryResult(memory=m, score=0.8, source="related")
            for m in related
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics including pattern intelligence."""
        graph_stats = self.graph.get_stats()

        stats = {
            "memories": graph_stats.get("nodes", 0),
            "relationships": graph_stats.get("relationships", 0),
            "vectors": self.semantic.count,
            "cache_size": len(self._query_cache),
            "patterns_learned": len(self._query_patterns),
        }

        # Add pattern intelligence stats
        if self.enable_patterns and self._intelligence:
            try:
                intel_stats = self._intelligence.get_stats()
                stats["pattern_intelligence"] = intel_stats
            except Exception:
                stats["pattern_intelligence"] = {"error": "unavailable"}

        return stats

    def get_pattern_predictions(self, query: str, context: str = None) -> List[str]:
        """Predict likely follow-up queries based on learned patterns."""
        pattern_key = f"{context}:{query[:50]}" if context else query[:50]

        if pattern_key in self._query_patterns:
            return self._query_patterns[pattern_key].get("follow_ups", [])

        return []

    def close(self) -> None:
        """Close all connections."""
        self.temporal.close()
        if hasattr(self.semantic, "close"):
            self.semantic.close()
        if hasattr(self.graph, "close"):
            self.graph.close()
        if self._intelligence:
            self._intelligence.close()
