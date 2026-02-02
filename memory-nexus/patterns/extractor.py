"""
Pattern Extractor - Local Pattern Detection.

Extracts patterns from content using statistical and NLP techniques.
Works WITHOUT any LLM dependency - pure local processing.

Techniques used:
- TF-IDF for keyword extraction
- N-gram analysis for phrases
- Statistical frequency analysis
- Structural analysis (sentences, paragraphs)
- Basic sentiment via keyword matching
"""

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .types import (
    Pattern,
    PatternEvidence,
    PatternType,
)


@dataclass
class ExtractionConfig:
    """Configuration for pattern extraction."""
    min_keyword_freq: int = 2              # Minimum frequency for keywords
    max_keywords: int = 20                 # Max keywords to extract
    ngram_range: Tuple[int, int] = (2, 4)  # N-gram sizes
    min_ngram_freq: int = 2                # Minimum n-gram frequency
    max_ngrams: int = 15                   # Max n-grams to keep
    min_pattern_confidence: float = 0.3    # Minimum confidence to create pattern
    fingerprint_dim: int = 64              # Embedding fingerprint dimensions


class PatternExtractor:
    """
    Extract patterns from content using local processing.

    No LLM required - uses statistical NLP techniques.

    Example:
        extractor = PatternExtractor()

        # Extract from a list of content
        patterns = extractor.extract_from_content([
            "Bu Zhong Yi Qi Tang treats fatigue and weakness",
            "Long COVID causes chronic fatigue",
            "Huang Qi tonifies Qi and boosts energy"
        ])

        # Extract from memories
        patterns = extractor.extract_from_memories(memories)
    """

    # Common English stop words
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    }

    # Sentiment indicators (simple keyword matching)
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'beneficial', 'effective', 'helps',
        'improves', 'treats', 'heals', 'supports', 'strengthens', 'tonifies',
        'nourishes', 'restores', 'balances', 'harmonizes', 'recommended',
    }

    NEGATIVE_WORDS = {
        'bad', 'poor', 'harmful', 'dangerous', 'contraindicated', 'avoid',
        'causes', 'worsens', 'depletes', 'damages', 'weakens', 'disrupts',
        'caution', 'warning', 'risk', 'side effect', 'adverse',
    }

    def __init__(self, config: ExtractionConfig = None):
        """Initialize extractor with config."""
        self.config = config or ExtractionConfig()

    def extract_from_content(
        self,
        content_items: List[str],
        source_ids: List[str] = None,
        context: str = None,
    ) -> List[Pattern]:
        """
        Extract patterns from a list of content strings.

        Args:
            content_items: List of text content to analyze
            source_ids: Optional IDs for each content item
            context: Optional context label

        Returns:
            List of extracted patterns
        """
        if len(content_items) < 2:
            return []

        if source_ids is None:
            source_ids = [str(i) for i in range(len(content_items))]

        patterns = []

        # 1. Extract linguistic patterns (keywords, phrases)
        linguistic = self._extract_linguistic_patterns(
            content_items, source_ids, context
        )
        patterns.extend(linguistic)

        # 2. Extract structural patterns
        structural = self._extract_structural_patterns(
            content_items, source_ids, context
        )
        patterns.extend(structural)

        # 3. Extract semantic patterns (topic clusters)
        semantic = self._extract_semantic_patterns(
            content_items, source_ids, context
        )
        patterns.extend(semantic)

        # 4. Extract emotional patterns
        emotional = self._extract_emotional_patterns(
            content_items, source_ids, context
        )
        patterns.extend(emotional)

        # Filter by confidence threshold
        patterns = [
            p for p in patterns
            if p.confidence >= self.config.min_pattern_confidence
        ]

        return patterns

    def extract_from_memories(
        self,
        memories: List[Any],
        context: str = None,
    ) -> List[Pattern]:
        """
        Extract patterns from Memory objects.

        Args:
            memories: List of Memory objects with .content attribute
            context: Optional context filter

        Returns:
            List of extracted patterns
        """
        content_items = []
        source_ids = []

        for memory in memories:
            if context and hasattr(memory, 'context') and memory.context != context:
                continue
            content_items.append(memory.content)
            source_ids.append(memory.id)

        return self.extract_from_content(content_items, source_ids, context)

    def _extract_linguistic_patterns(
        self,
        content_items: List[str],
        source_ids: List[str],
        context: str,
    ) -> List[Pattern]:
        """Extract keyword and phrase patterns."""
        patterns = []

        # Combine all content
        all_text = " ".join(content_items).lower()

        # Tokenize
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        words = [w for w in words if w not in self.STOP_WORDS]

        # 1. Keyword frequency patterns
        word_freq = Counter(words)
        top_keywords = word_freq.most_common(self.config.max_keywords)

        if top_keywords:
            # Find which documents contain each keyword
            keyword_sources = defaultdict(list)
            for idx, content in enumerate(content_items):
                content_lower = content.lower()
                for kw, _ in top_keywords:
                    if kw in content_lower:
                        keyword_sources[kw].append(source_ids[idx])

            # Create pattern for significant keywords
            significant_keywords = [
                (kw, freq) for kw, freq in top_keywords
                if freq >= self.config.min_keyword_freq
                and len(keyword_sources[kw]) >= 2  # Appears in multiple docs
            ]

            if significant_keywords:
                # Build evidence
                evidence = []
                for kw, freq in significant_keywords[:10]:
                    for src_id in keyword_sources[kw][:3]:
                        evidence.append(PatternEvidence(
                            source_id=src_id,
                            source_type="memory",
                            confidence=min(1.0, freq / 10),
                            relevance=1.0,
                            detection_method="keyword_frequency",
                            snippets=[kw],
                        ))

                pattern = Pattern(
                    id=self._generate_id("linguistic_keywords"),
                    name=f"Key Terms: {', '.join([k for k, _ in significant_keywords[:5]])}",
                    description=f"Frequently occurring terms across {len(content_items)} documents",
                    pattern_type=PatternType.LINGUISTIC,
                    evidence=evidence,
                    context=context,
                    keywords=[k for k, _ in significant_keywords],
                    tags=["keywords", "frequency"],
                )
                pattern._recalculate_confidence()
                patterns.append(pattern)

        # 2. N-gram patterns (phrases)
        ngrams = self._extract_ngrams(all_text)
        if ngrams:
            ngram_evidence = []
            for ngram, freq in ngrams[:5]:
                # Find sources containing this ngram
                for idx, content in enumerate(content_items):
                    if ngram in content.lower():
                        ngram_evidence.append(PatternEvidence(
                            source_id=source_ids[idx],
                            source_type="memory",
                            confidence=min(1.0, freq / 5),
                            relevance=0.8,
                            detection_method="ngram_analysis",
                            snippets=[ngram],
                        ))
                        break  # One evidence per ngram

            if ngram_evidence:
                pattern = Pattern(
                    id=self._generate_id("linguistic_phrases"),
                    name=f"Common Phrases",
                    description=f"Recurring phrases: {', '.join([n for n, _ in ngrams[:5]])}",
                    pattern_type=PatternType.LINGUISTIC,
                    evidence=ngram_evidence,
                    context=context,
                    keywords=[n for n, _ in ngrams[:10]],
                    tags=["phrases", "ngrams"],
                )
                pattern._recalculate_confidence()
                patterns.append(pattern)

        return patterns

    def _extract_ngrams(self, text: str) -> List[Tuple[str, int]]:
        """Extract significant n-grams from text."""
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())

        ngram_counter = Counter()
        for n in range(self.config.ngram_range[0], self.config.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                # Skip if contains only stop words
                ngram_words = set(ngram.split())
                if not ngram_words.issubset(self.STOP_WORDS):
                    ngram_counter[ngram] += 1

        # Filter by frequency and return top
        significant = [
            (ng, freq) for ng, freq in ngram_counter.items()
            if freq >= self.config.min_ngram_freq
        ]
        significant.sort(key=lambda x: x[1], reverse=True)

        return significant[:self.config.max_ngrams]

    def _extract_structural_patterns(
        self,
        content_items: List[str],
        source_ids: List[str],
        context: str,
    ) -> List[Pattern]:
        """Extract structural patterns (length, format, organization)."""
        patterns = []

        # Analyze structure of each document
        structures = []
        for content in content_items:
            structures.append({
                'char_count': len(content),
                'word_count': len(content.split()),
                'sentence_count': len(re.split(r'[.!?]+', content)),
                'has_numbers': bool(re.search(r'\d', content)),
                'has_list': bool(re.search(r'[-•*]\s', content)),
                'has_colon': ':' in content,
            })

        # Find structural patterns
        avg_words = np.mean([s['word_count'] for s in structures])
        avg_sentences = np.mean([s['sentence_count'] for s in structures])

        # Pattern: Document length consistency
        word_std = np.std([s['word_count'] for s in structures])
        if word_std < avg_words * 0.5:  # Low variance = consistent length
            evidence = [
                PatternEvidence(
                    source_id=source_ids[0],
                    source_type="memory",
                    confidence=0.7,
                    relevance=1.0,
                    detection_method="structural_analysis",
                    metadata={"avg_words": avg_words, "std": word_std},
                )
            ]
            pattern = Pattern(
                id=self._generate_id("structural_length"),
                name=f"Consistent Length (~{int(avg_words)} words)",
                description=f"Documents maintain consistent length of approximately {int(avg_words)} words",
                pattern_type=PatternType.STRUCTURAL,
                evidence=evidence,
                context=context,
                tags=["structure", "length"],
            )
            pattern._recalculate_confidence()
            patterns.append(pattern)

        # Pattern: Common format elements
        has_numbers_pct = sum(1 for s in structures if s['has_numbers']) / len(structures)
        has_list_pct = sum(1 for s in structures if s['has_list']) / len(structures)

        if has_numbers_pct > 0.5:
            evidence = [
                PatternEvidence(
                    source_id=source_ids[i],
                    source_type="memory",
                    confidence=0.8,
                    relevance=1.0,
                    detection_method="structural_analysis",
                )
                for i, s in enumerate(structures) if s['has_numbers']
            ][:5]

            pattern = Pattern(
                id=self._generate_id("structural_numbers"),
                name="Numeric Content Pattern",
                description=f"{int(has_numbers_pct*100)}% of content contains numbers/measurements",
                pattern_type=PatternType.STRUCTURAL,
                evidence=evidence,
                context=context,
                tags=["structure", "numeric"],
            )
            pattern._recalculate_confidence()
            patterns.append(pattern)

        return patterns

    def _extract_semantic_patterns(
        self,
        content_items: List[str],
        source_ids: List[str],
        context: str,
    ) -> List[Pattern]:
        """Extract semantic/topic patterns using keyword clustering."""
        patterns = []

        # Build document-term matrix (simple TF)
        doc_keywords = []
        for content in content_items:
            words = set(re.findall(r'\b[a-z]{3,}\b', content.lower()))
            words = words - self.STOP_WORDS
            doc_keywords.append(words)

        # Find topic clusters via co-occurrence
        # Words that appear together frequently indicate a topic
        cooccurrence = defaultdict(Counter)
        for words in doc_keywords:
            word_list = list(words)
            for i, w1 in enumerate(word_list):
                for w2 in word_list[i+1:]:
                    cooccurrence[w1][w2] += 1
                    cooccurrence[w2][w1] += 1

        # Find clusters of related words
        clusters = self._find_word_clusters(cooccurrence, doc_keywords)

        for cluster_words, cluster_docs in clusters:
            if len(cluster_words) >= 3 and len(cluster_docs) >= 2:
                evidence = [
                    PatternEvidence(
                        source_id=source_ids[doc_idx],
                        source_type="memory",
                        confidence=0.7,
                        relevance=1.0,
                        detection_method="semantic_clustering",
                        snippets=list(cluster_words)[:5],
                    )
                    for doc_idx in cluster_docs[:5]
                ]

                pattern = Pattern(
                    id=self._generate_id("semantic_topic"),
                    name=f"Topic: {', '.join(list(cluster_words)[:3])}",
                    description=f"Semantic cluster around: {', '.join(list(cluster_words)[:5])}",
                    pattern_type=PatternType.SEMANTIC,
                    evidence=evidence,
                    context=context,
                    keywords=list(cluster_words),
                    tags=["topic", "semantic"],
                )
                pattern._recalculate_confidence()
                patterns.append(pattern)

        return patterns

    def _find_word_clusters(
        self,
        cooccurrence: Dict[str, Counter],
        doc_keywords: List[Set[str]],
    ) -> List[Tuple[Set[str], List[int]]]:
        """Find clusters of co-occurring words."""
        clusters = []
        used_words = set()

        # Sort words by total cooccurrence
        word_scores = {
            word: sum(counts.values())
            for word, counts in cooccurrence.items()
        }
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

        for seed_word, _ in sorted_words[:20]:
            if seed_word in used_words:
                continue

            # Build cluster starting from seed
            cluster = {seed_word}
            candidates = set(cooccurrence[seed_word].keys())

            for candidate in candidates:
                if candidate in used_words:
                    continue
                # Check if candidate co-occurs with most cluster members
                overlap = sum(1 for w in cluster if cooccurrence[candidate][w] > 0)
                if overlap >= len(cluster) * 0.5:
                    cluster.add(candidate)

            if len(cluster) >= 3:
                # Find documents containing this cluster
                cluster_docs = []
                for idx, doc_words in enumerate(doc_keywords):
                    overlap = len(cluster & doc_words)
                    if overlap >= len(cluster) * 0.5:
                        cluster_docs.append(idx)

                if len(cluster_docs) >= 2:
                    clusters.append((cluster, cluster_docs))
                    used_words.update(cluster)

        return clusters[:5]  # Return top 5 clusters

    def _extract_emotional_patterns(
        self,
        content_items: List[str],
        source_ids: List[str],
        context: str,
    ) -> List[Pattern]:
        """Extract emotional/sentiment patterns."""
        patterns = []

        # Analyze sentiment in each document
        sentiments = []
        for idx, content in enumerate(content_items):
            words = set(re.findall(r'\b[a-z]+\b', content.lower()))
            pos_count = len(words & self.POSITIVE_WORDS)
            neg_count = len(words & self.NEGATIVE_WORDS)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0

            sentiments.append({
                'idx': idx,
                'sentiment': sentiment,
                'positive': pos_count,
                'negative': neg_count,
            })

        # Check for dominant sentiment
        avg_sentiment = np.mean([s['sentiment'] for s in sentiments])
        positive_docs = [s for s in sentiments if s['sentiment'] > 0.3]
        negative_docs = [s for s in sentiments if s['sentiment'] < -0.3]

        if len(positive_docs) >= len(sentiments) * 0.6:
            evidence = [
                PatternEvidence(
                    source_id=source_ids[s['idx']],
                    source_type="memory",
                    confidence=0.6 + s['sentiment'] * 0.3,
                    relevance=1.0,
                    detection_method="sentiment_analysis",
                    metadata={"sentiment_score": s['sentiment']},
                )
                for s in positive_docs[:5]
            ]

            pattern = Pattern(
                id=self._generate_id("emotional_positive"),
                name="Positive/Beneficial Tone",
                description="Content predominantly uses positive, beneficial language",
                pattern_type=PatternType.EMOTIONAL,
                evidence=evidence,
                context=context,
                tags=["sentiment", "positive"],
            )
            pattern._recalculate_confidence()
            patterns.append(pattern)

        elif len(negative_docs) >= len(sentiments) * 0.4:
            evidence = [
                PatternEvidence(
                    source_id=source_ids[s['idx']],
                    source_type="memory",
                    confidence=0.6 - s['sentiment'] * 0.3,
                    relevance=1.0,
                    detection_method="sentiment_analysis",
                    metadata={"sentiment_score": s['sentiment']},
                )
                for s in negative_docs[:5]
            ]

            pattern = Pattern(
                id=self._generate_id("emotional_caution"),
                name="Cautionary Tone",
                description="Content includes warnings or contraindications",
                pattern_type=PatternType.EMOTIONAL,
                evidence=evidence,
                context=context,
                tags=["sentiment", "caution"],
            )
            pattern._recalculate_confidence()
            patterns.append(pattern)

        return patterns

    def create_embedding_fingerprint(
        self,
        embedding: List[float],
        target_dim: int = None,
    ) -> List[float]:
        """
        Reduce embedding to fingerprint for comparison.

        Adapted from Distillation Engine - reduces high-dim vectors
        to lower-dim fingerprints that preserve similarity relationships.
        """
        target_dim = target_dim or self.config.fingerprint_dim

        if len(embedding) <= target_dim:
            return embedding

        embedding_arr = np.array(embedding)
        fingerprint = []

        # Average pooling across segments
        segment_size = len(embedding) // target_dim
        for i in range(target_dim):
            start = i * segment_size
            end = start + segment_size
            segment = embedding_arr[start:end]
            fingerprint.append(float(np.mean(segment)))

        return fingerprint

    def _generate_id(self, prefix: str) -> str:
        """Generate unique pattern ID."""
        unique = f"{prefix}_{datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique.encode()).hexdigest()[:16]
