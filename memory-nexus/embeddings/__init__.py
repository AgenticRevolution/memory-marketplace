"""
Embedding providers for Memory Nexus Lite.

Providers:
- "simple": Zero-dependency hash-based embeddings (just numpy)
- "local": Sentence-transformers (better quality, requires torch)
- "openai": OpenAI API (highest quality, requires API key)
"""

from .simple import SimpleEmbeddings

__all__ = ["SimpleEmbeddings", "create_embeddings"]


def create_embeddings(provider: str = "local", **kwargs):
    """
    Factory function to create embedding provider.

    Args:
        provider: "local" (default), "simple", or "openai"
        **kwargs: model_name, api_key, etc.
    """
    if provider == "simple":
        return SimpleEmbeddings(**kwargs)
    elif provider == "local":
        try:
            from .local import LocalEmbeddings
            return LocalEmbeddings(**kwargs)
        except ImportError:
            import sys
            print(
                "Warning: sentence-transformers not installed. "
                "Falling back to simple embeddings.",
                file=sys.stderr
            )
            return SimpleEmbeddings(**kwargs)
    elif provider == "openai":
        from .openai_embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
