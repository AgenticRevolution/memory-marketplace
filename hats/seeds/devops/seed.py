"""
DevOps Expert Hat - Seed Script

Loads all knowledge data files and seeds them into a MemoryStore.
Creates cross-context relationships between related knowledge.

Usage:
    # Via HatBuilder:
    builder = HatBuilder(hat_id="devops-expert", ...)
    builder.build_from_seed("hats/seeds/devops/seed.py")

    # Or standalone:
    python hats/seeds/devops/seed.py --output ./build/devops-expert
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Cross-context relationships to create after seeding
# Format: (source_keyword, target_keyword, relationship_type)
# We'll find memories containing these keywords and link them
RELATIONSHIPS = [
    # Docker <-> Railway
    ("Dockerfile", "Railway uses Nixpacks", "RELATED_TO"),
    ("multi-stage Docker builds", "Railway build failed with out of memory", "SOLVES"),
    (".dockerignore", "Railway deploy stuck at 'building'", "SOLVES"),
    ("Docker base image versions", "Nixpacks as the default builder", "RELATED_TO"),

    # Docker <-> CI/CD
    ("multi-stage Docker builds", "GitHub Actions workflow files", "USED_IN"),
    ("Docker base image versions", "matrix strategy to test across multiple", "RELATED_TO"),

    # Docker <-> Security
    ("Run containers as non-root", "Never commit secrets to git", "RELATED_TO"),
    ("Docker security: Run containers as non-root", "HTTPS everywhere", "RELATED_TO"),

    # Railway <-> Debugging
    ("Railway requires a PORT environment variable", "App crashes on deploy but works locally", "RELATED_TO"),
    ("Railway health checks", "502 Bad Gateway after deploy", "SOLVES"),
    ("Railway persistent storage", "Logs disappear on redeploy", "RELATED_TO"),
    ("Railway sleep/idle", "High latency on first request", "RELATED_TO"),

    # CI/CD <-> Security
    ("Never put secrets in GitHub Actions", "Never commit secrets to git", "REINFORCES"),
    ("GitHub Actions: Use environment protection", "Protect admin endpoints", "RELATED_TO"),

    # CI/CD <-> Debugging
    ("CI test flakiness", "npm install fails in CI", "RELATED_TO"),
    ("GitHub Actions debugging", "CI test flakiness", "SOLVES"),

    # Architecture <-> Railway
    ("Start with a monolith", "Render vs Railway vs Fly.io", "RELATED_TO"),
    ("Health check endpoints should verify", "Railway health checks", "RELATED_TO"),

    # Architecture <-> Security
    ("Use a reverse proxy", "HTTPS everywhere", "RELATED_TO"),
    ("12-factor app principles", "Environment variable best practices", "RELATED_TO"),

    # Architecture <-> Debugging
    ("Connection pooling is critical", "Connection timeout to database", "SOLVES"),
    ("Graceful shutdown", "Process killed with exit code 137", "RELATED_TO"),
    ("Structured logging", "Debugging production without SSH", "RELATED_TO"),

    # Debugging <-> Security
    ("CORS errors in production", "CORS configuration: In production", "RELATED_TO"),
    ("Webhook not receiving events", "Stripe webhook signature verification", "RELATED_TO"),
    ("Rate limiting yourself", "Rate limiting protects against abuse", "RELATED_TO"),

    # Docker <-> Debugging
    ("PYTHONDONTWRITEBYTECODE", "App works in dev, crashes in production", "RELATED_TO"),
    ("COPY requirements.txt", "App works in dev, crashes in production with import", "SOLVES"),
    ("Docker HEALTHCHECK instruction", "502 Bad Gateway after deploy", "SOLVES"),
    ("docker logs and docker exec", "Debugging production without SSH", "RELATED_TO"),
    ("Docker container exits immediately", "Exit code 1 on deploy", "RELATED_TO"),

    # Docker <-> Architecture
    ("multi-stage Docker builds", "Blue-green deployment", "USED_IN"),
    ("Docker Compose for local development", "Start with a monolith", "RELATED_TO"),
    ("Docker base image versions", "Dependency vulnerability scanning", "RELATED_TO"),

    # Railway <-> Security
    ("Railway environment variables", "Environment variable best practices", "RELATED_TO"),
    ("Railway requires a PORT", "12-factor app principles", "RELATED_TO"),
    ("Railway custom domains", "SSL certificate errors", "RELATED_TO"),

    # Railway <-> CI/CD
    ("Railway deploys automatically on git push", "GitHub Actions: For deploy-on-push-to-main", "RELATED_TO"),
    ("Railway CLI", "GitHub Actions workflow files", "USED_IN"),
    ("Railway webhook deploys", "Semantic versioning in CI", "RELATED_TO"),

    # CI/CD <-> Architecture
    ("CI/CD pipeline best practice: lint > test > build", "Health check endpoints should verify", "RELATED_TO"),
    ("GitHub Actions: Store build artifacts", "Blue-green deployment", "USED_IN"),
    ("Pre-commit hooks with Husky", "Dependency vulnerability scanning", "RELATED_TO"),

    # Security <-> Architecture
    ("API authentication options", "API versioning", "RELATED_TO"),
    ("Input validation at the boundary", "For APIs: Return appropriate HTTP status", "RELATED_TO"),
    ("Use helmet", "Use a reverse proxy", "RELATED_TO"),
    ("Logging security: Never log secrets", "Structured logging", "RELATED_TO"),

    # Debugging <-> Architecture
    ("ECONNREFUSED errors in production", "Choose your database based on access", "RELATED_TO"),
    ("Memory leak detection", "Horizontal scaling means adding more", "RELATED_TO"),
    ("Deployment succeeds but API returns 404", "Graceful shutdown", "RELATED_TO"),
    ("File uploads fail in production", "Queue systems", "RELATED_TO"),

    # More Docker links
    ("Use Alpine-based images", "Use multi-stage Docker builds", "RELATED_TO"),
    ("Pin Docker base image", "Dependency vulnerability scanning", "RELATED_TO"),
    ("Docker Compose for local development", "Health check endpoints should verify", "RELATED_TO"),
    ("docker logs and docker exec", "Logs disappear on redeploy", "RELATED_TO"),

    # More Railway links
    ("Railway Postgres", "Connection pooling is critical", "RELATED_TO"),
    ("Railway pricing", "Railway sleep/idle", "RELATED_TO"),
]


def seed(store):
    """
    Seed the DevOps Expert hat.

    Called by HatBuilder.build_from_seed() with a fresh MemoryStore.
    """
    memory_map = {}  # keyword -> memory_id for relationship linking

    # Load all knowledge files
    data_files = sorted(DATA_DIR.glob("*.json"))
    total = 0

    for data_file in data_files:
        data = json.loads(data_file.read_text())
        context = data["context"]

        for item in data["knowledge"]:
            mem = store.add(
                content=item["content"],
                context=context,
                importance=item.get("importance", 0.7),
                metadata=item.get("metadata", {}),
            )

            # Index by content snippet for relationship linking
            # Use first 60 chars as a key
            memory_map[item["content"][:80]] = mem.id
            total += 1

        logger.info(f"Loaded {len(data['knowledge'])} memories from {data_file.name} (context: {context})")

    logger.info(f"Total memories loaded: {total}")

    # Create cross-context relationships
    linked = 0
    for source_kw, target_kw, rel_type in RELATIONSHIPS:
        source_id = _find_memory(memory_map, source_kw)
        target_id = _find_memory(memory_map, target_kw)

        if source_id and target_id:
            try:
                store.relate(source_id, target_id, rel_type)
                linked += 1
            except Exception as e:
                logger.debug(f"Failed to link '{source_kw[:30]}' -> '{target_kw[:30]}': {e}")
        else:
            if not source_id:
                logger.debug(f"No memory found for source: '{source_kw[:50]}'")
            if not target_id:
                logger.debug(f"No memory found for target: '{target_kw[:50]}'")

    logger.info(f"Created {linked} cross-context relationships")


def _find_memory(memory_map: dict, keyword: str) -> str:
    """Find a memory ID by matching keyword against content keys."""
    # Exact prefix match first
    for key, mem_id in memory_map.items():
        if keyword in key:
            return mem_id
    # Fuzzy: check if keyword appears anywhere in stored keys
    keyword_lower = keyword.lower()
    for key, mem_id in memory_map.items():
        if keyword_lower in key.lower():
            return mem_id
    return None


if __name__ == "__main__":
    import argparse
    import sys

    # Set up path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "memory-nexus"))

    from hats.builder import HatBuilder

    parser = argparse.ArgumentParser(description="Build DevOps Expert hat")
    parser.add_argument("--output", default="./build/devops-expert", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    builder = HatBuilder(
        hat_id="devops-expert",
        hat_name="DevOps Deployment Expert",
        output_dir=args.output,
        author="Mobius",
        description="Railway, Docker, CI/CD, deployment debugging, architecture, and security. 100+ expert memories with cross-domain relationships.",
        domain="devops",
        tags=["docker", "railway", "cicd", "debugging", "architecture", "security", "deployment"],
        tier="standard",
        price_cents=1499,
        sample_queries=[
            "Railway deploy failed with exit code 1",
            "Dockerfile for Python FastAPI app",
            "GitHub Actions CI pipeline for Node.js",
            "502 Bad Gateway after deploy",
            "How to manage secrets in production",
            "Docker container keeps getting killed",
        ],
    )

    result = builder.build_from_seed(str(Path(__file__)))
    print(f"\nBuilt DevOps Expert hat:")
    print(f"  Memories: {result['memories']}")
    print(f"  Relationships: {result['relationships']}")
    print(f"  Patterns: {result['patterns']}")
    print(f"  Output: {result['output_dir']}")
