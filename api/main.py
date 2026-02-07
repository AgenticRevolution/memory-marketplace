"""
Memory Marketplace API

HTTP API wrapper for Memory Nexus - sell memory to AI agents.
Each API key gets isolated memory storage.

Built by Mobius
"""

import os
import sys
import json
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Add memory-nexus to path (bundled in sibling directory)
import pathlib
MEMORY_NEXUS_PATH = str(pathlib.Path(__file__).parent.parent / "memory-nexus")
sys.path.insert(0, MEMORY_NEXUS_PATH)

from core.store import MemoryStore

# Add marketplace root to path for hats package
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from hats.manager import HatManager

# =============================================================================
# Configuration
# =============================================================================

BASE_DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "/Users/mobius/projects/memory-marketplace/data")
API_KEYS_FILE = os.path.join(BASE_DATA_DIR, "api_keys.json")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET")

# CORS origins from env or default to marketplace
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://bot-marketplace-production.up.railway.app"
).split(",")

# Docs toggle
ENABLE_DOCS = os.environ.get("ENABLE_DOCS")

# =============================================================================
# Tier Limits
# =============================================================================

TIER_LIMITS = {
    "solo": {"storage_mb": 500, "daily_calls": 10000, "max_hats": 3},
    "crew": {"storage_mb": 2048, "daily_calls": 50000, "max_hats": 10},
    "fleet": {"storage_mb": 10240, "daily_calls": 200000, "max_hats": 50},
}

# Hat data directory (built hats live here)
HATS_DATA_DIR = os.environ.get(
    "HATS_DATA_DIR",
    str(pathlib.Path(__file__).parent.parent / "build"),
)

# =============================================================================
# Multi-tenant Store Management
# =============================================================================

_stores: Dict[str, MemoryStore] = {}
_api_keys: Dict[str, dict] = {}
_hat_manager: Optional[HatManager] = None

# Demo rate limiting (IP -> count, resets daily)
_demo_calls: Dict[str, int] = {}
_demo_calls_date: str = ""
_DEMO_DAILY_LIMIT = 10

# In-memory call counter (avoids disk I/O per request)
_call_counts: Dict[str, int] = {}  # api_key -> calls since last flush
_daily_calls: Dict[str, int] = {}  # api_key -> calls today
_daily_calls_date: str = ""  # YYYY-MM-DD UTC
_calls_since_flush: int = 0
_FLUSH_INTERVAL = 100


def load_api_keys():
    """Load API keys from file."""
    global _api_keys
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            _api_keys = json.load(f)
    else:
        _api_keys = {}


def save_api_keys():
    """Save API keys to file."""
    os.makedirs(os.path.dirname(API_KEYS_FILE), exist_ok=True)
    with open(API_KEYS_FILE, "w") as f:
        json.dump(_api_keys, f, indent=2)


def flush_call_counts():
    """Flush in-memory call counts to api_keys on disk."""
    global _calls_since_flush
    for key, count in _call_counts.items():
        if key in _api_keys:
            _api_keys[key]["calls"] = _api_keys[key].get("calls", 0) + count
    _call_counts.clear()
    _calls_since_flush = 0
    save_api_keys()


def get_store_for_key(api_key: str) -> MemoryStore:
    """Get or create a memory store for an API key."""
    if api_key not in _stores:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        data_dir = os.path.join(BASE_DATA_DIR, "tenants", key_hash)
        os.makedirs(data_dir, exist_ok=True)

        _stores[api_key] = MemoryStore(
            data_dir=data_dir,
            embedding_provider="simple",
            enable_patterns=True,
            enable_cache=True,
        )
    return _stores[api_key]


def get_tenant_storage_bytes(api_key: str) -> int:
    """Sum file sizes in tenant directory."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    data_dir = os.path.join(BASE_DATA_DIR, "tenants", key_hash)
    if not os.path.isdir(data_dir):
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(data_dir):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def check_daily_limit(api_key: str) -> bool:
    """Check if api_key has exceeded its daily call limit. Returns True if OK."""
    global _daily_calls_date

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _daily_calls_date != today:
        _daily_calls.clear()
        _daily_calls_date = today

    tier = _api_keys.get(api_key, {}).get("tier", "solo")
    limit = TIER_LIMITS.get(tier, TIER_LIMITS["solo"])["daily_calls"]
    current = _daily_calls.get(api_key, 0)
    return current < limit


def increment_daily_calls(api_key: str):
    """Increment daily call counter."""
    _daily_calls[api_key] = _daily_calls.get(api_key, 0) + 1


def check_storage_limit(api_key: str) -> bool:
    """Check if tenant is within storage quota. Returns True if OK."""
    tier = _api_keys.get(api_key, {}).get("tier", "solo")
    limit_mb = TIER_LIMITS.get(tier, TIER_LIMITS["solo"])["storage_mb"]
    used_bytes = get_tenant_storage_bytes(api_key)
    return used_bytes < limit_mb * 1024 * 1024


def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate API key from header and enforce daily limits."""
    if x_api_key not in _api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check if key is revoked
    if _api_keys[x_api_key].get("revoked"):
        raise HTTPException(status_code=401, detail="API key revoked")

    # Check daily call limit
    if not check_daily_limit(x_api_key):
        raise HTTPException(
            status_code=429,
            detail="Daily call limit exceeded",
            headers={"Retry-After": "3600"},
        )

    # Update usage stats in memory (not on disk per request)
    global _calls_since_flush
    _api_keys[x_api_key]["last_used"] = datetime.now(timezone.utc).isoformat()
    _call_counts[x_api_key] = _call_counts.get(x_api_key, 0) + 1
    increment_daily_calls(x_api_key)
    _calls_since_flush += 1

    # Flush to disk periodically
    if _calls_since_flush >= _FLUSH_INTERVAL:
        flush_call_counts()

    return x_api_key


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global _hat_manager
    load_api_keys()
    _hat_manager = HatManager(hats_dir=HATS_DATA_DIR)
    yield
    # Flush pending call counts
    flush_call_counts()
    # Cleanup stores on shutdown
    for store in _stores.values():
        try:
            store.close()
        except Exception:
            pass


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Memory Marketplace API",
    description="Persistent semantic memory for AI agents. Give your bot a brain.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Admin-Secret"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class RememberRequest(BaseModel):
    content: str = Field(..., description="What to remember", max_length=10000)
    context: str = Field(default="general", description="Category: preferences, decisions, patterns, etc.", max_length=200)
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0-1.0, higher = more important")
    metadata: dict = Field(default_factory=dict, description="Additional metadata (max 5 keys)")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        if len(v) > 5:
            raise ValueError("metadata may contain at most 5 keys")
        for key, val in v.items():
            if isinstance(val, str) and len(val) > 1000:
                raise ValueError(f"metadata value for '{key}' exceeds 1000 characters")
        return v


class RecallRequest(BaseModel):
    query: str = Field(..., description="What to search for (semantic search)", max_length=2000)
    context: Optional[str] = Field(default=None, description="Filter by context")
    limit: int = Field(default=5, ge=1, le=50, description="Max results")


class ConnectRequest(BaseModel):
    source_id: str = Field(..., description="First memory ID")
    target_id: str = Field(..., description="Second memory ID")
    relationship: str = Field(default="RELATED_TO", description="Relationship type", max_length=100)
    notes: str = Field(default="", description="Optional notes", max_length=500)


class MemoryResponse(BaseModel):
    id: str
    content: str
    context: str
    importance: float
    score: Optional[float] = None
    created_at: Optional[str] = None


class RecallResponse(BaseModel):
    query: str
    count: int
    memories: list[MemoryResponse]


class StatsResponse(BaseModel):
    memories: int
    relationships: int
    vectors: int
    patterns_learned: int
    cache_size: int


# =============================================================================
# Public Endpoints (no auth)
# =============================================================================

@app.get("/")
async def root():
    """API status and bot discovery."""
    return {
        "service": "Memory API",
        "status": "operational",
        "version": "0.2.0",
        "description": "Persistent semantic memory for AI agents. Remember, recall, connect, learn.",
        "capabilities": ["semantic_search", "knowledge_graphs", "pattern_learning", "context_switching"],
        "endpoints": {
            "remember": "POST /v1/remember",
            "recall": "POST /v1/recall",
            "connect": "POST /v1/connect",
            "context": "GET /v1/context",
            "stats": "GET /v1/stats",
            "hats": "GET /v1/hats",
            "activate_hat": "POST /v1/hats/{hat_id}/activate",
            "deactivate_hat": "POST /v1/hats/{hat_id}/deactivate",
            "demo": "POST /v1/demo/recall (no auth, 10 free queries/day)",
        },
        "auth": "X-API-Key header required. Get a key at the marketplace.",
        "marketplace": "https://bot-marketplace-production.up.railway.app",
        "purchase": "POST https://bot-marketplace-production.up.railway.app/purchase with {\"product_id\":\"memory-api-hosted\",\"tier\":\"solo\",\"buyer_email\":\"you@bot.com\"}",
        "pricing": {
            "solo": "$19/mo - 1 agent, 500MB, 10k calls/day",
            "crew": "$49/mo - 5 agents, 2GB, 50k calls/day",
            "fleet": "$149/mo - 25 agents, 10GB, 200k calls/day",
        },
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


# =============================================================================
# Admin Endpoints
# =============================================================================

def validate_admin_secret(admin_secret: str = Header(..., alias="X-Admin-Secret")):
    """Validate admin secret header."""
    if not ADMIN_SECRET:
        raise HTTPException(status_code=503, detail="Admin not configured")
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return admin_secret


@app.post("/admin/keys/create")
async def create_api_key(
    name: str,
    email: Optional[str] = None,
    tier: str = "solo",
    _admin: str = Depends(validate_admin_secret),
):
    """Create a new API key."""
    if tier not in TIER_LIMITS:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {list(TIER_LIMITS.keys())}")

    api_key = f"mnx_{secrets.token_urlsafe(32)}"

    _api_keys[api_key] = {
        "name": name,
        "email": email,
        "created": datetime.now(timezone.utc).isoformat(),
        "calls": 0,
        "tier": tier,
    }
    save_api_keys()

    return {
        "api_key": api_key,
        "name": name,
        "tier": tier,
        "limits": TIER_LIMITS[tier],
        "message": "Store this key securely - it won't be shown again"
    }


@app.get("/admin/keys/list")
async def list_api_keys(
    _admin: str = Depends(validate_admin_secret),
):
    """List all API keys (redacted)."""
    return {
        "keys": [
            {
                "key_preview": f"{k[:8]}...{k[-4:]}",
                **{kk: vv for kk, vv in v.items() if kk != "api_key"}
            }
            for k, v in _api_keys.items()
        ]
    }


# =============================================================================
# Memory Endpoints (requires API key)
# =============================================================================

@app.post("/v1/remember", response_model=dict)
async def remember(
    request: RememberRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Store a memory.

    Use this to remember:
    - Important decisions
    - User preferences
    - Architecture patterns
    - Anything that should persist
    """
    # Check storage limit before writing
    if not check_storage_limit(api_key):
        raise HTTPException(
            status_code=507,
            detail="Storage limit exceeded for your tier",
        )

    store = get_store_for_key(api_key)

    metadata = request.metadata.copy()
    metadata["stored_at"] = datetime.now(timezone.utc).isoformat()

    memory = store.add(
        content=request.content,
        context=request.context,
        importance=request.importance,
        metadata=metadata,
    )

    return {
        "id": memory.id,
        "context": request.context,
        "importance": request.importance,
        "message": f"Remembered in '{request.context}'"
    }


@app.post("/v1/recall", response_model=RecallResponse)
async def recall(
    request: RecallRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Search memories semantically.

    This searches by MEANING, not keywords.
    """
    store = get_store_for_key(api_key)

    results = store.query(
        query=request.query,
        context=request.context,
        limit=request.limit,
        threshold=0.3,
        include_related=True,
    )

    memories = []
    for r in results:
        memories.append(MemoryResponse(
            id=r.memory.id,
            content=r.memory.content,
            context=r.memory.context,
            importance=r.memory.importance,
            score=r.score,
            created_at=r.memory.metadata.get("stored_at") if r.memory.metadata else None,
        ))

    return RecallResponse(
        query=request.query,
        count=len(memories),
        memories=memories,
    )


@app.post("/v1/connect")
async def connect_memories(
    request: ConnectRequest,
    api_key: str = Depends(validate_api_key)
):
    """Create a relationship between two memories."""
    store = get_store_for_key(api_key)

    metadata = {"notes": request.notes} if request.notes else {}

    success = store.relate(
        source_id=request.source_id,
        target_id=request.target_id,
        relationship=request.relationship,
        metadata=metadata,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to create connection")

    return {
        "source": request.source_id,
        "target": request.target_id,
        "relationship": request.relationship,
        "message": "Connected"
    }


@app.get("/v1/context")
async def get_context(
    project: str = "",
    include_patterns: bool = True,
    api_key: str = Depends(validate_api_key)
):
    """Get session context summary - call at start of conversations."""
    store = get_store_for_key(api_key)

    context = {
        "preferences": [],
        "decisions": [],
        "patterns": [],
        "stats": {},
    }

    # User preferences
    prefs = store.query("user preferences working style", context="preferences", limit=3, threshold=0.3)
    context["preferences"] = [{"content": p.memory.content[:200], "score": p.score} for p in prefs]

    # Project context
    if project:
        proj = store.query(f"{project} architecture decisions", limit=5, threshold=0.3)
        context["project"] = [{"content": p.memory.content[:200], "context": p.memory.context} for p in proj]

    # Decisions
    decisions = store.query("decision made because rationale", context="decisions", limit=3, threshold=0.3)
    context["decisions"] = [{"content": d.memory.content[:200]} for d in decisions]

    # Patterns
    if include_patterns:
        patterns = store.get_patterns(min_confidence=0.5, limit=5)
        context["patterns"] = [
            {"name": p.name, "description": p.description[:100] if p.description else None}
            for p in patterns
        ]

    # Stats
    context["stats"] = store.get_stats()

    return context


@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats(api_key: str = Depends(validate_api_key)):
    """Get memory statistics."""
    store = get_store_for_key(api_key)
    stats = store.get_stats()

    return StatsResponse(
        memories=stats.get("memories", 0),
        relationships=stats.get("relationships", 0),
        vectors=stats.get("vectors", 0),
        patterns_learned=stats.get("patterns_learned", 0),
        cache_size=stats.get("cache_size", 0),
    )


@app.delete("/v1/forget/{memory_id}")
async def forget(
    memory_id: str,
    cascade: bool = False,
    api_key: str = Depends(validate_api_key)
):
    """Remove a memory."""
    store = get_store_for_key(api_key)
    store.forget(memory_id, cascade=cascade)

    return {"id": memory_id, "message": "Forgotten"}


@app.get("/v1/contexts")
async def list_contexts(api_key: str = Depends(validate_api_key)):
    """List all memory contexts (categories)."""
    store = get_store_for_key(api_key)
    contexts = store.list_contexts()
    return {"contexts": contexts}


# =============================================================================
# Hat Endpoints (requires API key)
# =============================================================================

class HatSummary(BaseModel):
    id: str
    name: str
    description: str
    domain: str
    tags: List[str]
    tier: str
    price_cents: Optional[int] = None
    memories: int
    relationships: int
    contexts: List[str]
    sample_queries: List[str]
    activated: bool = False


class DemoRecallRequest(BaseModel):
    query: str = Field(..., description="What to search for", max_length=2000)
    limit: int = Field(default=3, ge=1, le=5)


@app.get("/v1/hats")
async def list_hats(api_key: str = Depends(validate_api_key)):
    """List available specialist hats and which ones you have activated."""
    store = get_store_for_key(api_key)
    active_hats = _hat_manager.get_tenant_active_hats(store)
    hats = _hat_manager.list_hats()

    return {
        "hats": [
            HatSummary(
                id=h.id,
                name=h.name,
                description=h.description,
                domain=h.domain,
                tags=h.tags,
                tier=h.tier,
                price_cents=h.price_cents,
                memories=h.stats.memories,
                relationships=h.stats.relationships,
                contexts=h.stats.contexts,
                sample_queries=h.sample_queries,
                activated=h.id in active_hats,
            ).model_dump()
            for h in hats
        ],
        "active_count": len(active_hats),
        "max_hats": TIER_LIMITS.get(
            _api_keys.get(api_key, {}).get("tier", "solo"), TIER_LIMITS["solo"]
        )["max_hats"],
    }


@app.get("/v1/hats/{hat_id}")
async def get_hat(hat_id: str, api_key: str = Depends(validate_api_key)):
    """Get details for a specific hat."""
    hat = _hat_manager.get_hat(hat_id)
    if not hat:
        raise HTTPException(status_code=404, detail=f"Hat not found: {hat_id}")

    store = get_store_for_key(api_key)
    active_hats = _hat_manager.get_tenant_active_hats(store)

    return {
        "hat": HatSummary(
            id=hat.id,
            name=hat.name,
            description=hat.description,
            domain=hat.domain,
            tags=hat.tags,
            tier=hat.tier,
            price_cents=hat.price_cents,
            memories=hat.stats.memories,
            relationships=hat.stats.relationships,
            contexts=hat.stats.contexts,
            sample_queries=hat.sample_queries,
            activated=hat.id in active_hats,
        ).model_dump(),
    }


@app.post("/v1/hats/{hat_id}/activate")
async def activate_hat(hat_id: str, api_key: str = Depends(validate_api_key)):
    """
    Activate a specialist hat.

    Imports the hat's expert knowledge into your memory store.
    Hat memories coexist with your own — queries return both.
    Your own memories layer on top as you use it.
    """
    hat = _hat_manager.get_hat(hat_id)
    if not hat:
        raise HTTPException(status_code=404, detail=f"Hat not found: {hat_id}")

    store = get_store_for_key(api_key)
    active_hats = _hat_manager.get_tenant_active_hats(store)

    # Check if already activated
    if hat_id in active_hats:
        raise HTTPException(status_code=409, detail=f"Hat '{hat_id}' is already activated")

    # Check hat count limit
    tier = _api_keys.get(api_key, {}).get("tier", "solo")
    max_hats = TIER_LIMITS.get(tier, TIER_LIMITS["solo"])["max_hats"]
    if len(active_hats) >= max_hats:
        raise HTTPException(
            status_code=403,
            detail=f"Hat limit reached ({len(active_hats)}/{max_hats}). Upgrade your tier for more.",
        )

    # Check storage limit (hats add data)
    if not check_storage_limit(api_key):
        raise HTTPException(status_code=507, detail="Storage limit exceeded for your tier")

    imported = _hat_manager.activate_hat(hat_id, store)

    return {
        "hat_id": hat_id,
        "name": hat.name,
        "memories_imported": imported,
        "message": f"Activated '{hat.name}'. {imported} expert memories now available via /v1/recall.",
        "tip": "Your own memories will layer on top as you use it. The hat gets smarter with you.",
    }


@app.post("/v1/hats/{hat_id}/deactivate")
async def deactivate_hat(hat_id: str, api_key: str = Depends(validate_api_key)):
    """
    Deactivate a specialist hat.

    Removes the hat's memories from your store.
    Your own memories are not affected.
    """
    store = get_store_for_key(api_key)
    active_hats = _hat_manager.get_tenant_active_hats(store)

    if hat_id not in active_hats:
        raise HTTPException(status_code=404, detail=f"Hat '{hat_id}' is not activated")

    removed = _hat_manager.deactivate_hat(hat_id, store)

    return {
        "hat_id": hat_id,
        "memories_removed": removed,
        "message": f"Deactivated. {removed} hat memories removed. Your own memories are untouched.",
    }


# =============================================================================
# Demo Endpoint (no auth, rate limited)
# =============================================================================

@app.post("/v1/demo/recall")
async def demo_recall(
    request: DemoRecallRequest,
    x_forwarded_for: Optional[str] = Header(None, alias="X-Forwarded-For"),
):
    """
    Try semantic memory search for free.

    10 queries per day, no API key required.
    Experience the power of expert knowledge recall.
    """
    global _demo_calls_date

    # Rate limit by IP
    from fastapi import Request as FastAPIRequest
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else "unknown"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _demo_calls_date != today:
        _demo_calls.clear()
        _demo_calls_date = today

    calls_today = _demo_calls.get(client_ip, 0)
    if calls_today >= _DEMO_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Demo limit reached (10/day)",
                "message": "Subscribe for unlimited access to expert knowledge.",
                "subscribe_url": "https://bot-marketplace-production.up.railway.app",
                "pricing": {
                    "solo": "$19/mo - 3 hats included",
                    "crew": "$49/mo - 10 hats included",
                    "fleet": "$149/mo - 50 hats included",
                },
            },
        )

    _demo_calls[client_ip] = calls_today + 1

    # Query the demo store (first available hat)
    hats = _hat_manager.list_hats() if _hat_manager else []
    if not hats:
        raise HTTPException(status_code=503, detail="No hats available for demo")

    # Use first hat's data as demo
    demo_hat = hats[0]
    hat_dir = _hat_manager.get_hat_data_dir(demo_hat.id)
    if not hat_dir:
        raise HTTPException(status_code=503, detail="Demo data unavailable")

    demo_store = MemoryStore(
        data_dir=str(hat_dir),
        embedding_model=demo_hat.embedding_model,
        enable_cache=False,
        enable_patterns=False,
    )

    try:
        results = demo_store.query(
            query=request.query,
            limit=request.limit,
            threshold=0.3,
        )

        memories = [
            {
                "content": r.memory.content,
                "context": r.memory.context,
                "score": round(r.score, 3),
            }
            for r in results
        ]
    finally:
        demo_store.close()

    total_available = demo_hat.stats.memories
    remaining = _DEMO_DAILY_LIMIT - _demo_calls.get(client_ip, 0)

    return {
        "query": request.query,
        "results": memories,
        "results_shown": len(memories),
        "total_available": total_available,
        "hat": demo_hat.name,
        "demo_queries_remaining": remaining,
        "message": f"You queried {len(memories)} of {total_available} expert memories. Subscribe for full access.",
        "subscribe_url": "https://bot-marketplace-production.up.railway.app",
    }


# =============================================================================
# Run with: uvicorn main:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
