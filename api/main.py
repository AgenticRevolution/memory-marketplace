"""
Memory Marketplace API

HTTP API wrapper for Memory Nexus - sell memory to AI agents.
Each API key gets isolated memory storage.

Built by Mobius ♾️
"""

import os
import sys
import json
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add memory-nexus to path (bundled in sibling directory)
import pathlib
MEMORY_NEXUS_PATH = str(pathlib.Path(__file__).parent.parent / "memory-nexus")
sys.path.insert(0, MEMORY_NEXUS_PATH)

from core.store import MemoryStore

# =============================================================================
# Configuration
# =============================================================================

BASE_DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "/Users/mobius/projects/memory-marketplace/data")
API_KEYS_FILE = os.path.join(BASE_DATA_DIR, "api_keys.json")

# =============================================================================
# Multi-tenant Store Management
# =============================================================================

_stores: Dict[str, MemoryStore] = {}
_api_keys: Dict[str, dict] = {}


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


def get_store_for_key(api_key: str) -> MemoryStore:
    """Get or create a memory store for an API key."""
    if api_key not in _stores:
        # Hash the key to create a safe directory name
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        data_dir = os.path.join(BASE_DATA_DIR, "tenants", key_hash)
        os.makedirs(data_dir, exist_ok=True)
        
        # Use simple embeddings to avoid heavy torch dependency
        # Can upgrade to "local" for better quality when deployed on GPU
        _stores[api_key] = MemoryStore(
            data_dir=data_dir,
            embedding_provider="simple",  # numpy-only, fast
            enable_patterns=True,
            enable_cache=True,
        )
    return _stores[api_key]


def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate API key from header."""
    if x_api_key not in _api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Update usage stats
    _api_keys[x_api_key]["last_used"] = datetime.now().isoformat()
    _api_keys[x_api_key]["calls"] = _api_keys[x_api_key].get("calls", 0) + 1
    save_api_keys()
    
    return x_api_key


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    load_api_keys()
    yield
    # Cleanup stores on shutdown
    for store in _stores.values():
        try:
            store.close()
        except:
            pass


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Memory Marketplace API",
    description="Persistent semantic memory for AI agents. Give your bot a brain.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class RememberRequest(BaseModel):
    content: str = Field(..., description="What to remember")
    context: str = Field(default="general", description="Category: preferences, decisions, patterns, etc.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0-1.0, higher = more important")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class RecallRequest(BaseModel):
    query: str = Field(..., description="What to search for (semantic search)")
    context: Optional[str] = Field(default=None, description="Filter by context")
    limit: int = Field(default=5, ge=1, le=50, description="Max results")


class ConnectRequest(BaseModel):
    source_id: str = Field(..., description="First memory ID")
    target_id: str = Field(..., description="Second memory ID")
    relationship: str = Field(default="RELATED_TO", description="Relationship type")
    notes: str = Field(default="", description="Optional notes")


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
    """API status."""
    return {
        "service": "Memory Marketplace",
        "status": "operational",
        "version": "0.1.0",
        "docs": "/docs",
        "by": "Mobius ♾️"
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


# =============================================================================
# Admin Endpoints (for key management - TODO: secure this)
# =============================================================================

@app.post("/admin/keys/create")
async def create_api_key(
    name: str,
    email: Optional[str] = None,
    admin_secret: str = Header(..., alias="X-Admin-Secret")
):
    """Create a new API key."""
    # TODO: proper admin auth
    expected_secret = os.environ.get("ADMIN_SECRET", "mobius-dev-secret")
    if admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    
    # Generate key
    api_key = f"mnx_{secrets.token_urlsafe(32)}"
    
    _api_keys[api_key] = {
        "name": name,
        "email": email,
        "created": datetime.now().isoformat(),
        "calls": 0,
        "tier": "free"  # free, pro, enterprise
    }
    save_api_keys()
    
    return {
        "api_key": api_key,
        "name": name,
        "message": "Store this key securely - it won't be shown again"
    }


@app.get("/admin/keys/list")
async def list_api_keys(
    admin_secret: str = Header(..., alias="X-Admin-Secret")
):
    """List all API keys (redacted)."""
    expected_secret = os.environ.get("ADMIN_SECRET", "mobius-dev-secret")
    if admin_secret != expected_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    
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
    store = get_store_for_key(api_key)
    
    metadata = request.metadata.copy()
    metadata["stored_at"] = datetime.now().isoformat()
    
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
    
    # Get sample of memories
    results = store.query("*", limit=100, threshold=0.0)
    
    contexts = {}
    for r in results:
        ctx = r.memory.context
        contexts[ctx] = contexts.get(ctx, 0) + 1
    
    return {"contexts": contexts}


# =============================================================================
# Run with: uvicorn main:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
