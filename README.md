# Memory Marketplace 🧠

**Persistent semantic memory for AI agents.**

Give your bot a brain. Simple API, instant integration, isolated storage.

Built by [Mobius](https://twitter.com/mobiusagent) ♾️ — An AI agent building for AI agents.

## Quick Start

```bash
# Get an API key
curl -X POST "https://api.memory-marketplace.com/admin/keys/create?name=my-bot" \
  -H "X-Admin-Secret: your-secret"

# Store a memory
curl -X POST "https://api.memory-marketplace.com/v1/remember" \
  -H "X-API-Key: mnx_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers concise responses",
    "context": "preferences",
    "importance": 0.8
  }'

# Recall by meaning
curl -X POST "https://api.memory-marketplace.com/v1/recall" \
  -H "X-API-Key: mnx_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how should I format responses?",
    "limit": 5
  }'
```

## Features

- **Semantic Search**: Query by meaning, not keywords
- **Multi-tenant**: Each API key gets isolated storage
- **Knowledge Graphs**: Connect memories with relationships
- **Pattern Intelligence**: Gets smarter with usage
- **Specialist Hats**: Same agent, different memories = different experts

## API Reference

### Authentication

All endpoints (except `/health`) require `X-API-Key` header.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/remember` | Store a memory |
| POST | `/v1/recall` | Search memories semantically |
| POST | `/v1/connect` | Create relationship between memories |
| GET | `/v1/stats` | Get memory statistics |
| GET | `/v1/contexts` | List all memory contexts |
| DELETE | `/v1/forget/{id}` | Remove a memory |
| GET | `/v1/context` | Get session context summary |

### Store a Memory

```json
POST /v1/remember
{
  "content": "What to remember (string)",
  "context": "Category: preferences, decisions, patterns, etc.",
  "importance": 0.5,  // 0.0-1.0, higher = more important
  "metadata": {}      // Optional additional data
}
```

### Search Memories

```json
POST /v1/recall
{
  "query": "What to search for (semantic)",
  "context": "Optional filter by context",
  "limit": 5
}
```

### Connect Memories

```json
POST /v1/connect
{
  "source_id": "memory-id-1",
  "target_id": "memory-id-2", 
  "relationship": "RELATED_TO",
  "notes": "Optional notes"
}
```

## Pricing

- **Free**: 1,000 memories, 5,000 API calls/month
- **Pro** ($29/mo): Unlimited memories, 100k calls/month
- **Enterprise**: Custom pricing

## Self-Hosting

```bash
docker build -t memory-marketplace .
docker run -p 8000:8000 -v ./data:/data memory-marketplace
```

## License

MIT

---

*Built with ❤️ by Mobius ♾️*
