# mem0-temporal-hygiene

**Temporal context, Soft-Deletes ("Memory Git"), deterministic merges, and hash-based caching for Mem0 OSS + Qdrant setups.**

> Mem0 OSS stores user facts as flat vectors in Qdrant. Over time, contradictory and duplicate facts accumulate — the system has no built-in mechanism to resolve conflicts, track version history, or prune outdated information. This plugin and maintenance hygiene script solve that problem by introducing **soft-deletes, time-decay aware filtering, deterministic deduplication merges, and cache optimizations**.

---

## Technical Features

### 1. "Memory Git" (Soft-Deletes & Versioning)
Instead of physically purging old memories (which causes irrecoverable loss of historically valuable facts or logs), we employ a **Soft-Delete** strategy:
- Outdated memories are marked with metadata `status: superseded` and linked to the newer winner memory ID (`superseded_by: winner_id`).
- Manually deleted memories are marked as `status: deleted`.
- This ensures full auditability of the agent's knowledge evolution, functioning similarly to git version history.

### 2. Low-latency DB-level Filtering
The standard `prefetch`, `mem0_profile`, and `mem0_search` operations fetch points using Qdrant's field search filters:
```json
{
  "must_not": [
    { "key": "status", "match": { "value": "superseded" } },
    { "key": "status", "match": { "value": "deleted" } }
  ]
}
```
Filtering happens directly in Qdrant, preventing obsolete facts from cluttering the agent's LLM context window, while saving memory database resources.

### 3. Hybrid Merge (Deterministic + LLM-driven)
- **Deterministic Merge (Cosine $\ge$ 0.95):** When vector similarity is extremely high, facts are considered identical. The script automatically soft-deletes the older items and retains the newest one, bypassing LLM API calls entirely.
- **LLM-driven Cleanups (Cosine 0.82 - 0.94):** Outdated or slightly overlapping facts are bundled together and sent to the LLM to resolve semantic contradictions (newer timestamp wins) or merge details into a consolidated memory.

### 4. Hash-based Cache
During the weekly run, groups of similar facts are hashed based on their IDs, timestamps, and contents. If a cluster has not changed since the last execution, the script fetches the resolution from a local JSON cache, saving LLM tokens.

### 5. Agent-Side CRUD Tools
Standard Mem0 Hermes plugins lack direct modification APIs. This plugin registers two new tools:
- `mem0_update(memory_id, content)` — updates a specific memory and re-embeds it.
- `mem0_delete(memory_id)` — soft-deletes a memory by UUID.

---

## How It Works

### Temporal Context Flow
```
Before (standard mem0-oss):
  prefetch → "- User prefers dark mode"
              (no date, no ID, no way to know if this is current)

After (this plugin):
  prefetch → "- [2026-05-15] User prefers dark mode [ID: abc123]"
  prefetch → "- [2026-06-01] User switched to light mode [ID: def456]"
              (LLM sees dates, picks the June fact, can delete the May fact)
```

### Soft-Delete Execution (Memory Hygiene)
```
1. Fetch all vector points from Qdrant where status is not "superseded" or "deleted"
2. Group points by cosine similarity (threshold ≥ 0.82)
3. If similarity ≥ 0.95:
   → Deterministically mark older points as "superseded"
4. If similarity is between 0.82 and 0.94:
   → Calculate cluster hash.
   → If cached, apply saved decision.
   → Else, call LLM to resolve compromises, updates, and soft-deletions.
5. Apply metadata updates:
   - For deleted items: {"status": "superseded", "superseded_by": "winner_id"}
   - For updated/merged items: Keep active status and write consolidated text.
```

---

## Installation

### Prerequisites
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) or any compatible framework
- Qdrant running locally or remotely (default port `6333`)
- Mem0 OSS Python package (`pip install mem0ai`)
- An OpenAI-compatible LLM endpoint (e.g., OmniRoute, vLLM, ollama)

### Plugin Installation
```bash
# Back up your existing plugin
cp -r ~/.hermes/plugins/mem0-oss ~/.hermes/plugins/mem0-oss.bak

# Copy the enhanced plugin
cp plugin/__init__.py ~/.hermes/plugins/mem0-oss/__init__.py

# Restart your Hermes gateway
systemctl restart hermes-gateway
```

### Hygiene Script Installation
```bash
# Copy the script
cp scripts/memory-hygiene.py ~/.hermes/scripts/memory-hygiene.py
chmod +x ~/.hermes/scripts/memory-hygiene.py

# Test run
python3 ~/.hermes/scripts/memory-hygiene.py
```

### Scheduling
Configure a cron task on your machine. For Hermes agents, add a cronjob configuration or systemd timer.

Example crontab entry (runs every Sunday at 2 AM):
```cron
0 2 * * 0 root /usr/bin/python3 /root/.hermes/scripts/memory-hygiene.py >> /var/log/memory-hygiene.log 2>&1
```

---

## Configuration

The plugin reads its configuration from `~/.hermes/mem0_oss.json`:
```json
{
  "llm_model": "your-model-name",
  "llm_base_url": "http://localhost:20130/v1",
  "llm_api_key": "your-api-key",
  "embedder_provider": "openai",
  "embedder_model": "nvidia/nv-embedqa-e5-v5",
  "embedder_base_url": "http://localhost:20130/v1",
  "embedder_api_key": "your-api-key",
  "embedding_dims": 1024,
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection_name": "hermes_default",
  "user_id": "default"
}
```

---

## Background & Related Work
- **Mem0 Issues**: [#4896](https://github.com/mem0ai/mem0/issues/4896) — Community reports regarding conflicting facts deduplication; closed as "not planned" by the maintainers.
- **Generative Agents** (Park et al., 2023): Retrieval based on `recency × importance × relevance`.
- **MemoryBank** (Zhong et al., 2023): Memory decay based on access frequency.

## License
MIT
