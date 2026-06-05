# mem0-temporal-hygiene

**Temporal context, conservative write-time conflict detection, soft-deletes, guarded deterministic merges, and hash-based caching for Mem0 OSS + Qdrant setups.**

Mem0 OSS stores user facts as flat vectors in Qdrant. Over time, contradictory and duplicate facts accumulate: embeddings can make opposite preferences look very similar, older facts can remain in retrieval, and destructive cleanup can erase useful history. This plugin and maintenance script add a conservative metadata overlay around Mem0 so agents can keep active context cleaner without losing auditability.

> This is a **validity/versioning overlay**, not a full immutable event log. It marks facts as active/superseded/deleted in metadata and links related/conflicting records, but it does not yet provide full rollback, generations, or append-only storage guarantees.

---

## Technical Features

### 1. Soft-Deletes / “Memory Git” Validity Overlay
Instead of physically purging old memories, the hygiene script and plugin use metadata:

- Active memories keep `status: active` or no legacy status.
- Outdated memories are marked `status: superseded` and may include `superseded_by: <winner_id>`.
- Manually deleted memories are marked `status: deleted`.
- Searches exclude `superseded` and `deleted` records at the Qdrant filter layer.

This preserves historical records while preventing obsolete facts from being injected into the agent’s context.

### 2. Conservative Write-Time Guard
The enhanced Hermes plugin annotates new writes with provenance and relation metadata. Before writing, it searches nearby active memories and classifies the new fact as:

- `none` — no notable overlap.
- `related` — semantically related but not a duplicate/conflict.
- `possible_duplicate` — high similarity without negation/toggle risk.
- `suspected_conflict` — high/medium similarity where either side contains negation or toggle language.

The guard is intentionally conservative: it **detects and links** (`conflicts_with`, `possible_duplicate_of`, `related_memory_ids`) but does not automatically delete or supersede existing memories during normal writes.

### 3. Provenance, Confidence, and Schema Metadata
New write/update/delete paths add metadata such as:

```json
{
  "memory_schema_version": "2026-06-05-write-guard-v1",
  "source": "hermes_agent_tool",
  "provenance": "user_explicit_or_agent_observed",
  "confidence": 0.85,
  "source_confidence": 0.85,
  "created_at_client": "2026-06-05T00:00:00+00:00",
  "updated_at_client": "2026-06-05T00:00:00+00:00",
  "write_guard": "detect_append_no_delete",
  "conflict_status": "suspected_conflict"
}
```

Legacy memories without this schema remain readable; active filters treat missing `status` as active.

### 4. DB-Level Active Filtering
The standard `prefetch`, `mem0_profile`, and `mem0_search` operations use Qdrant filters equivalent to:

```json
{
  "must_not": [
    { "key": "status", "match": { "value": "superseded" } },
    { "key": "status", "match": { "value": "deleted" } }
  ]
}
```

Filtering happens in Qdrant so obsolete facts do not clutter the agent’s LLM context.

### 5. Guarded Hybrid Merge
The hygiene script groups semantically similar active points and resolves them via two paths:

- **Guarded deterministic merge (`cosine >= 0.95`)** — only for near-identical facts that do **not** contain negation/toggle language. Older entries are marked `superseded`; the newest survives.
- **LLM-driven cleanup (`cosine >= 0.82`)** — for less obvious clusters. The prompt explicitly warns not to merge opposite toggle/negation preferences merely because embeddings are close.

This avoids the classic failure mode where “enable dark mode” and “disable dark mode” have very high embedding similarity.

### 6. Hash-Based Cache
Clusters are hashed by IDs, timestamps, and content. Unchanged clusters can reuse a prior decision, reducing repeated LLM calls. In `--dry-run` mode, cache writes are skipped.

### 7. Agent-Side CRUD Tools
The plugin adds/extends Hermes tools:

- `mem0_remember(content)` — adds a fact with write-guard metadata.
- `mem0_update(memory_id, content)` — updates a memory with provenance/confidence metadata.
- `mem0_delete(memory_id)` — soft-deletes a memory by UUID instead of physically deleting it.

---

## Important Limitations

- `mem0_remember` still uses Mem0 extraction (`infer=True`) for UX. Depending on Mem0 internals, `add(... infer=True)` may choose hidden UPDATE/DELETE actions. Treat this as a known risk until a strict append-only mode (`infer=False`) is added.
- This project currently provides a metadata validity overlay, not full immutable rollback/generation history.
- Production write tests should be done only in an isolated Qdrant collection; routine validation should prefer mocks/simulations plus read-only retrieval checks.
- The hygiene script can write changes unless run with `--dry-run`.

---

## How It Works

### Temporal Context Flow

```text
Before standard mem0-oss:
  prefetch → "- User prefers dark mode"
              (no date, no ID, no clear validity status)

After this overlay:
  prefetch → "- [2026-05-15] User prefers dark mode [ID: abc123]"
  prefetch → "- [2026-06-01] User switched to light mode [ID: def456]"
              (agent sees dates/IDs and active filters hide superseded facts)
```

### Hygiene Flow

```text
1. Fetch active Qdrant points (exclude status=superseded/deleted).
2. Group by cosine similarity.
3. For near-identical groups, run negation/toggle guard before deterministic merge.
4. For ambiguous groups, call an LLM with anti-conflict rules.
5. Apply metadata updates/soft-deletes, or only report them with --dry-run.
```

---

## Installation

### Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) or compatible framework.
- Qdrant running locally or remotely.
- Mem0 OSS Python package (`pip install mem0ai`).
- An OpenAI-compatible LLM endpoint.

### Plugin Installation

```bash
cp -r ~/.hermes/plugins/mem0-oss ~/.hermes/plugins/mem0-oss.bak
cp plugin/__init__.py ~/.hermes/plugins/mem0-oss/__init__.py
systemctl restart hermes-gateway
```

### Hygiene Script Installation

```bash
cp scripts/memory-hygiene.py ~/.hermes/scripts/memory-hygiene.py
chmod +x ~/.hermes/scripts/memory-hygiene.py

# Safe report-only check
python3 ~/.hermes/scripts/memory-hygiene.py --dry-run

# Apply changes intentionally
python3 ~/.hermes/scripts/memory-hygiene.py
```

### Scheduling Guidance

Do **not** schedule the write-capable command blindly. Prefer a report-first watchdog that runs `--dry-run`, stays silent when there are no proposed changes, and alerts only when cleanup candidates appear.

Example system cron pattern:

```cron
0 2 * * 0 root /usr/bin/python3 /root/.hermes/scripts/memory-hygiene.py --dry-run >> /var/log/memory-hygiene-audit.log 2>&1
```

For Hermes cron jobs, use a script-only/no-agent job that emits stdout only when there is something to review.


### Hermes Cron Audit Wrapper

A safe no-agent audit wrapper is included for deployments that want periodic reports without automatic writes:

```bash
cp scripts/memory-hygiene-audit.sh ~/.hermes/scripts/memory-hygiene-audit.sh
chmod +x ~/.hermes/scripts/memory-hygiene-audit.sh
~/.hermes/scripts/memory-hygiene-audit.sh  # silent when no changes are proposed
```

Hermes cron example:

```text
schedule: 0 3 * * 0
script: memory-hygiene-audit.sh
no_agent: true
```

---

## Configuration

The plugin reads configuration from `~/.hermes/mem0_oss.json` and environment variables. Keep secrets out of git.

```json
{
  "llm_model": "your-model-name",
  "llm_base_url": "http://localhost:20130/v1",
  "llm_api_key": "[REDACTED]",
  "embedder_provider": "openai",
  "embedder_model": "your-embedding-model",
  "embedder_base_url": "http://localhost:20130/v1",
  "embedder_api_key": "[REDACTED]",
  "embedding_dims": 1024,
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection_name": "hermes_default",
  "user_id": "default"
}
```

---

## Related Work

- Mem0 issue [#4896](https://github.com/mem0ai/mem0/issues/4896) — community reports around conflicting facts and deduplication.
- Generative Agents (Park et al., 2023) — retrieval based on `recency × importance × relevance`.
- MemoryBank (Zhong et al., 2023) — memory decay based on access frequency.

## License

MIT
