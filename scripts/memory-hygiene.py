#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Hygiene and De-duplication script for KUZHOMESRV.
Walks through the Mem0 Qdrant collection, groups semantically similar facts,
and uses LLM-driven consolidation to resolve contradictions and remove obsolete points.
"""

import os
import sys
import json
import urllib.request
import logging
import hashlib
import re
import argparse
from datetime import datetime, timezone
import numpy as np
import dotenv
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory-hygiene")


MEMORY_SCHEMA_VERSION = "2026-06-05-write-guard-v1"
NEGATION_RE = re.compile(
    r"\b(не|нет|никогда|запрет|запрещ|не надо|no|not|never|disable|disabled|off|without|avoid|don't|do not)\b",
    re.IGNORECASE,
)
TOGGLE_RE = re.compile(
    r"\b(enable|enabled|disable|disabled|on|off|turn on|turn off|включи|включать|выключи|выключать|активир|деактивир)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def has_negation_or_toggle(text: str) -> bool:
    return bool(NEGATION_RE.search(text or "") or TOGGLE_RE.search(text or ""))


def point_text(point: dict) -> str:
    payload = point.get("payload", {})
    return payload.get("data") or payload.get("memory") or ""

# Load environment variables
dotenv.load_dotenv("/root/.hermes/.env")

# Ensure paths
sys.path.insert(0, "/root/.hermes")
sys.path.insert(0, "/root/.hermes/plugins")
sys.path.insert(0, "/opt/hermes-agent")

from mem0 import Memory

def load_mem0_config() -> dict:
    default_cfg = {
        "llm_model": "hermes-nvidia-fast",
        "llm_base_url": "http://localhost:20130/v1",
        "llm_api_key": os.environ.get("OPENAI_API_KEY", "***"),
        "embedder_provider": "openai",
        "embedder_model": "nvidia/nv-embedqa-e5-v5",
        "embedder_base_url": "http://localhost:20130/v1",
        "embedder_api_key": os.environ.get("OPENAI_API_KEY", "***"),
        "embedder_model_kwargs": {"extra_body": {"input_type": "passage"}, "omit_dimensions": True},
        "embedding_dims": 1024,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "hermes_dmitry",
        "user_id": "139351986",
    }
    
    path = "/root/.hermes/mem0_oss.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                # Resolve masking
                for k, v in user_cfg.items():
                    if v == "***":
                        if "llm" in k or "embedder" in k:
                            user_cfg[k] = os.environ.get("OPENAI_API_KEY", "***")
                default_cfg.update(user_cfg)
        except Exception as e:
            logger.warning("Failed to load mem0_oss.json: %s", e)
            
    return default_cfg

def build_mem0_params(cfg: dict) -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": cfg["llm_model"],
                "openai_base_url": cfg["llm_base_url"],
                "api_key": cfg["llm_api_key"],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": cfg.get("embedder_provider", "openai"),
            "config": {
                "model": cfg["embedder_model"],
                "openai_base_url": cfg.get("embedder_base_url", cfg["llm_base_url"]),
                "api_key": cfg.get("embedder_api_key", cfg["llm_api_key"]),
                "embedding_dims": cfg.get("embedding_dims"),
                "model_kwargs": cfg.get("embedder_model_kwargs", {}),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": cfg["qdrant_host"],
                "port": cfg["qdrant_port"],
                "collection_name": cfg["collection_name"],
                "embedding_model_dims": cfg["embedding_dims"],
            },
        },
        "version": "v1.1",
    }

def cosine_similarity(v1, v2) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def fetch_qdrant_points(cfg: dict) -> list:
    url = f"http://{cfg['qdrant_host']}:{cfg['qdrant_port']}/collections/{cfg['collection_name']}/points/scroll"
    payload = {
        "limit": 500,
        "with_payload": True,
        "with_vector": True,
        "filter": {
            "must_not": [
                {
                    "key": "status",
                    "match": {
                        "value": "superseded"
                    }
                },
                {
                    "key": "status",
                    "match": {
                        "value": "deleted"
                    }
                }
            ]
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        return res.get("result", {}).get("points", [])

def group_points_by_similarity(points: list, threshold: float = 0.82) -> list:
    groups = []
    visited = set()
    
    for i, p_i in enumerate(points):
        if p_i["id"] in visited:
            continue
            
        vector_i = p_i.get("vector", {}).get("")
        if not vector_i:
            continue
            
        current_group = [p_i]
        visited.add(p_i["id"])
        
        for j in range(i + 1, len(points)):
            p_j = points[j]
            if p_j["id"] in visited:
                continue
                
            vector_j = p_j.get("vector", {}).get("")
            if not vector_j:
                continue
                
            sim = cosine_similarity(vector_i, vector_j)
            if sim >= threshold:
                current_group.append(p_j)
                visited.add(p_j["id"])
                
        if len(current_group) >= 2:
            groups.append(current_group)
            
    return groups

def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
    return {}

def save_cache(cache_path: str, cache: dict):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save cache: %s", e)

def get_group_hash(group: list) -> str:
    sorted_points = sorted(group, key=lambda x: x["id"])
    hasher = hashlib.sha256()
    for p in sorted_points:
        p_id = p["id"]
        p_payload = p.get("payload", {})
        p_data = p_payload.get("data") or p_payload.get("memory") or ""
        created_at = p_payload.get("created_at") or ""
        item_str = f"{p_id}:{created_at}:{p_data}"
        hasher.update(item_str.encode('utf-8'))
    return hasher.hexdigest()

def check_deterministic_merge(group: list, threshold: float = 0.95) -> dict | None:
    if len(group) < 2:
        return None

    # Guard against antiphrase/toggle false merges. Vector similarity can put
    # "enable X" and "disable X" very close; leave those to the LLM path.
    guarded = [p for p in group if has_negation_or_toggle(point_text(p))]
    if guarded:
        logger.info("  [DETERMINISTIC SKIP] Negation/toggle marker found; routing group to LLM analysis instead of auto-merge.")
        return None

    sorted_group = sorted(group, key=lambda x: x.get("payload", {}).get("created_at") or "")
    newest = sorted_group[-1]
    newest_vector = newest.get("vector", {}).get("")
    if not newest_vector:
        return None

    all_matching = True
    for p in sorted_group[:-1]:
        p_vector = p.get("vector", {}).get("")
        if not p_vector:
            all_matching = False
            break
        sim = cosine_similarity(newest_vector, p_vector)
        if sim < threshold:
            all_matching = False
            break

    if all_matching:
        deletions = [p["id"] for p in sorted_group[:-1]]
        logger.info(f"  [DETERMINISTIC MERGE] Group of {len(group)} items is near-identical (cosine >= {threshold}) and passed negation guard. Keeping newest [{newest['id'][:8]}] and soft-deleting others.")
        return {"deletions": deletions, "updates": [], "decision_source": "deterministic_similarity_guarded"}

    return None

def analyze_group_with_llm(client: openai.OpenAI, model_name: str, group: list) -> dict:
    memories_str = []
    for p in group:
        p_id = p["id"]
        p_payload = p.get("payload", {})
        p_data = p_payload.get("data") or p_payload.get("memory") or ""
        created_at = p_payload.get("created_at") or ""
        memories_str.append({
            "id": p_id,
            "text": p_data,
            "created_at": created_at
        })
        
    prompt = f"""You are a Database Memory Janitor. You are given a group of vector-similar memories stored about a user.
They may represent duplicates, overlapping details, or contradictory facts where one fact is outdated and was replaced by a newer decision.

Review this list of memories:
{json.dumps(memories_str, indent=2, ensure_ascii=False)}

Your Goal:
1. Determine if there are contradictory or overlapping facts.
2. If there are contradictions, the newer facts (based on 'created_at' timestamp) override older ones.
3. If facts are fully duplicated or represent minor redundant details, merge them into a single, clean, concise memory (using Russian, keeping key facts like paths, names, numbers intact).
4. If some memories are obsolete or fully redundant, output their IDs in "deletions".
5. If an existing memory needs to be updated with consolidated clean text, output its ID and consolidated text in "updates".
6. If the memories are actually about completely different things and should both be kept as they are, return empty updates and deletions.

SPECIAL RULES:
- CRITICAL: ANTIPHASE / CONFLICTING CONFIGURATIONS
  Pay extremely close attention to negation or toggle words (e.g., "off", "on", "not", "no", "never", "inactive", "active", "не", "нет", "включить", "отключить", "загружать", "блокировать").
  If the memories represent polar opposite states or different toggle options (e.g., "dark mode is on" vs "dark mode is off", "agent must ask before X" vs "agent must do X automatically", "Yandex cookies refresh using CDP" vs "cookies expired, auth manually"), do NOT merge them or mark either as deleted. Keep them both as separate, distinct facts of different circumstances.
  
- USER INTENT & VERIFIED SOURCE PRIORITY
  Usually, newer facts override older ones. However, if an older fact represents a verified explicit user preference (e.g. "Dmitry prefers setting X") and a newer fact is a transient observation of a server error or a temporary state (e.g., "Script failed with error Y"), do NOT let the newer error state override the user's explicit preference. Explicit intent and preference MUST override transient observations.

Format the output strictly as a JSON object, like this:
{{
  "deletions": ["uuid-to-delete-1", "uuid-to-delete-2"],
  "updates": [
    {{
      "id": "uuid-to-keep-and-update",
      "text": "New consolidated fact text goes here."
    }}
  ]
}}

No extra comments, no markdown code blocks, just raw JSON.
"""
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a database cleanup utility. Respond ONLY with valid JSON structure."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = res.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                content = "\n".join(lines[1:-1])
        return json.loads(content)
    except Exception as e:
        logger.error("LLM group analysis failed: %s", e)
        return {"deletions": [], "updates": []}

def main(dry_run: bool = False):
    logger.info("Initializing Memory Hygiene Job%s...", " (dry-run/report-only)" if dry_run else "")
    cfg = load_mem0_config()
    
    # Initialize Mem0 Memory SDK
    mem0_params = build_mem0_params(cfg)
    m = Memory.from_config(mem0_params)
    
    # Initialize OpenAI Client via OmniRoute
    client = openai.OpenAI(
        base_url=cfg["llm_base_url"],
        api_key=cfg["llm_api_key"]
    )
    model_name = cfg["llm_model"]
    
    # Fetch all points
    try:
        points = fetch_qdrant_points(cfg)
        logger.info(f"Successfully fetched {len(points)} points from Qdrant.")
    except Exception as e:
        logger.error(f"Failed to fetch points from Qdrant: {e}")
        sys.exit(1)
        
    if not points:
        logger.info("Database is empty. Nothing to clean.")
        return
        
    # Group by similarity
    logger.info("Grouping semantically similar points...")
    groups = group_points_by_similarity(points, threshold=0.82)
    logger.info(f"Found {len(groups)} groups of potentially similar/duplicate memories.")
    
    # Load cache
    cache_path = "/root/.hermes/memory_hygiene_cache.json"
    cache = load_cache(cache_path)
    
    total_deletions = 0
    total_updates = 0
    
    for idx, group in enumerate(groups):
        logger.info(f"Analyzing group {idx+1}/{len(groups)} with {len(group)} items...")
        for p in group:
            p_payload = p.get("payload", {})
            p_data = p_payload.get("data") or p_payload.get("memory") or ""
            logger.info(f"  - [{p['id'][:8]}] {p_payload.get('created_at', '')[:10]}: {p_data[:80]}...")
            
        group_hash = get_group_hash(group)
        
        # Check rule-based deterministic merge first (similarity >= 0.95)
        deterministic_decision = check_deterministic_merge(group, threshold=0.95)
        
        if deterministic_decision is not None:
            decision = deterministic_decision
            if not dry_run:
                cache[group_hash] = decision
                save_cache(cache_path, cache)
        else:
            if group_hash in cache:
                logger.info(f"  [CACHE HIT] Cluster is unchanged since last hygiene run. Skipping LLM call and applying cached decision.")
                decision = cache[group_hash]
            else:
                decision = analyze_group_with_llm(client, model_name, group)
                if not dry_run:
                    cache[group_hash] = decision
                    save_cache(cache_path, cache)
            
        logger.info(f"  Decision: {json.dumps(decision, ensure_ascii=False)}")
        
        deletions = decision.get("deletions", [])
        updates = decision.get("updates", [])
        
        # Performance updates via Mem0 SDK to sync vector embedding changes
        for update_item in updates:
            up_id = update_item.get("id")
            up_text = update_item.get("text")
            if up_id and up_text:
                try:
                    meta = {
                        "status": "active",
                        "user_id": cfg.get("user_id", "dmitry"),
                        "memory_schema_version": MEMORY_SCHEMA_VERSION,
                        "source": "memory_hygiene",
                        "provenance": "batch_hygiene_llm_or_rule",
                        "confidence": 0.8,
                        "source_confidence": 0.8,
                        "updated_at_client": utc_now(),
                        "hygiene_decision": "consolidated_update",
                    }
                    if dry_run:
                        logger.info(f"  [DRY-RUN][UPDATE] Would update fact {up_id[:8]} -> '{up_text}' with metadata {meta}")
                    else:
                        m.update(up_id, up_text, metadata=meta)
                        logger.info(f"  [UPDATED] Fact {up_id[:8]} -> '{up_text}' with metadata {meta}")
                    total_updates += 1
                except Exception as e:
                    logger.error(f"  Failed to update fact {up_id}: {e}")
                    
        # Perform soft-deletions via Mem0 SDK to retain historical record
        user_id = cfg.get("user_id", "dmitry")
        for del_id in deletions:
            try:
                del_text = None
                for p in group:
                    if p["id"] == del_id:
                        p_payload = p.get("payload", {})
                        del_text = p_payload.get("data") or p_payload.get("memory") or ""
                        break
                
                if not del_text:
                    try:
                        mem_info = m.get(del_id)
                        del_text = mem_info.get("memory") if mem_info else None
                    except Exception:
                        pass
                
                if not del_text:
                    del_text = "[Superseded fact]"

                winner_id = None
                if updates:
                    winner_id = updates[0].get("id")
                else:
                    remaining = [p["id"] for p in group if p["id"] not in deletions]
                    if remaining:
                        winner_id = remaining[0]

                meta = {
                    "status": "superseded",
                    "user_id": user_id,
                    "memory_schema_version": MEMORY_SCHEMA_VERSION,
                    "source": "memory_hygiene",
                    "provenance": "batch_hygiene_llm_or_rule",
                    "confidence": 0.75,
                    "source_confidence": 0.8,
                    "updated_at_client": utc_now(),
                    "superseded_at_client": utc_now(),
                    "hygiene_decision": "soft_delete_superseded",
                }
                if winner_id:
                    meta["superseded_by"] = winner_id
                else:
                    meta["status"] = "deleted"
                    meta["deleted_at_client"] = utc_now()

                if dry_run:
                    logger.info(f"  [DRY-RUN][SOFT-DELETE] Would mark fact {del_id[:8]} with metadata {meta}")
                else:
                    m.update(del_id, del_text, metadata=meta)
                    logger.info(f"  [SOFT-DELETED] Fact {del_id[:8]} with metadata {meta}")
                total_deletions += 1
            except Exception as e:
                logger.error(f"  Failed to soft-delete fact {del_id}: {e}")
                
    action = "would soft-delete" if dry_run else "deleted"
    update_action = "would update" if dry_run else "updated"
    logger.info(f"Hygiene job complete. Deletions check: {action} {total_deletions} points, {update_action} {total_updates} points.")


def parse_args():
    parser = argparse.ArgumentParser(description="Mem0/Qdrant memory hygiene with soft-delete metadata overlay.")
    parser.add_argument("--dry-run", action="store_true", help="Report proposed updates/soft-deletes without writing to Mem0/Qdrant.")
    parser.add_argument("--report-only", action="store_true", help="Alias for --dry-run; intended for safe scheduled audits.")
    args = parser.parse_args()
    if args.report_only:
        args.dry_run = True
    return args


if __name__ == "__main__":
    args = parse_args()
    main(dry_run=args.dry_run)
