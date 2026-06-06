"""Mem0 OSS memory plugin — self-hosted memory via local embedder + Qdrant.

Unlike the bundled mem0 plugin (which uses MemoryClient for Mem0 Cloud),
this one wraps the local ``mem0.Memory`` class so everything stays on your
machine: embeddings computed locally (bge-m3), vectors in Qdrant, LLM for
fact extraction via OmniRoute.

Config loaded from $HERMES_HOME/mem0_oss.json (optional). Defaults suit
Dmitry's setup: OmniRoute on :20130, Qdrant on :6333, bge-m3 embedder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


def _default_config() -> dict:
    return {
        "llm_model": "hermes-nvidia-fast",
        "llm_base_url": "http://localhost:20130/v1",
        "llm_api_key": os.environ.get("OPENAI_API_KEY", "sk-hermes-admin"),
        "embedder_provider": "openai",
        "embedder_model": "nvidia/nv-embedqa-e5-v5",
        "embedder_base_url": "http://localhost:20130/v1",
        "embedder_api_key": os.environ.get("OPENAI_API_KEY", "sk-hermes-admin"),
        "embedder_model_kwargs": {"extra_body": {"input_type": "passage"}},
        "embedding_dims": None,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "hermes_dmitry",
        "user_id": "dmitry",
    }


def _load_config() -> dict:
    from hermes_constants import get_hermes_home
    cfg = _default_config()
    path = get_hermes_home() / "mem0_oss.json"
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("mem0_oss.json parse error: %s", e)
    return cfg


def _clip_text(text: str, limit: int) -> str:
    text = text or ""
    if limit and len(text) > limit:
        return text[:limit] + "…"
    return text



MEMORY_SCHEMA_VERSION = "2026-06-05-write-guard-v1"
NEGATION_RE = re.compile(
    r"\b(не|нет|никогда|запрет|запрещ|не надо|no|not|never|disable|disabled|off|without|avoid|don't|do not)\b",
    re.IGNORECASE,
)
TOGGLE_RE = re.compile(
    r"\b(enable|enabled|disable|disabled|on|off|turn on|turn off|включи|включать|выключи|выключать|активир|деактивир)\b",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _extract_subject_keys(text: str) -> set[str]:
    t = (text or "").lower()
    entities = [
        "omniroute", "qdrant", "plex", "torrserver", "transmission", "rclone", 
        "yandex", "backup", "cookie", "vps", "dreame", "obsidian", "sasha", 
        "nadya", "nestor", "andrey", "stt", "tts", "whisper", "slack", 
        "telegram", "ha", "home assistant", "фото", "видео", "мантра", 
        "песня", "санскрит", "порт", "таймаут", "логи"
    ]
    found = {ent for ent in entities if ent in t}
    # Match structural variables like 'name = value' or 'name: value'
    match = re.search(r"^\s*([a-zа-я0-9_\-\s]{3,30})\s*[:=]", t)
    if match:
        var_name = match.group(1).strip()
        # Only add if it's a solid word combination
        if len(var_name.split()) <= 3:
            found.add(var_name)
    return found


def _calculate_temporal_decay(created_at_str: str, source: str) -> float:
    try:
        if not created_at_str:
            return 1.0
        # Parse ISO client time or created_at
        dt_str = created_at_str.replace("Z", "+00:00")
        # Handle formats like 2026-06-06T18:00:58
        if "T" in dt_str and "+" not in dt_str and "-" not in dt_str[10:]:
            dt_str += "+00:00"
        created_at = datetime.fromisoformat(dt_str)
        now = datetime.now(timezone.utc)
        days = (now - created_at).days
        if days < 0:
            days = 0
            
        # Select lambda based on source
        # user explicit never decays
        if "explicit" in (source or "") or (source or "") in ("user", "user_explicit"):
            decay_rate = 0.0
        elif (source or "") in ("tool", "tool_log"):
            decay_rate = 0.05  # fast decay: half-life ~14 days
        else:
            decay_rate = 0.005 # default gentle decay (half-life ~138 days)
            
        import math
        return math.exp(-decay_rate * days)
    except Exception:
        return 1.0


def _fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _has_negation_or_toggle(text: str) -> bool:
    return bool(NEGATION_RE.search(text or "") or TOGGLE_RE.search(text or ""))


def _memory_text(item: dict) -> str:
    return item.get("memory") or item.get("text") or item.get("data") or ""

def _build_mem0_config(cfg: dict) -> dict:
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


PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Получить все факты о пользователе из долгосрочной памяти (Mem0 OSS self-hosted). "
        "Вызывай в начале разговора чтобы понять кто это."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Семантический поиск по памяти. Возвращает релевантные факты. "
        "Используй когда нужно вспомнить что-то специфичное о пользователе."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Что искать."},
            "top_k": {"type": "integer", "description": "Сколько результатов (default 5, max 20)."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "mem0_remember",
    "description": (
        "Сохранить факт о пользователе в долгосрочную память. Mem0 сам извлечёт "
        "атомарные факты и дедуплицирует. Используй при явных предпочтениях, "
        "корректировках, важных деталях."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Текст для запоминания."},
        },
        "required": ["content"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Удалить факт о пользователе из долгосрочной памяти (Mem0 OSS) по его UUID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "UUID факта (отображается в квадратных скобках [ID: ...] в памяти)."}
        },
        "required": ["memory_id"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Обновить существующий факт о пользователе в долгосрочной памяти по его UUID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "UUID факта для редактирования."},
            "content": {"type": "string", "description": "Новый текст факта, заменяющий старый."}
        },
        "required": ["memory_id", "content"],
    },
}


class Mem0OSSProvider(MemoryProvider):
    """Self-hosted Mem0 via local embedder + Qdrant + OmniRoute LLM."""

    def __init__(self):
        self._cfg: dict | None = None
        self._memory = None
        self._lock = threading.Lock()
        self._user_id = "dmitry"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None

    @property
    def name(self) -> str:
        return "mem0-oss"

    def is_available(self) -> bool:
        # Qdrant up?
        try:
            import urllib.request
            cfg = _load_config()
            url = f"http://{cfg['qdrant_host']}:{cfg['qdrant_port']}/"
            urllib.request.urlopen(url, timeout=2)
            import mem0  # noqa: F401
            return True
        except Exception:
            return False

    def _get_memory(self):
        with self._lock:
            if self._memory is not None:
                return self._memory
            from mem0 import Memory
            self._cfg = _load_config()
            self._memory = Memory.from_config(_build_mem0_config(self._cfg))
            return self._memory

    def _get_active_filters(self) -> dict:
        return {
            "user_id": self._user_id,
            "NOT": [
                {"status": "superseded"},
                {"status": "deleted"}
            ]
        }

    def _base_metadata(self, *, source: str = "explicit_tool", confidence: float = 0.85) -> dict:
        now = _utc_now()
        return {
            "status": "active",
            "user_id": self._user_id,
            "memory_schema_version": MEMORY_SCHEMA_VERSION,
            "source": source,
            "provenance": source,
            "confidence": confidence,
            "source_confidence": confidence,
            "created_at_client": now,
            "updated_at_client": now,
        }

    def _search_write_candidates(self, m, content: str, *, top_k: int = 5) -> list[dict]:
        try:
            clipped = (content or "")[:800]
            if not clipped.strip():
                return []
            res = m.search(query=clipped, filters=self._get_active_filters(), top_k=top_k)
            items = res.get("results", []) if isinstance(res, dict) else res
            return [r for r in items if _memory_text(r)]
        except TypeError:
            res = m.search(query=(content or "")[:800], filters=self._get_active_filters(), limit=top_k)
            items = res.get("results", []) if isinstance(res, dict) else res
            return [r for r in items if _memory_text(r)]
        except Exception as e:
            logger.debug("mem0 write-candidate search failed: %s", e)
            return []

    def _classify_write_relation(self, content: str, candidates: list[dict]) -> dict:
        related = []
        possible_duplicates = []
        possible_conflicts = []
        new_guarded = _has_negation_or_toggle(content)
        keys_content = _extract_subject_keys(content)
        
        for c in candidates:
            cid = c.get("id") or c.get("memory_id") or ""
            if not cid:
                continue
            score = float(c.get("score") or 0.0)
            text = _memory_text(c)
            
            # Key/subject intersection safety check
            keys_cand = _extract_subject_keys(text)
            common_entities = keys_content.intersection(keys_cand)
            if common_entities:
                diff_content = keys_content - keys_cand
                diff_cand = keys_cand - keys_content
                specifiers = {"порт", "port", "timeout", "таймаут", "логи", "log", "backup", "бэкап", "почта", "email", "url", "путь", "path", "token", "токен"}
                has_diff_spec = bool(diff_content.intersection(specifiers) and diff_cand.intersection(specifiers))
                if has_diff_spec:
                    # They speak about different keys of the same entity (e.g. port vs timeout). Bypass classification.
                    logger.debug("Bypass: candidate has mismatching specifier keys compared to request")
                    continue

            if score >= 0.72:
                related.append(cid)
            if score >= 0.92 and not (new_guarded or _has_negation_or_toggle(text)):
                possible_duplicates.append(cid)
            if score >= 0.78 and (new_guarded or _has_negation_or_toggle(text)):
                possible_conflicts.append(cid)

        status = "none"
        if possible_conflicts:
            status = "suspected_conflict"
        elif possible_duplicates:
            status = "possible_duplicate"

        return {
            "conflict_status": status,
            "conflicts_with": possible_conflicts[:5],
            "possible_duplicate_of": possible_duplicates[:5],
            "related_memory_ids": related[:5],
            "write_guard": "detect_append_no_delete",
            "candidate_count": len(candidates),
        }

    def save_config(self, values, hermes_home):
        from pathlib import Path
        path = Path(hermes_home) / "mem0_oss.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                pass
        existing.update({k: v for k, v in values.items() if v})
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    def get_config_schema(self):
        return [
            {"key": "user_id", "description": "User identifier", "default": "dmitry"},
            {"key": "llm_model", "description": "LLM via OmniRoute", "default": "hermes-nvidia-fast"},
            {"key": "llm_base_url", "description": "OmniRoute URL", "default": "http://localhost:20130/v1"},
            {"key": "embedder_model", "description": "Local embedding model", "default": "BAAI/bge-m3"},
            {"key": "qdrant_host", "description": "Qdrant host", "default": "localhost"},
            {"key": "qdrant_port", "description": "Qdrant port", "default": 6333},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cfg = _load_config()
        self._user_id = kwargs.get("user_id") or self._cfg.get("user_id", "dmitry")

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 OSS Memory (self-hosted)\n"
            f"Активна. User: {self._user_id}. Хранилище: Qdrant локально. "
            "Эмбеддинги через OmniRoute/NVIDIA, LLM для извлечения фактов через OmniRoute.\n"
            "Инструменты: mem0_profile (всё о пользователе), mem0_search (поиск), "
            "mem0_remember (сохранить факт), mem0_update (обновить факт), mem0_delete (удалить факт)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return f"## Mem0 OSS Memory\n{result}" if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            try:
                m = self._get_memory()
                # Clip search query to prevent embedding token limit errors (e.g. 512 tokens for NVIDIA NIM)
                clipped = query[:800] if query else ""
                res = m.search(query=clipped, filters=self._get_active_filters(), limit=5)
                items = res.get("results", []) if isinstance(res, dict) else res
                lines = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    created_at = r.get("created_at") or ""
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    mem_id = r.get("id") or ""
                    id_suffix = f" [ID: {mem_id}]" if mem_id else ""
                    lines.append(f"- {date_prefix}{mem}{id_suffix}")
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("mem0-oss prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0oss-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Disabled: auto-sync creates too much noise (session logs, not durable facts).
        # Memory is now write-only via explicit mem0_remember calls.
        # To re-enable, uncomment the block below.
        pass
        # def _sync():
        #     try:
        #         m = self._get_memory()
        #         user_limit = int((self._cfg or {}).get("sync_user_char_limit", 700))
        #         assistant_limit = int((self._cfg or {}).get("sync_assistant_char_limit", 700))
        #         messages = [
        #             {"role": "user", "content": _clip_text(user_content, user_limit)},
        #             {"role": "assistant", "content": _clip_text(assistant_content, assistant_limit)},
        #         ]
        #         m.add(messages, user_id=self._user_id)
        #     except Exception as e:
        #         logger.warning("mem0-oss sync failed: %s", e)
        #
        # if self._sync_thread and self._sync_thread.is_alive():
        #     self._sync_thread.join(timeout=5.0)
        # self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0oss-sync")
        # self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            m = self._get_memory()
        except Exception as e:
            return tool_error(f"Mem0 OSS unavailable: {e}")

        if tool_name == "mem0_profile":
            try:
                res = m.get_all(filters=self._get_active_filters())
                items = res.get("results", []) if isinstance(res, dict) else res
                if not items:
                    return json.dumps({"result": "Память пуста."})
                lines = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    created_at = r.get("created_at") or ""
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    mem_id = r.get("id") or ""
                    id_suffix = f" [ID: {mem_id}]" if mem_id else ""
                    lines.append(f"{date_prefix}{mem}{id_suffix}")
                return json.dumps({"result": "\n".join(lines), "count": len(lines)}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Ошибка получения профиля: {e}")

        if tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Нужен параметр query")
            top_k = min(int(args.get("top_k", 5)), 20)
            try:
                # Clip search query to prevent embedding token limit errors (e.g. 512 tokens for NVIDIA NIM)
                clipped = query[:800]
                # Query a wider pool of candidates to allow decay re-ranking
                search_limit = min(top_k * 3, 50)
                res = m.search(query=clipped, filters=self._get_active_filters(), limit=search_limit)
                items = res.get("results", []) if isinstance(res, dict) else res
                if not items:
                    return json.dumps({"result": "Релевантных фактов не найдено."}, ensure_ascii=False)
                
                # Perform hybrid scoring (re-ranking)
                scored_items = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    
                    metadata = r.get("metadata", {}) or {}
                    created_at = r.get("created_at") or metadata.get("created_at") or metadata.get("created_at_client") or ""
                    source = metadata.get("source") or metadata.get("provenance") or "agent_decision"
                    confidence = float(metadata.get("confidence") or metadata.get("source_confidence") or 0.85)
                    
                    decay_factor = _calculate_temporal_decay(created_at, source)
                    sim_score = float(r.get("score", 0.0))
                    
                    # Hybrid formula: 70% similarity, 30% decay and confidence
                    final_score = 0.7 * sim_score + 0.3 * decay_factor * confidence
                    
                    scored_items.append({
                        "id": r.get("id") or "",
                        "memory": mem,
                        "created_at": created_at,
                        "raw_score": sim_score,
                        "score": final_score,
                        "decay_factor": decay_factor,
                        "confidence": confidence
                    })
                
                # Sort descending by updated final score
                scored_items.sort(key=lambda x: x["score"], reverse=True)
                
                # Slice down to requested top_k
                final_results = scored_items[:top_k]
                
                out = []
                for r in final_results:
                    created_at = r["created_at"]
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    out.append({
                        "id": r["id"],
                        "memory": f"{date_prefix}{r['memory']}",
                        "score": r["score"],
                        "raw_score": r["raw_score"],
                        "created_at": created_at
                    })
                return json.dumps({"results": out, "count": len(out)}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Поиск не удался: {e}")

        if tool_name == "mem0_remember":
            content = args.get("content", "")
            if not content:
                return tool_error("Нужен параметр content")
            try:
                limit = int((self._cfg or {}).get("remember_char_limit", 900))
                clipped_content = _clip_text(content, limit)
                candidates = self._search_write_candidates(m, clipped_content)
                relation = self._classify_write_relation(clipped_content, candidates)
                metadata = self._base_metadata(source="explicit_mem0_remember", confidence=0.85)
                metadata.update(relation)
                metadata["content_fingerprint"] = _fingerprint(clipped_content)

                # Conservative write path: detect and annotate conflicts/duplicates, but let Mem0
                # keep its normal extraction UX. We do not soft-delete or supersede anything here.
                res = m.add(
                    clipped_content,
                    user_id=self._user_id,
                    metadata=metadata,
                    infer=bool((self._cfg or {}).get("remember_infer", True)),
                )
                items = res.get("results", []) if isinstance(res, dict) else res
                added = [r for r in items if r.get("event") == "ADD"]
                return json.dumps({
                    "result": f"Сохранено {len(added)} факт(ов).",
                    "facts": [r.get("memory", "") for r in added],
                    "write_guard": {
                        "conflict_status": relation["conflict_status"],
                        "conflicts_with": relation["conflicts_with"],
                        "possible_duplicate_of": relation["possible_duplicate_of"],
                    },
                }, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Сохранение не удалось: {e}")

        if tool_name == "mem0_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Нужен параметр memory_id")
            try:
                # Get the existing memory text to preserve it during soft-delete update
                mem_item = m.get(memory_id)
                if not mem_item:
                    return tool_error(f"Факт с ID {memory_id} не найден.")
                content = mem_item.get("memory") or mem_item.get("text") or ""
                # Perform soft-delete by setting status payload to "deleted" while keeping provenance.
                metadata = self._base_metadata(source="explicit_mem0_delete", confidence=1.0)
                metadata.update({"status": "deleted", "deleted_at_client": _utc_now()})
                m.update(memory_id, content, metadata=metadata)
                return json.dumps({"result": f"Факт {memory_id} успешно помечен как удалённый (soft-delete)."}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Не удалось удалить факт: {e}")

        if tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            content = args.get("content", "")
            if not memory_id or not content:
                return tool_error("Нужны параметры memory_id и content")
            try:
                candidates = [c for c in self._search_write_candidates(m, content) if (c.get("id") or c.get("memory_id")) != memory_id]
                relation = self._classify_write_relation(content, candidates)
                metadata = self._base_metadata(source="explicit_mem0_update", confidence=0.95)
                metadata.update(relation)
                metadata["content_fingerprint"] = _fingerprint(content)
                metadata["updated_memory_id"] = memory_id
                m.update(memory_id, content, metadata=metadata)
                return json.dumps({
                    "result": f"Факт {memory_id} успешно обновлён.",
                    "write_guard": {
                        "conflict_status": relation["conflict_status"],
                        "conflicts_with": relation["conflicts_with"],
                        "possible_duplicate_of": relation["possible_duplicate_of"],
                    },
                }, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Не удалось обновить факт: {e}")

        return tool_error(f"Неизвестный tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    ctx.register_memory_provider(Mem0OSSProvider())
