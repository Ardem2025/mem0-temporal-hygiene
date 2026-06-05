"""Mem0 OSS memory plugin — self-hosted memory via local embedder + Qdrant.

Unlike the bundled mem0 plugin (which uses MemoryClient for Mem0 Cloud),
this one wraps the local ``mem0.Memory`` class so everything stays on your
machine: embeddings computed locally (bge-m3), vectors in Qdrant, LLM for
fact extraction via OmniRoute.

Config loaded from $HERMES_HOME/mem0_oss.json (optional). Defaults suit
Dmitry's setup: OmniRoute on :20130, Qdrant on :6333, bge-m3 embedder.
"""
from __future__ import annotations

import json
import logging
import os
import threading
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
                res = m.search(query=clipped, filters=self._get_active_filters(), limit=top_k)
                items = res.get("results", []) if isinstance(res, dict) else res
                if not items:
                    return json.dumps({"result": "Релевантных фактов не найдено."}, ensure_ascii=False)
                out = []
                for r in items:
                    mem = r.get("memory", "")
                    if not mem:
                        continue
                    created_at = r.get("created_at") or ""
                    date_prefix = f"[{created_at[:10]}] " if created_at and len(created_at) >= 10 else ""
                    mem_id = r.get("id") or ""
                    out.append({
                        "id": mem_id,
                        "memory": f"{date_prefix}{mem}",
                        "score": r.get("score", 0),
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
                res = m.add(_clip_text(content, limit), user_id=self._user_id)
                items = res.get("results", []) if isinstance(res, dict) else res
                added = [r for r in items if r.get("event") == "ADD"]
                return json.dumps({
                    "result": f"Сохранено {len(added)} факт(ов).",
                    "facts": [r.get("memory", "") for r in added],
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
                # Perform soft-delete by setting status payload to "deleted"
                m.update(memory_id, content, metadata={"status": "deleted", "user_id": self._user_id})
                return json.dumps({"result": f"Факт {memory_id} успешно помечен как удалённый (soft-delete)."}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Не удалось удалить факт: {e}")

        if tool_name == "mem0_update":
            memory_id = args.get("memory_id", "")
            content = args.get("content", "")
            if not memory_id or not content:
                return tool_error("Нужны параметры memory_id и content")
            try:
                m.update(memory_id, content)
                return json.dumps({"result": f"Факт {memory_id} успешно обновлён."}, ensure_ascii=False)
            except Exception as e:
                return tool_error(f"Не удалось обновить факт: {e}")

        return tool_error(f"Неизвестный tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    ctx.register_memory_provider(Mem0OSSProvider())
