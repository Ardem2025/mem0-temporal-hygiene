#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Memory Hygiene & Reflection Report collector.
Fetches all points from Qdrant, aggregates statistics on active/superseded/deleted records,
and compiles a clean report of transactions (additions, consolidations, soft-deletions) over the last 24 hours.
"""

import os
import sys
import json
import urllib.request
import logging
from datetime import datetime, timezone, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory-reflection")

# Load environment
sys.path.insert(0, "/root/.hermes")
sys.path.insert(0, "/opt/hermes-agent")

def load_mem0_config() -> dict:
    default_cfg = {
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "hermes_dmitry",
        "user_id": "dmitry",
    }
    path = "/root/.hermes/mem0_oss.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                default_cfg.update(json.load(f))
        except Exception as e:
            logger.warning("Failed to load mem0_oss.json: %s", e)
    return default_cfg

def fetch_all_qdrant_points(cfg: dict) -> list:
    url = f"http://{cfg['qdrant_host']}:{cfg['qdrant_port']}/collections/{cfg['collection_name']}/points/scroll"
    payload = {
        "limit": 1000,
        "with_payload": True,
        "with_vector": False
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

def print_reflection_report():
    cfg = load_mem0_config()
    try:
        points = fetch_all_qdrant_points(cfg)
    except Exception as e:
        print(f"⚠️ **Не удалось собрать статистику памяти:** Qdrant недоступен или коллекция пуста ({e})")
        return

    now = datetime.now(timezone.utc)
    one_day_ago = now - timedelta(days=1)

    total_count = len(points)
    active_count = 0
    superseded_count = 0
    deleted_count = 0

    added_today = []
    updated_today = []
    deleted_today = []

    for p in points:
        payload = p.get("payload", {}) or {}
        status = payload.get("status", "active")
        text = payload.get("data") or payload.get("memory") or ""
        
        # Count statuses
        if status == "active":
            active_count += 1
        elif status == "superseded":
            superseded_count += 1
        elif status == "deleted":
            deleted_count += 1

        # Parse timestamps to detect changes today
        created_at_str = payload.get("created_at") or payload.get("created_at_client") or ""
        updated_at_str = payload.get("updated_at_client") or ""
        superseded_at_str = payload.get("superseded_at_client") or ""
        deleted_at_str = payload.get("deleted_at_client") or ""

        def parse_date(date_str):
            if not date_str:
                return None
            try:
                ds = date_str.replace("Z", "+00:00")
                if "T" in ds and "+" not in ds and "-" not in ds[10:]:
                    ds += "+00:00"
                return datetime.fromisoformat(ds)
            except Exception:
                return None

        created_dt = parse_date(created_at_str)
        updated_dt = parse_date(updated_at_str)
        superseded_dt = parse_date(superseded_at_str)
        deleted_dt = parse_date(deleted_at_str)

        # Classify changes today
        is_new = created_dt and created_dt > one_day_ago
        is_updated = updated_dt and updated_dt > one_day_ago and not is_new
        is_superseded = superseded_dt and superseded_dt > one_day_ago
        is_deleted = deleted_dt and deleted_dt > one_day_ago

        if is_new and status == "active":
            added_today.append(text)
        elif is_updated and status == "active":
            updated_today.append(text)
        elif is_superseded:
            winner_id = payload.get("superseded_by", "")
            winner_suffix = f" (заменено на [{winner_id[:8]}])" if winner_id else ""
            added_today.append(f"~~{text}~~{winner_suffix}")
        elif is_deleted:
            deleted_today.append(text)

    # Format report
    print(f"🏠 **Домовой порядок: Отчёт по памяти Нафани за {datetime.now().strftime('%d.%m.%Y')}**\n")
    print(f"📊 **Общая статистика базы:**")
    print(f"• Всего записей в БД: **{total_count}**")
    print(f"• Активных воспоминаний: **{active_count}**")
    print(f"• Устаревших (superseded) фактов в архиве: **{superseded_count}**")
    print(f"• Логически удалённых: **{deleted_count}**\n")

    print(f"🔄 **Операции за последние 24 часа:**")
    if not (added_today or updated_today or deleted_today):
        print("• За день изменений не зарегистрировано. Память стабильна. 🙂")
        return

    if added_today:
        print("\n➕ **Добавленные / Устаревшие за день факты:**")
        for txt in added_today:
            print(f" - {txt}")

    if updated_today:
        print("\n✏️ **Обновлённые / Консолидированные факты:**")
        for txt in updated_today:
            print(f" - {txt}")

    if deleted_today:
        print("\n❌ **Удалённые факты:**")
        for txt in deleted_today:
            print(f" - {txt}")

if __name__ == "__main__":
    print_reflection_report()
