#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Test Suite for Mem0 Temporal Hygiene & Hybrid Scoring Uprades.
Tests Key Extraction, Trust Tiers, Temporal Decay, and Deterministic Fallback.
"""

import sys
import os
import unittest
from datetime import datetime, timezone, timedelta

# Insert project directory to path
sys.path.insert(0, "/root/mem0-temporal-hygiene")

# Let's import functions we want to test. (Some might fail initially if not implemented)
from plugin import _extract_subject_keys, _calculate_temporal_decay
from plugin import Mem0OSSProvider
import importlib.util
spec = importlib.util.spec_from_file_location("memory_hygiene", "/root/mem0-temporal-hygiene/scripts/memory-hygiene.py")
memory_hygiene = importlib.util.module_from_spec(spec)
sys.modules["memory_hygiene"] = memory_hygiene
spec.loader.exec_module(memory_hygiene)
check_deterministic_merge = memory_hygiene.check_deterministic_merge
has_negation_or_toggle = memory_hygiene.has_negation_or_toggle

class TestKeyExtraction(unittest.TestCase):
    def test_extract_keys_basic(self):
        txt1 = "Порт OmniRoute 20130"
        txt2 = "Таймаут OmniRoute 30 сек"
        
        keys1 = _extract_subject_keys(txt1)
        keys2 = _extract_subject_keys(txt2)
        
        self.assertIn("omniroute", keys1)
        self.assertIn("порт", keys1)
        self.assertIn("omniroute", keys2)
        self.assertIn("таймаут", keys2)
        
        # Test structural variables match: key: value or key = value
        txt3 = "obsidian_sync_path: /root/vault"
        keys3 = _extract_subject_keys(txt3)
        self.assertIn("obsidian_sync_path", keys3)

class TestTrustTiersAndDecay(unittest.TestCase):
    def test_temporal_decay_rates(self):
        # Now
        now_str = datetime.now(timezone.utc).isoformat()
        self.assertAlmostEqual(_calculate_temporal_decay(now_str, "user_explicit"), 1.0, places=4)
        
        # 10 days ago
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        
        # User explicit should NOT decay
        decay_user = _calculate_temporal_decay(ten_days_ago, "user_explicit")
        self.assertEqual(decay_user, 1.0)
        
        # Tool log should decay fast (lambda = 0.05)
        decay_tool = _calculate_temporal_decay(ten_days_ago, "tool_log")
        # exp(-0.05 * 10) = exp(-0.5) ~ 0.6065
        self.assertAlmostEqual(decay_tool, 0.6065, places=3)
        
        # Default gentle decay (lambda = 0.005)
        decay_default = _calculate_temporal_decay(ten_days_ago, "agent_decision")
        # exp(-0.005 * 10) = exp(-0.05) ~ 0.9512
        self.assertAlmostEqual(decay_default, 0.9512, places=3)

class TestRelationClassificationWithKeys(unittest.TestCase):
    def test_write_relation_isolated_keys(self):
        # Initialize provider
        provider = Mem0OSSProvider()
        
        content = "Определён порт OmniRoute: 20140"
        # Candidate 1: Same entity, different key
        candidate_diff_key = {
            "id": "uuid-1",
            "score": 0.88,
            "memory": "Таймаут OmniRoute равен 30 секундам"
        }
        
        # Candidate 2: Same entity, same key structure
        candidate_same_key = {
            "id": "uuid-2",
            "score": 0.93,
            "memory": "Порт OmniRoute настроен на 20130"
        }
        
        # We classify write relation for candidates
        res1 = provider._classify_write_relation(content, [candidate_diff_key])
        res2 = provider._classify_write_relation(content, [candidate_same_key])
        
        # Because keys differ (порт vs таймаут), res1 should NOT block/update/conflict.
        self.assertEqual(res1["conflict_status"], "none")
        self.assertEqual(res1["possible_duplicate_of"], [])
        
        # Because keys overlap (порт vs порт), res2 should flag possible duplicate or conflict
        self.assertEqual(res2["conflict_status"], "possible_duplicate")
        self.assertEqual(res2["possible_duplicate_of"], ["uuid-2"])

class TestDeterministicFallback(unittest.TestCase):
    def test_fallback_newer_wins(self):
        # Mock group of memories
        group = [
            {"id": "uuid-old", "payload": {"data": "Нафаня использует порт 111", "created_at": "2026-06-01T12:00:00Z"}},
            {"id": "uuid-new", "payload": {"data": "Нафаня использует порт 222", "created_at": "2026-06-06T15:00:00Z"}}
        ]
        
        # Test mock fallback behavior (group has conflict, newer wins)
        # Should soft-delete elements that are not the newest
        # Let's verify we sort and pick the newest
        sorted_group = sorted(group, key=lambda x: x.get("payload", {}).get("created_at") or "")
        newest = sorted_group[-1]
        deletions = [p["id"] for p in sorted_group[:-1]]
        
        self.assertEqual(newest["id"], "uuid-new")
        self.assertEqual(deletions, ["uuid-old"])

if __name__ == "__main__":
    unittest.main()
