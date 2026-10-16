#!/usr/bin/env python3
"""Tests for serial, concurrent, and resumable donor allocation."""

from __future__ import annotations

import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from r1_donor_allocator import AtomicDonorLedger, DonorAllocator


class DonorAllocatorTests(unittest.TestCase):
    def test_serial_and_resume_are_deterministic(self):
        allocator = DonorAllocator()
        first = allocator.reserve(10, [("train", "scene", "a"), ("train", "scene", "b")])
        second = allocator.reserve(11, [("train", "scene", "a"), ("train", "scene", "b")])
        self.assertEqual(first, ("train", "scene", "a"))
        self.assertEqual(second, ("train", "scene", "b"))
        self.assertEqual(allocator.reserve(10, [("train", "scene", "a")]), first)

    def test_atomic_parallel_reservation_has_no_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AtomicDonorLedger(Path(directory) / "donors.json")
            candidates = [("train", "scene", "a"), ("train", "scene", "b")]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda index: ledger.reserve(index, candidates), (1, 2)))
            self.assertEqual(set(results), set(candidates))
            self.assertEqual(len(ledger.snapshot()), 2)

    def test_resume_preserves_owner_and_rejects_cross_split_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AtomicDonorLedger(Path(directory) / "donors.json")
            donor = ("train", "scene", "pair")
            self.assertEqual(ledger.reserve(4, [donor]), donor)
            self.assertEqual(ledger.reserve(4, [donor]), donor)
            self.assertIsNone(ledger.reserve(5, [donor]))
            ledger.release(4, donor)
            self.assertEqual(ledger.reserve(5, [donor]), donor)

    def test_lifecycle_attempt_state_and_rgb_release(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AtomicDonorLedger(Path(directory) / "donors.json")
            donor = ("id_test", "scene", "pair")
            attempt = "id_test:4:attempt7"
            self.assertEqual(ledger.reserve("id_test:4", [donor], attempt=attempt), donor)
            self.assertEqual(ledger.states()["\u001f".join(donor)]["state"], "reserved")
            ledger.mark_validation_pending("id_test:4", donor, attempt=attempt)
            self.assertEqual(ledger.states()["\u001f".join(donor)]["state"], "validation_pending")
            ledger.commit("id_test:4", donor, attempt=attempt)
            self.assertEqual(ledger.states()["\u001f".join(donor)]["state"], "committed")
            with self.assertRaises(ValueError):
                ledger.release("id_test:4", donor, attempt="wrong", reason="stale_worker")
            ledger.release("id_test:4", donor, attempt=attempt, reason="rgb_observability_rejected")
            self.assertNotIn("\u001f".join(donor), ledger.states())

    def test_scheduler_recovery_keeps_pending_evidence_only(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = AtomicDonorLedger(Path(directory) / "donors.json")
            pending = ("train", "scene", "pending")
            reserved = ("train", "scene", "reserved")
            ledger.reserve("train:1", [pending], attempt="train:1:attempt1")
            ledger.mark_validation_pending("train:1", pending, attempt="train:1:attempt1")
            ledger.reserve("train:2", [reserved], attempt="train:2:attempt1")
            released = ledger.recover_owner("train:2", "worker_timeout", keep_states={"validation_pending"})
            self.assertEqual(released, [reserved])
            self.assertIn("\u001f".join(pending), ledger.snapshot())
            self.assertNotIn("\u001f".join(reserved), ledger.snapshot())


if __name__ == "__main__":
    unittest.main()
