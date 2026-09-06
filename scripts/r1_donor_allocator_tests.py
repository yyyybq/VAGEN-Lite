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


if __name__ == "__main__":
    unittest.main()
