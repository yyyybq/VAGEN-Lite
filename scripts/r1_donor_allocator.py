#!/usr/bin/env python3
"""Deterministic, resumable donor reservation for R1 replacements."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import fcntl


DonorKey = tuple[str, ...]


def _key(value: Iterable[str]) -> DonorKey:
    return tuple(str(part) for part in value)


class DonorAllocator:
    """In-memory deterministic allocator used by serial and parallel callers."""

    def __init__(self, reservations: dict[DonorKey, str] | None = None) -> None:
        self.reservations: dict[DonorKey, str] = dict(reservations or {})

    def reserve(self, source_index: int | str, candidates: Iterable[DonorKey]) -> DonorKey | None:
        owner_key = str(source_index)
        for candidate in sorted({_key(value) for value in candidates}):
            owner = self.reservations.get(candidate)
            if owner is None or owner == owner_key:
                self.reservations[candidate] = owner_key
                return candidate
        return None

    def seed(self, source_index: int | str, donor: DonorKey) -> None:
        donor = _key(donor)
        owner_key = str(source_index)
        owner = self.reservations.get(donor)
        if owner is not None and owner != owner_key:
            raise ValueError(f"donor {donor} already reserved by source {owner}")
        self.reservations[donor] = owner_key

    def release(self, source_index: int | str, donor: DonorKey) -> None:
        donor = _key(donor)
        owner_key = str(source_index)
        if self.reservations.get(donor) == owner_key:
            del self.reservations[donor]


class AtomicDonorLedger:
    """File-backed allocator with a lock and atomic replacement of its JSON."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> DonorAllocator:
        if not self.path.is_file():
            return DonorAllocator()
        payload = json.loads(self.path.read_text())
        return DonorAllocator(
            {_key(key.split("\u001f")): str(value) for key, value in payload.get("reservations", {}).items()}
        )

    def _write(self, allocator: DonorAllocator) -> None:
        payload = {
            "version": "r1_donor_ledger_v1",
            "reservations": {
                "\u001f".join(key): owner
                for key, owner in sorted(allocator.reservations.items())
            },
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=str(self.path.parent), text=True
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def reserve(self, source_index: int | str, candidates: Iterable[DonorKey]) -> DonorKey | None:
        with self.lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            allocator = self._read()
            selected = allocator.reserve(source_index, candidates)
            if selected is not None:
                self._write(allocator)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            return selected

    def seed(self, source_index: int | str, donor: DonorKey) -> None:
        with self.lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            allocator = self._read()
            allocator.seed(source_index, donor)
            self._write(allocator)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def release(self, source_index: int | str, donor: DonorKey) -> None:
        with self.lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            allocator = self._read()
            allocator.release(source_index, donor)
            self._write(allocator)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def snapshot(self) -> dict[str, int]:
        with self.lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            payload = json.loads(self.path.read_text()) if self.path.is_file() else {"reservations": {}}
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return dict(payload.get("reservations", {}))
