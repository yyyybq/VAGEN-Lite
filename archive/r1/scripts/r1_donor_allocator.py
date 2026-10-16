#!/usr/bin/env python3
"""Deterministic, resumable donor reservation for R1 replacements."""

from __future__ import annotations

import json
import os
import tempfile
import time
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
        # v1 stored donor->owner strings.  Keep that format readable while
        # v2 stores lifecycle metadata beside the compatibility map.
        reservations = payload.get("reservations", {})
        if payload.get("version") == "r1_donor_ledger_v2":
            reservations = {
                key: value.get("owner")
                for key, value in reservations.items()
                if isinstance(value, dict) and value.get("owner") is not None
            }
        return DonorAllocator(
            {_key(key.split("\u001f")): str(value) for key, value in reservations.items()}
        )

    def _payload(self) -> dict:
        if not self.path.is_file():
            return {"version": "r1_donor_ledger_v2", "reservations": {}, "history": []}
        payload = json.loads(self.path.read_text())
        if payload.get("version") == "r1_donor_ledger_v2":
            payload.setdefault("history", [])
            return payload
        # Upgrade a v1 map without inventing a validation result.
        return {
            "version": "r1_donor_ledger_v2",
            "reservations": {
                key: {"owner": str(owner), "attempt": "legacy", "state": "reserved", "updated_at": 0.0}
                for key, owner in payload.get("reservations", {}).items()
            },
            "history": [{"event": "upgrade_v1", "at": time.time()}],
        }

    def _write(self, allocator: DonorAllocator, payload: dict | None = None) -> None:
        payload = payload or self._payload()
        payload["version"] = "r1_donor_ledger_v2"
        payload["reservations"] = {
            "\u001f".join(key): {
                "owner": owner,
                "attempt": payload.get("reservations", {}).get("\u001f".join(key), {}).get("attempt", "legacy"),
                "state": payload.get("reservations", {}).get("\u001f".join(key), {}).get("state", "reserved"),
                "updated_at": time.time(),
            }
            for key, owner in sorted(allocator.reservations.items())
        }
        payload.setdefault("history", [])
        payload = {
            **payload,
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

    def _mutate(self, callback):
        with self.lock_path.open("a+") as lock_handle:
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
            payload = self._payload()
            allocator = self._read()
            result = callback(allocator, payload)
            self._write(allocator, payload)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            return result

    def reserve(self, source_index: int | str, candidates: Iterable[DonorKey], attempt: str | None = None) -> DonorKey | None:
        owner = str(source_index)
        attempt = attempt or f"{owner}:{int(time.time() * 1000)}"
        def callback(allocator, payload):
            selected = allocator.reserve(owner, candidates)
            if selected is not None:
                key = "\u001f".join(selected)
                payload["reservations"][key] = {
                    "owner": owner, "attempt": attempt, "state": "reserved", "updated_at": time.time()
                }
                payload.setdefault("history", []).append({"event": "reserved", "owner": owner, "attempt": attempt, "donor": key, "at": time.time()})
            return selected
        return self._mutate(callback)

    def seed(self, source_index: int | str, donor: DonorKey) -> None:
        owner = str(source_index)
        def callback(allocator, payload):
            allocator.seed(owner, donor)
            key = "\u001f".join(_key(donor))
            payload["reservations"][key] = {
                "owner": owner, "attempt": "resume", "state": "validation_pending", "updated_at": time.time()
            }
        self._mutate(callback)

    def _transition(self, source_index: int | str, donor: DonorKey, state: str, attempt: str | None = None) -> None:
        owner = str(source_index)
        key = "\u001f".join(_key(donor))
        def callback(allocator, payload):
            if allocator.reservations.get(_key(donor)) != owner:
                raise ValueError(f"donor {donor} is not owned by source {owner}")
            current = payload["reservations"].get(key, {})
            if attempt is not None and current.get("attempt") not in (None, attempt):
                raise ValueError(f"attempt mismatch for donor {donor}: {current.get('attempt')} != {attempt}")
            current.update({"owner": owner, "attempt": attempt or current.get("attempt", "legacy"), "state": state, "updated_at": time.time()})
            payload["reservations"][key] = current
            payload.setdefault("history", []).append({"event": state, "owner": owner, "attempt": current["attempt"], "donor": key, "at": time.time()})
        self._mutate(callback)

    def mark_validation_pending(self, source_index: int | str, donor: DonorKey, attempt: str | None = None) -> None:
        self._transition(source_index, donor, "validation_pending", attempt)

    def commit(self, source_index: int | str, donor: DonorKey, attempt: str | None = None) -> None:
        self._transition(source_index, donor, "committed", attempt)

    def release(self, source_index: int | str, donor: DonorKey, attempt: str | None = None, reason: str = "released") -> None:
        owner = str(source_index)
        key = "\u001f".join(_key(donor))
        def callback(allocator, payload):
            current = payload.get("reservations", {}).get(key)
            if allocator.reservations.get(_key(donor)) != owner:
                return
            if attempt is not None and current and current.get("attempt") not in (None, attempt):
                raise ValueError(f"attempt mismatch for donor {donor}: {current.get('attempt')} != {attempt}")
            allocator.release(owner, donor)
            payload.get("reservations", {}).pop(key, None)
            payload.setdefault("history", []).append({"event": "released", "reason": reason, "owner": owner, "attempt": (current or {}).get("attempt", attempt or "legacy"), "donor": key, "at": time.time()})
        self._mutate(callback)

    def recover_owner(self, source_index: int | str, reason: str, keep_states: set[str] | None = None) -> list[DonorKey]:
        """Release owner reservations unless explicitly retained by scheduler evidence."""
        owner = str(source_index)
        keep_states = keep_states or set()
        def callback(allocator, payload):
            released = []
            for key, record in list(payload.get("reservations", {}).items()):
                if not isinstance(record, dict) or record.get("owner") != owner or record.get("state") in keep_states:
                    continue
                donor = _key(key.split("\u001f"))
                allocator.release(owner, donor)
                payload["reservations"].pop(key, None)
                payload.setdefault("history", []).append({"event": "released", "reason": reason, "owner": owner, "attempt": record.get("attempt"), "donor": key, "at": time.time()})
                released.append(donor)
            return released
        return self._mutate(callback)

    def states(self) -> dict[str, dict]:
        with self.lock_path.open("a+") as lock_handle:
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
            payload = self._payload()
            allocator = self._read()
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return dict(payload.get("reservations", {}))

    def snapshot(self) -> dict[str, int]:
        with self.lock_path.open("a+") as lock_handle:
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
            payload = json.loads(self.path.read_text()) if self.path.is_file() else {"reservations": {}}
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return {
            key: (value.get("owner") if isinstance(value, dict) else value)
            for key, value in payload.get("reservations", {}).items()
        }
