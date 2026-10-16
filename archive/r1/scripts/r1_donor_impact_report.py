#!/usr/bin/env python3
"""Account for stale donor reservations and observed conflict skips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def owner_of(value):
    return value.get("owner") if isinstance(value, dict) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ledger", type=Path, required=True)
    parser.add_argument("--new-ledger", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.old_ledger.read_text()).get("reservations", {})
    new = json.loads(args.new_ledger.read_text()).get("reservations", {})
    old_map = {key: owner_of(value) for key, value in old.items()}
    new_map = {key: owner_of(value) for key, value in new.items()}
    conflicts = []
    for event_path in args.jobs_root.glob("*/repair/*/stage_events.jsonl"):
        for line in event_path.open():
            event = json.loads(line)
            if event.get("reason") == "donor_reservation_conflict":
                conflicts.append({
                    "owner": f"{event['split']}:{event['source_row_index']}",
                    "donor_source": f"{event.get('donor_split')}:{event.get('donor_source_row_index')}",
                    "event": str(event_path),
                })
    payload = {
        "version": "r1_donor_impact_v1",
        "old_reservations": len(old_map),
        "new_reservations": len(new_map),
        "stale_reservations": [
            {"donor": donor, "old_owner": owner}
            for donor, owner in sorted(old_map.items()) if donor not in new_map
        ],
        "observed_conflict_skips": conflicts,
        "affected_owner_count": len({row["owner"] for row in conflicts}),
        "old_owner_count": len(set(old_map.values())),
        "new_owner_count": len(set(new_map.values())),
        "new_donor_unique": len(new_map) == len(set(new_map)),
        "new_owner_unique": len(new_map.values()) == len(set(new_map.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
