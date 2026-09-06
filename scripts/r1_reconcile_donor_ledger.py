#!/usr/bin/env python3
"""Reconcile reservations after downstream stages reject a generated row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    owners = {
        f"{row['split']}:{row['source_row_index']}"
        for line in args.sample_manifest.open()
        for row in [json.loads(line)]
        if row.get("stage", {}).get("final_accepted")
        and row.get("generation_status") == "count_matched_replacement"
    }
    reservations = {
        donor: owner
        for donor, owner in ledger.get("reservations", {}).items()
        if owner in owners
    }
    payload = {
        "version": "r1_donor_ledger_reconciled_v1",
        "source_ledger": str(args.ledger),
        "retained_final_accepted_owners": sorted(owners),
        "reservations": reservations,
        "released_reservations": len(ledger.get("reservations", {})) - len(reservations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
