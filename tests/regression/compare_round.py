#!/usr/bin/env python3
"""Compare two rounds written by check_round.sh (cleanup equivalence check).

    python tests/regression/compare_round.py <before round dir> <after round dir> [--moved-ok FILE] [--added-ok FILE]

Tests are matched by (test file name, class, test name), not by path, so a test that only moved directory still
matches. Reports, for suite A and suite B: counts per outcome, tests only in one round, tests whose outcome changed.
Then runs compare_golden.py on the two golden/ folders. Exit 0 when no test changed outcome, every test only in the
before round is in a test file listed in --moved-ok (archived tests), every test only in the after round is in a file
listed in --added-ok (tests the batch added) and passes, and golden is equivalent.
"""
from __future__ import annotations

import collections
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def outcomes(xml: Path) -> dict:
    out: dict = {}
    if not xml.is_file():
        return out
    for tc in ET.parse(xml).getroot().iter("testcase"):
        cls = tc.get("classname", "")
        parts = cls.split(".")
        # classname = dotted module path [+ class]; the module is the last part starting with "test_"
        mod_i = max([i for i, p in enumerate(parts) if p.startswith("test_") or p.endswith("_test")] or [len(parts) - 1])
        key = (parts[mod_i], ".".join(parts[mod_i + 1:]), tc.get("name"))
        state = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                state = "failed"
            elif child.tag == "skipped":
                state = "skipped"
        while key in out:  # same file name in two directories: keep both
            key = key + ("dup",)
        out[key] = state
    return out


def main() -> int:
    args = sys.argv[1:]
    moved_ok: set = set()
    if "--moved-ok" in args:
        i = args.index("--moved-ok")
        moved_ok = {l.strip() for l in Path(args[i + 1]).read_text().splitlines() if l.strip()}
        del args[i:i + 2]
    added_ok: set = set()
    if "--added-ok" in args:
        i = args.index("--added-ok")
        added_ok = {l.strip() for l in Path(args[i + 1]).read_text().splitlines() if l.strip()}
        del args[i:i + 2]
    a, b = Path(args[0]), Path(args[1])
    bad = 0
    for suite in ("suiteA", "suiteB"):
        oa, ob = outcomes(a / f"{suite}.xml"), outcomes(b / f"{suite}.xml")
        ca, cb = collections.Counter(oa.values()), collections.Counter(ob.values())
        print(f"== {suite}: before {dict(ca)} ({len(oa)})   after {dict(cb)} ({len(ob)})")
        only_a = sorted(set(oa) - set(ob)); only_b = sorted(set(ob) - set(oa))
        changed = sorted(k for k in set(oa) & set(ob) if oa[k] != ob[k])
        unexplained = [k for k in only_a if k[0] not in moved_ok]
        print(f"   only before: {len(only_a)} (archived test files: {len(only_a) - len(unexplained)}, other: {len(unexplained)})")
        for k in unexplained[:30]:
            print("     -", "::".join(x for x in k if x), oa[k])
        added = [k for k in only_b if k[0] in added_ok and ob[k] == "passed"]
        extra = [k for k in only_b if k not in added]
        print(f"   only after: {len(only_b)} (new tests of this batch, passing: {len(added)}, other: {len(extra)})")
        for k in extra[:30]:
            print("     +", "::".join(x for x in k if x), ob[k])
        print(f"   outcome changed: {len(changed)}")
        for k in changed[:30]:
            print("     *", "::".join(x for x in k if x), oa[k], "->", ob[k])
        bad += len(unexplained) + len(extra) + len(changed)
    print("== golden")
    g = subprocess.run([sys.executable, str(HERE / "compare_golden.py"), str(a / "golden"), str(b / "golden")],
                       capture_output=True, text=True)
    print(g.stdout.strip().splitlines()[-1] if g.stdout.strip() else g.stderr.strip())
    for line in g.stdout.splitlines():
        if line.startswith(("DIFFERS", "only in")):
            print("   ", line)
    bad += g.returncode != 0
    print("RESULT:", "equivalent" if bad == 0 else f"{bad} differences")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
