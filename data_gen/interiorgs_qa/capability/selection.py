"""Select an unchanged, globally balanced subset of capability questions on CPU.

Both semantic answer keys and displayed answer letters are constrained within
each question kind. Rearranging answer letters cannot fix a semantic imbalance.
The solver never fills a shortage by relabelling or relaxing a quality rule.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

VERSION = "capability_global_selection_v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _key(value):
    return value if isinstance(value, str) else _canonical(value)


def _identity(row):
    for field in ("qid", "question_id", "id", "sample_id"):
        if row.get(field) is not None:
            return str(row[field])
    return hashlib.sha256(_canonical(row).encode()).hexdigest()


def _normalize(rows, families=None):
    requested = None if families is None else set(families)
    unique, dropped, outside = {}, 0, 0
    kind_options = {}
    for row in rows:
        if requested is not None and row["group"] not in requested:
            outside += 1
            continue
        for field in ("group", "kind", "scene_id", "parent_task_id", "answer", "answer_key",
                      "options", "images", "template_id"):
            if field not in row:
                raise ValueError("Missing question field: " + field)
        if not isinstance(row["options"], (list, tuple)) or not 2 <= len(row["options"]) <= 26:
            raise ValueError("options must contain at least two choices")
        if row["answer"] not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(row["options"])]):
            raise ValueError("answer must be one valid option letter")
        if not isinstance(row["images"], (list, tuple)) or not row["images"]:
            raise ValueError("questions require their original image list")
        if row["answer_key"] is None:
            raise ValueError("Missing semantic answer key")
        kind = row["kind"]
        n_options = len(row["options"])
        if kind in kind_options and kind_options[kind] != n_options:
            raise ValueError("One kind has inconsistent choice counts: " + str(kind))
        kind_options[kind] = n_options
        identity = _identity(row)
        if identity in unique:
            if _canonical(unique[identity]) != _canonical(row):
                raise ValueError("Conflicting records use the same question ID: " + identity)
            dropped += 1
        else:
            unique[identity] = row
    return unique, kind_options, dropped, outside


def balance_report(rows, kinds=None):
    """Verify integer counts directly, including empty kinds; no rounding of percentages."""
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    output = {}
    for kind in sorted(set(kinds or ()) | set(by_kind)):
        subset = by_kind[kind]
        if not subset:
            output[kind] = {"selected": 0, "status": "empty", "valid": True,
                            "semantic_counts": {}, "letter_counts": {}, "max_share": None}
            continue
        choices = {len(r["options"]) for r in subset}
        if len(choices) != 1:
            raise ValueError("Inconsistent number of choices")
        n_options = choices.pop()
        percent = 55 if n_options == 2 else 40
        semantic = Counter(_key(r["answer_key"]) for r in subset)
        letters = Counter(r["answer"] for r in subset)
        n = len(subset)
        valid = all(100 * c <= percent * n for c in list(semantic.values()) + list(letters.values()))
        output[kind] = {"selected": n, "choices": n_options, "maximum_allowed_share": percent / 100,
                        "semantic_counts": dict(sorted(semantic.items())), "letter_counts": dict(sorted(letters.items())),
                        "max_semantic_share": max(semantic.values()) / n,
                        "max_letter_share": max(letters.values()) / n,
                        "status": "balanced" if valid else "violation", "valid": valid}
    return output


def select_questions(rows, *, families=None, family_cap=20, total_cap=100, parent_cap=8,
                     seed=20261016, time_limit_s=60):
    """Return (selected_original_rows, report), with no mutation or answer permutation.

    ``families`` lists the expected parent task families so absent ones still get
    a shortage entry. Stage one maximizes the number of questions. At that fixed
    number, stage two discourages concentration across families and parents and
    prefers covering different scenes within each family. A timeout is reported;
    any returned incumbent is independently checked against every hard limit.
    """
    if min(family_cap, total_cap, parent_cap) < 1 or time_limit_s <= 0:
        raise ValueError("Caps and time_limit_s must be positive")
    rows = list(rows)
    unique, kind_options, duplicates, outside = _normalize(rows, families)
    # A content-based order makes results independent of input file ordering.
    ordered = sorted(unique, key=lambda k: (hashlib.sha256(f"{seed}|{k}".encode()).hexdigest(), k))
    bank = [unique[k] for k in ordered]
    n = len(bank)
    groups = sorted(set(families or ()) | {r["group"] for r in bank})
    by_group, by_parent, by_scene, by_kind = (defaultdict(list) for _ in range(4))
    for i, row in enumerate(bank):
        by_group[row["group"]].append(i)
        by_parent[row["parent_task_id"]].append(i)
        by_scene[(row["group"], row["scene_id"])].append(i)
        by_kind[row["kind"]].append(i)
    report = {"version": VERSION, "seed": seed, "input_rows": len(rows), "unique_candidates": n,
              "duplicate_rows_removed": duplicates, "outside_requested_families": outside,
              "limits": {"per_family": family_cap, "total": total_cap, "per_parent": parent_cap,
                         "binary_answer_share": 0.55, "multiple_choice_answer_share": 0.40},
              "quality_relaxed": False, "answers_changed": False,
              "selection_priority": ["maximum question count", "less family concentration",
                                     "family-scene coverage", "less parent concentration", "stable seeded ordering"]}
    chosen = []
    solver_runs = []
    maximum_proven = True
    if n:
        # Sparse integer constraints; each answer count <= 40/55% of its kind.
        entries, lower, upper = [], [], []
        preference = [0.0001 * (i + 1) / (n + 1) for i in range(n)]
        def variable(cost):
            preference.append(float(cost))
            return len(preference) - 1
        def constrain(coefficients, lo=-np.inf, hi=np.inf):
            entries.append(coefficients)
            lower.append(lo)
            upper.append(hi)
        constrain({i: 1 for i in range(n)}, hi=total_cap)
        # Auxiliary slots express increasing costs for repeated selections.
        for indices in by_group.values():
            slots = [variable(100000 * (2 * j - 1)) for j in range(1, min(family_cap, len(indices)) + 1)]
            constrain({**{i: 1 for i in indices}, **{i: -1 for i in slots}}, 0, 0)
        for indices in by_parent.values():
            slots = [variable(2 * j - 1) for j in range(1, min(parent_cap, len(indices)) + 1)]
            constrain({**{i: 1 for i in indices}, **{i: -1 for i in slots}}, 0, 0)
        for indices in by_scene.values():
            covered = variable(-1000)
            constrain({**{i: 1 for i in indices}, covered: -1}, lo=0)
            constrain({**{i: 1 for i in indices}, covered: -len(indices)}, hi=0)
        for kind, indices in by_kind.items():
            percent = 55 if kind_options[kind] == 2 else 40
            for field in ("answer_key", "answer"):
                for key in sorted({_key(bank[i][field]) for i in indices}):
                    constrain({i: (100 if _key(bank[i][field]) == key else 0) - percent for i in indices}, hi=0)
        rr, cc, vv = [], [], []
        for r, terms in enumerate(entries):
            for c, value in terms.items():
                rr.append(r); cc.append(c); vv.append(value)
        matrix = coo_matrix((np.asarray(vv, float), (rr, cc)), shape=(len(entries), len(preference))).tocsc()
        bounds = Bounds(np.zeros(len(preference)), np.ones(len(preference)))
        integrality = np.ones(len(preference), dtype=int)
        objective = np.zeros(len(preference)); objective[:n] = -1
        options = {"time_limit": float(time_limit_s), "mip_rel_gap": 0.0}
        first = milp(objective, integrality=integrality, bounds=bounds,
                     constraints=LinearConstraint(matrix, lower, upper), options=options)
        solver_runs.append({"stage": "maximum_count", "status": int(first.status), "message": str(first.message),
                            "optimal": bool(first.success), "mip_gap": float(first.mip_gap) if first.get("mip_gap") is not None else None})
        maximum_proven = bool(first.success)
        if first.x is not None:
            incumbent = first.x
            target = int(np.rint(first.x[:n]).sum())
            count_row = coo_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(n))),
                                   shape=(1, len(preference))).tocsc()
            second = milp(np.asarray(preference), integrality=integrality, bounds=bounds,
                          constraints=LinearConstraint(vstack([matrix, count_row], format="csc"),
                                                       np.append(lower, target), np.append(upper, target)), options=options)
            solver_runs.append({"stage": "coverage_and_concentration", "status": int(second.status),
                                "message": str(second.message), "optimal": bool(second.success)})
            if second.x is not None:
                incumbent = second.x
            if not np.allclose(incumbent[:n], np.rint(incumbent[:n]), atol=1e-5, rtol=0):
                raise RuntimeError("Solver returned a non-integer question selection")
            chosen = [bank[i] for i in range(n) if incumbent[i] > 0.5]
    checks = balance_report(chosen, by_kind)
    group_counts = Counter(r["group"] for r in chosen)
    parent_counts = Counter(r["parent_task_id"] for r in chosen)
    if (len(chosen) > total_cap or max(group_counts.values(), default=0) > family_cap
            or max(parent_counts.values(), default=0) > parent_cap
            or not all(c["valid"] for c in checks.values()) or len({_identity(r) for r in chosen}) != len(chosen)):
        raise RuntimeError("Solver selection failed independent integer-count verification")
    report.update(selected=len(chosen), total_shortfall=max(0, min(total_cap, family_cap * len(groups)) - len(chosen)),
                  maximum_size_proven=maximum_proven, solver_runs=solver_runs, balance=checks,
                  balance_valid=True, maximum_parent_count=max(parent_counts.values(), default=0),
                  selected_ids=[_identity(r) for r in chosen])
    report["families"] = {}
    for family in groups:
        subset = [r for r in chosen if r["group"] == family]
        eligible = [bank[i] for i in by_group[family]]
        count = len(subset)
        shortage = max(0, family_cap - count)
        report["families"][family] = {"available": len(eligible), "selected": count,
            "target": family_cap, "shortfall": shortage,
            "reason": None if not shortage else ("no_candidates" if not eligible else
                       "solver_limit_unresolved" if not maximum_proven else
                       "candidate_shortage" if len(eligible) < family_cap else "balance_parent_or_total_limit"),
            "available_by_scene": dict(sorted(Counter(r["scene_id"] for r in eligible).items())),
            "selected_by_scene": dict(sorted(Counter(r["scene_id"] for r in subset).items())),
            "selected_parent_count": len({r["parent_task_id"] for r in subset})}
    report["kinds"] = {kind: {"available": len(indices), "selected": checks[kind]["selected"],
                              "available_semantic_counts": dict(sorted(Counter(_key(bank[i]["answer_key"]) for i in indices).items())),
                              "available_letter_counts": dict(sorted(Counter(bank[i]["answer"] for i in indices).items()))}
                       for kind, indices in sorted(by_kind.items())}
    return chosen, report


def estimate_capacity(bank_rows, selected_rows=None, *, act_image_inputs=25243, scene_ids=None, families=None):
    """Describe this observed bank only; scale estimates are unverified assumptions.

    Raw-bank counts precede global answer balancing and parent caps, so their
    extrapolation is explicitly an optimistic capacity bound, not proven yield.
    Supplying a selected subset adds a separate actually-selected pilot rate.
    """
    bank_rows = list(bank_rows)
    unique, _, duplicates, _ = _normalize(bank_rows, families)
    bank = list(unique.values())
    scenes = sorted(set(scene_ids or ()) | {r["scene_id"] for r in bank})
    groups = sorted(set(families or ()) | {r["group"] for r in bank})
    if act_image_inputs <= 0:
        raise ValueError("Act image-input budget must be positive")
    result = {"scenes": scenes, "scene_count": len(scenes), "candidate_questions": len(bank),
              "duplicate_rows_removed": duplicates, "act_image_inputs_per_epoch": act_image_inputs,
              "families": {}, "extrapolation_verified": False,
              "scope": "Observed geometry/depth-admitted candidates only; they are not human-approved or all globally balanced.",
              "scaling_caveat": "Four-scene rates do not establish yield in other scenes; repeated input images count once per use, matching the Act-SFT budget."}
    for group in groups:
        counts = Counter(r["scene_id"] for r in bank if r["group"] == group)
        values = [counts[s] for s in scenes]
        result["families"][group] = {"per_scene": {s: counts[s] for s in scenes},
            "mean_per_scene": sum(values) / len(values) if values else None,
            "min_per_scene": min(values) if values else None, "max_per_scene": max(values) if values else None}
    def scaling(rows, kind):
        inputs = sum(len(r["images"]) for r in rows)
        per_scene = inputs / len(scenes) if scenes else 0
        return {"basis": kind, "questions": len(rows), "image_input_occurrences": inputs,
                "mean_image_inputs_per_scene": per_scene,
                "estimated_total_scenes_for_act_budget": math.ceil(act_image_inputs / per_scene) if per_scene else None,
                "verified_outside_pilot": False}
    result["raw_bank_optimistic_scaling"] = scaling(bank, "unbalanced candidate upper bound")
    if selected_rows is not None:
        selected_rows = list(selected_rows)
        bank_ids = set(unique)
        if (len({_identity(r) for r in selected_rows}) != len(selected_rows)
                or any(_identity(r) not in bank_ids or _canonical(unique[_identity(r)]) != _canonical(r)
                       for r in selected_rows)):
            raise ValueError("Selected rows are not a subset of this bank")
        result["selected_pilot_scaling"] = scaling(selected_rows, "actual selected pilot rate, including its sample quota")
    return result
