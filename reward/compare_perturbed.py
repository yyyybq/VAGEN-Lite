"""Reward line: the perturbed-start evaluation's comparisons, exactly as docs/reward/PREREG_PERTURB.md fixed them
before the evaluation ran (the user's 3 of 2026-11-05).

Sets: the four perturbed-start sets, named {val|test}_k{k} (each a folder with tasks.jsonl and one result folder per
model); every k needs both a val and a test set, and each folder must hold what its name says (its rows' split and
perturbation.k, each task once).  A task's margin is LIMIT - (d* + k), its new d_star being d* + k; bands 0, 1-2, 3-4,
5-; margin-0 tasks stay out of the main and the secondary comparisons ("pool" = margin >= 1, validation + test merged).

* main (PREREG section 4): O-keep vs S0, two same-seed pairs, k = 4, the pool: it passes when the mean of the two
  differences (O-keep - S0, points, unrounded) is at least 5.0 and both differences are above 0; besides: each pair's
  difference, wins / losses and the two-sided sign test, k = 2, the test set alone (section 3: the RL checkpoints were
  picked on the validation set), every margin band;
* secondary (section 5): every RL checkpoint vs its start, k = 2 and 4, the pool and the test set alone, no threshold;
* success per model, k, set and band (validation, test, merged); the share of unreadable turns per k and band;
* original vs perturbed: the same tasks' success from the original start (the models' earlier results: the validation
  folder for the val sets, the test folder for the test sets), and the drop, per k and band;
* with --turns (a reward.turn_metrics output over the perturbed results, labels MODEL@k<k>): per model, k and band the
  share of episodes with an error, the per-turn error rate, recovery within two turns, the share of episodes that got
  back to the original start's level (a remaining number <= d* at some point), success with and without an error;
  and the replay check (section 6: the replays must agree with the records): episodes whose replay disagreed with the
  record or replayed another number of turns than it recorded;
* with --fail-end (a reward.fail_end_steps output over the perturbed results): where failed episodes end, per band,
  with its coverage (every result with failed episodes present, as many failed episodes as the result has).

    python -m reward.compare_perturbed --set val_k2=DIR --set test_k2=DIR --set val_k4=DIR --set test_k4=DIR \
        --original MODEL=VALDIR,TESTDIR [...] [--turns FILE] [--fail-end FILE] --out OUT.json

A model whose result folder is missing, unfinished (no done.json) or has a task missing or an error row is incomplete:
every comparison it is in reads "missing" and the main verdict is "no verdict" (PREREG section 7).
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LIMIT = 12
BANDS = (("0", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5-", 5, 10 ** 9))
BAND_NAMES = [b[0] for b in BANDS]
MAIN_PAIRS = (("seed 20261019", "rw_ok_250", "rw_s0_150"), ("seed 20261021", "rw_s2ok_250", "rw_s2s0_300"))
MAIN_K = 4
MAIN_MIN_MEAN = 5.0  # points
SECONDARY_PAIRS = (("rw_s0_150", "sft_q3formal_e8"), ("rw_s2s0_300", "sft_q3formal_e8"),
                   ("rw_ok_250", "sft_q3formal_e8"), ("rw_s2ok_250", "sft_q3formal_e8"),
                   ("rw_g_250", "sft_q3formal_e8"), ("rw_o_200", "sft_q3formal_e8"), ("m3ds1d_300", "sft_m3d_e9"))
MODELS = ("rw_s0_150", "rw_s2s0_300", "rw_ok_250", "rw_s2ok_250", "rw_g_250", "rw_o_200",
          "sft_q3formal_e8", "sft_m3d_e9", "m3ds1d_300")
SET_NAME = re.compile(r"^(val|test)_k(\d+)$")


def split_of(label: str) -> str:
    return SET_NAME.match(label).group(1)


def k_of(label: str) -> int:
    return int(SET_NAME.match(label).group(2))


def band(margin: int) -> str:
    for name, low, high in BANDS:
        if low <= margin <= high:
            return name
    raise ValueError(f"margin {margin} outside every band")


def sign_test(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact binomial (p = 1/2) on the tasks one side solved and the other did not."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt_diff(c: Optional[Dict[str, Any]]) -> str:
    """A comparison for the tables: the difference (from the unrounded value), wins / losses and p."""
    if c is None:
        return "—"
    if "missing" in c:
        return "缺"
    if c.get("diff_raw") is None:
        return "—"
    return f"{c['diff_raw']:+.1f}（{c['wins']}/{c['losses']}，p {fmt_p(c['p_sign'])}）"


def load_tasks(folder: Path, label: str) -> Dict[str, Dict[str, Any]]:
    """The set's tasks by id; the folder must hold what its label says (split, k), each task once."""
    out = {}
    for line in (folder / "tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = r["perturbation"]
        if int(p["k"]) != k_of(label):
            raise ValueError(f"{label} ({folder}): task {r['task_id']} has k {p['k']}")
        if r.get("split") is not None and r["split"] != split_of(label):
            raise ValueError(f"{label} ({folder}): task {r['task_id']} is from split {r['split']}")
        if r["task_id"] in out:
            raise ValueError(f"{label} ({folder}): task {r['task_id']} twice")
        d_new = int(r["difficulty"]["d_star"])
        if d_new != int(p["d_star_original"]) + int(p["k"]):
            raise ValueError(f"{r['task_id']}: d_star {d_new} is not d* {p['d_star_original']} + k {p['k']}")
        out[r["task_id"]] = {"k": int(p["k"]), "d_orig": int(p["d_star_original"]), "d_new": d_new,
                             "margin": LIMIT - d_new, "family": r.get("task_type")}
    return out


def load_result(folder: Path, tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """{"rows": {task_id: row}, "complete": bool, "problems": [...]} for one model on one set."""
    problems = []
    if not (folder / "done.json").is_file():
        return {"rows": {}, "complete": False, "problems": ["no done.json"]}
    rows, errors = {}, 0
    for line in (folder / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        if r["task_id"] in rows:
            problems.append(f"task {r['task_id']} twice")
        rows[r["task_id"]] = r
    missing = sorted(set(tasks) - set(rows))
    extra = sorted(set(rows) - set(tasks))
    if errors:
        problems.append(f"{errors} error rows")
    if missing:
        problems.append(f"{len(missing)} tasks missing")
    if extra:
        problems.append(f"{len(extra)} tasks not in the set")
    return {"rows": rows, "complete": not problems, "problems": problems}


def rate(a: int, b: int) -> Optional[float]:
    """Unrounded (the tables round once, for display; rounding here too flipped x.x5 values)."""
    return a / b if b else None


def success(rows: Dict[Any, Any], ids: Sequence[Any]) -> Dict[str, Any]:
    got = [bool(rows[t]["success"]) for t in ids if t in rows]
    return {"n": len(got), "success": sum(got), "rate": rate(sum(got), len(got))}


def unreadable(rows: Dict[Any, Any], ids: Sequence[Any]) -> Optional[float]:
    pairs = [(int(rows[t].get("invalid_turns") or 0), int(rows[t].get("turns") or 0)) for t in ids if t in rows]
    return rate(sum(a for a, _ in pairs), sum(b for _, b in pairs))


def paired(a: Dict[Any, Any], b: Dict[Any, Any], ids: Sequence[Any]) -> Dict[str, Any]:
    both = [t for t in ids if t in a and t in b]
    sa = [bool(a[t]["success"]) for t in both]
    sb = [bool(b[t]["success"]) for t in both]
    wins = sum(x and not y for x, y in zip(sa, sb))
    losses = sum(y and not x for x, y in zip(sa, sb))
    n = len(both)
    raw = 100 * (sum(sa) - sum(sb)) / n if n else None
    return {"n": n, "a": rate(sum(sa), n), "b": rate(sum(sb), n), "diff_raw": raw,
            "diff_points": None if raw is None else round(raw, 2), "wins": wins, "losses": losses,
            "p_sign": sign_test(wins, losses)}


class Data:
    def __init__(self, sets: Dict[str, Path], models: Sequence[str]):
        for s in sets:
            if not SET_NAME.match(s):
                raise SystemExit(f"set name {s}: not val_k<k> / test_k<k>")
        self.sets, self.models = sets, list(models)
        self.ks = sorted({k_of(s) for s in sets})
        for k in self.ks:
            have = sorted(split_of(s) for s in sets if k_of(s) == k)
            if have != ["test", "val"]:
                raise SystemExit(f"k = {k}: sets {have}, need one val and one test set")
        self.tasks = {s: load_tasks(p, s) for s, p in sets.items()}
        self.res = {(m, s): load_result(p / m, self.tasks[s]) for m in self.models for s, p in sets.items()}

    def sets_of(self, k: int, which: str = "both") -> List[str]:
        return [s for s in self.sets if k_of(s) == k and (which == "both" or split_of(s) == which)]

    def complete(self, model: str, k: int) -> bool:
        return all(self.res[(model, s)]["complete"] for s in self.sets_of(k))

    def ids(self, k: int, which: str = "both", bands: Optional[Sequence[str]] = None,
            pool: bool = False) -> List[Tuple[str, str]]:
        """(set, task_id) keys of the k's sets; pool: margin >= 1."""
        out = []
        for s in self.sets_of(k, which):
            for t, info in self.tasks[s].items():
                if pool and info["margin"] < 1:
                    continue
                if bands is not None and band(info["margin"]) not in bands:
                    continue
                out.append((s, t))
        return out

    def rows(self, model: str, k: int) -> Dict[Tuple[str, str], Any]:
        return {(s, t): r for s in self.sets_of(k) for t, r in self.res[(model, s)]["rows"].items()}


def compare(data: Data, a: str, b: str, k: int, which: str = "both", bands: Optional[Sequence[str]] = None,
            pool: bool = True) -> Dict[str, Any]:
    if not (data.complete(a, k) and data.complete(b, k)):
        return {"missing": [m for m in (a, b) if not data.complete(m, k)]}
    return paired(data.rows(a, k), data.rows(b, k), data.ids(k, which, bands, pool))


def main_verdict(data: Data) -> Dict[str, Any]:
    pairs = {}
    for seed, a, b in MAIN_PAIRS:
        pairs[seed] = {"o_keep": a, "s0": b, **compare(data, a, b, MAIN_K)}
    if any("missing" in p or not p["n"] for p in pairs.values()):
        verdict, mean = "no verdict (a model's results are incomplete)", None
    else:
        diffs = [p["diff_raw"] for p in pairs.values()]
        raw_mean = statistics.mean(diffs)
        mean = raw_mean
        verdict = "pass" if raw_mean >= MAIN_MIN_MEAN and all(d > 0 for d in diffs) else "no pass"
    extra = {}
    for seed, a, b in MAIN_PAIRS:
        p = pairs[seed]
        extra[seed] = {
            "each_at_least_5": None if "missing" in p or not p["n"] else p["diff_raw"] >= MAIN_MIN_MEAN,
            "k2_pool": compare(data, a, b, 2) if 2 in data.ks else None,
            "k2_test_pool": compare(data, a, b, 2, "test") if 2 in data.ks else None,
            "k4_test_pool": compare(data, a, b, MAIN_K, "test"),
            "k4_by_band": {name: compare(data, a, b, MAIN_K, bands=[name], pool=False) for name in BAND_NAMES},
        }
    return {"rule": f"k = {MAIN_K}, margin >= 1, validation + test: mean of the two differences >= {MAIN_MIN_MEAN} "
                    "points and both above 0", "pairs": pairs, "mean_diff_points": mean, "verdict": verdict,
            "reported_besides": extra}


def secondary(data: Data) -> List[Dict[str, Any]]:
    out = []
    for a, b in SECONDARY_PAIRS:
        row = {"rl": a, "start": b}
        for k in data.ks:
            row[f"k{k}"] = compare(data, a, b, k)
            row[f"k{k}_test"] = compare(data, a, b, k, "test")
        out.append(row)
    return out


def success_table(data: Data) -> Dict[str, Any]:
    out = {}
    for m in data.models:
        per = {}
        for k in data.ks:
            rows = data.rows(m, k)
            per[f"k{k}"] = {
                "complete": data.complete(m, k),
                "problems": {s: data.res[(m, s)]["problems"] for s in data.sets_of(k) if data.res[(m, s)]["problems"]},
                "val": success(rows, data.ids(k, "val")), "test": success(rows, data.ids(k, "test")),
                "all": success(rows, data.ids(k)), "pool": success(rows, data.ids(k, pool=True)),
                "by_band": {name: success(rows, data.ids(k, bands=[name])) for name in BAND_NAMES},
                "by_band_val": {name: success(rows, data.ids(k, "val", bands=[name])) for name in BAND_NAMES},
                "by_band_test": {name: success(rows, data.ids(k, "test", bands=[name])) for name in BAND_NAMES},
                "unreadable_turn_share": unreadable(rows, data.ids(k)),
                "unreadable_by_band": {name: unreadable(rows, data.ids(k, bands=[name])) for name in BAND_NAMES},
            }
        out[m] = per
    return out


def original_vs_perturbed(data: Data, originals: Dict[str, List[Path]]) -> Dict[str, Any]:
    """The same tasks from the original start: the first folder is the model's validation result (for the val sets),
    the second its test result (for the test sets); a task is matched by its split and id."""
    out = {}
    for m, folders in originals.items():
        if not 1 <= len(folders) <= 2:
            raise SystemExit(f"--original {m}: give VALDIR[,TESTDIR]")
        orig: Dict[str, Dict[str, bool]] = {"val": {}, "test": {}}
        complete = {}
        for split, f in zip(("val", "test"), folders):
            complete[split] = (f / "done.json").is_file()
            for line in (f / "results.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        orig[split][r["task_id"]] = bool(r["success"])
        per: Dict[str, Any] = {"original_complete": complete}
        for k in data.ks:
            rows = data.rows(m, k)

            def stats(keys: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
                common = [(s, t) for (s, t) in keys if t in orig[split_of(s)] and (s, t) in rows]
                n = len(common)
                o = sum(orig[split_of(s)][t] for s, t in common)
                p = sum(bool(rows[key]["success"]) for key in common)
                return {"n": n, "original": rate(o, n), "perturbed": rate(p, n),
                        "drop_points": 100 * (o - p) / n if n else None,
                        "tasks_without_original": len(keys) - n}

            per[f"k{k}"] = {**stats(data.ids(k)),
                            "by_band": {name: stats(data.ids(k, bands=[name])) for name in BAND_NAMES}}
        out[m] = per
    return out


def turn_side(data: Data, turns: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from reward.turn_bins import band_summary

    folder_to_set = {p.name: s for s, p in data.sets.items()}
    merged: Dict[Tuple[str, int], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    unknown = []
    for lab, eps in turns["episodes"].items():
        head, _, folder = lab.partition("|")
        model, _, kk = head.rpartition("@k")
        s = folder_to_set.get(folder)
        if s is None or model not in data.models or not kk.isdigit() or int(kk) != k_of(s):
            unknown.append(lab)
            continue
        merged[(model, int(kk))] += [(s, e) for e in eps]
    out: Dict[str, Any] = {}
    for (model, k), items in sorted(merged.items()):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        check = {"replayed": 0, "replay_same": 0, "turns_match": 0, "not_replayed": 0, "mismatches": []}
        for s, e in items:
            if "d_star" not in e:
                check["not_replayed"] += 1
                continue
            row = data.res[(model, s)]["rows"].get(e["task_id"])
            recorded = None if row is None else (len(row["trajectory"]) if "trajectory" in row else row.get("turns"))
            same_len = recorded is not None and len(e["d"]) - 1 == int(recorded)
            check["replayed"] += 1
            check["replay_same"] += bool(e.get("replay_same"))
            check["turns_match"] += same_len
            if not (same_len and e.get("replay_same")) and len(check["mismatches"]) < 10:
                check["mismatches"].append(f"{s}/{e['task_id']}")
            mg = LIMIT - int(e["d_star"])
            groups[band(mg)].append(e)
            if mg >= 1:
                groups["pool"].append(e)
            groups["all"].append(e)
        res = {}
        for name in BAND_NAMES + ["pool", "all"]:
            s = band_summary(groups[name])
            back = [any(x is not None and x <= int(e["d_star"]) - k for x in e["d"]) for e in groups[name] if "d" in e]
            s["back_to_original_level"] = rate(sum(back), len(back))
            res[name] = s
        res["replay_check"] = check
        out.setdefault(model, {})[f"k{k}"] = res
    coverage = {"unknown_labels": unknown,
                "missing": [f"{m}@k{k}" for m in data.models for k in data.ks if (m, k) not in merged],
                "episodes_expected": sum(len(data.res[(m, s)]["rows"]) for m in data.models for s in data.sets),
                "episodes_found": sum(len(v) for v in merged.values())}
    return out, coverage


def fail_end_side(data: Data, fail_end: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    key_of = {(p.name, m): (m, s) for s, p in data.sets.items() for m in data.models}
    acc: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    found: Dict[Tuple[str, str], int] = defaultdict(int)
    unknown = []
    for path, eps in fail_end["episodes"].items():
        pp = Path(path)
        key = (pp.parent.name, pp.name)
        if key not in key_of:
            unknown.append(path)
            continue
        m, s = key_of[key]
        k = k_of(s)
        for e in eps:
            found[(m, s)] += 1
            if "error" in e:
                acc[(m, k, "errors")].append(e)
                continue
            mg = LIMIT - int(e["d_star"])
            for name in (band(mg), "pool" if mg >= 1 else None, "all"):
                if name:
                    acc[(m, k, name)].append(e)
    out: Dict[str, Any] = {}
    for m in data.models:
        for k in data.ks:
            per = {}
            for name in BAND_NAMES + ["pool", "all"]:
                eps = acc.get((m, k, name), [])
                ds = [e["d_end"] for e in eps if e["kind"] in ("d", "near")]
                per[name] = {"failed": len(eps), "known": len(ds),
                             "d_end_mean": statistics.mean(ds) if ds else None,
                             "exact": sum(bool(e.get("exact")) for e in eps if e["kind"] == "d"),
                             "d_end_le2_exact": sum(1 for e in eps if e["kind"] == "d" and e.get("exact")
                                                    and e["d_end"] <= 2),
                             "d_end_median": statistics.median(ds) if ds else None,
                             "d_end_le2_share": rate(sum(d <= 2 for d in ds), len(ds)),
                             "unreached": sum(e["kind"] == "unreached" for e in eps),
                             "missing": sum(e["kind"] == "missing" for e in eps),
                             "replay_same": sum(bool(e["replay_same"]) for e in eps)}
            per["errors"] = len(acc.get((m, k, "errors"), []))
            out.setdefault(m, {})[f"k{k}"] = per
    count_mismatch = []
    for m in data.models:
        for s in data.sets:
            failed = sum(not r["success"] for r in data.res[(m, s)]["rows"].values())
            if failed != found.get((m, s), 0):
                count_mismatch.append(f"{s}/{m}: {found.get((m, s), 0)} failed episodes replayed, {failed} in the result")
    coverage = {"unknown_paths": unknown, "count_mismatch": count_mismatch, "no_table": fail_end.get("no_table") or {}}
    return out, coverage


def markdown(res: Dict[str, Any]) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
    mv = res["main"]
    L = ["## 主比较（退 4 步，余量 ≥ 1，val + test）", "",
         "| 种子 | O-keep | S0 | 题数 | O-keep 成功率 | S0 成功率 | 差（胜/负，p） |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for seed, p in mv["pairs"].items():
        if "missing" in p:
            L.append(f"| {seed} | {p['o_keep']} | {p['s0']} | 缺 | | | |")
        else:
            L.append(f"| {seed} | {p['o_keep']} | {p['s0']} | {p['n']} | {pct(p['a'])} | {pct(p['b'])} | {fmt_diff(p)} |")
    mean_txt = "—" if mv["mean_diff_points"] is None else f"{mv['mean_diff_points']:+.2f}"
    L += ["", f"两对差的平均：{mean_txt}；判定：**{mv['verdict']}**（{mv['rule']}）", "",
          "| 种子 | 各自 ≥ 5 | 退 2 步（余量 ≥ 1） | 退 2 步只看 test | 退 4 步只看 test | 余量 0 | 1–2 | 3–4 | 5+ |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for seed, e in mv["reported_besides"].items():
        b = e["k4_by_band"]
        L.append(f"| {seed} | {e['each_at_least_5']} | {fmt_diff(e['k2_pool'])} | {fmt_diff(e['k2_test_pool'])} | "
                 f"{fmt_diff(e['k4_test_pool'])} | " + " | ".join(fmt_diff(b[n]) for n in BAND_NAMES) + " |")
    L += ["", "## 次比较：RL 对起点（余量 ≥ 1）", "",
          "| RL | 起点 | 退 2 步 | 退 2 步只看 test | 退 4 步 | 退 4 步只看 test |", "| --- | --- | --- | --- | --- | --- |"]
    for c in res["secondary"]:
        L.append(f"| {c['rl']} | {c['start']} | {fmt_diff(c.get('k2'))} | {fmt_diff(c.get('k2_test'))} | "
                 f"{fmt_diff(c.get('k4'))} | {fmt_diff(c.get('k4_test'))} |")
    L += ["", "## 成功率（%）", "",
          "| 模型 | 档 | val | test | 合并 | 余量 ≥ 1 | 余量 0 | 1–2 | 3–4 | 5+ | 读不出（回合） | 原起点（同题） | 退步后（同题） | 掉 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, per in res["success"].items():
        for kk, s in per.items():
            o = (res.get("original") or {}).get(m, {}).get(kk, {})
            L.append(f"| {m} | {kk} | {pct(s['val']['rate'])} | {pct(s['test']['rate'])} | {pct(s['all']['rate'])} | "
                     f"{pct(s['pool']['rate'])} | " + " | ".join(pct(s['by_band'][n]['rate']) for n in BAND_NAMES) +
                     f" | {pct(s['unreadable_turn_share'])} | {pct(o.get('original'))} | {pct(o.get('perturbed'))} | "
                     f"{'—' if o.get('drop_points') is None else format(o['drop_points'], '.1f')} |")
    if res.get("original"):
        L += ["", "## 原起点对退步后，按余量分档（同题，val + test；原起点 % / 退步后 % / 题数）", "",
              "| 模型 | 档 | 余量 0 | 1–2 | 3–4 | 5+ |", "| --- | --- | --- | --- | --- | --- |"]
        for m, per in res["original"].items():
            for kk in [f"k{k}" for k in sorted({int(x[1:]) for x in per if x.startswith('k')})]:
                bb = per[kk]["by_band"]
                L.append(f"| {m} | {kk} | " + " | ".join(
                    f"{pct(bb[n]['original'])} / {pct(bb[n]['perturbed'])} / {bb[n]['n']}" if bb[n]["n"] else "—"
                    for n in BAND_NAMES) + " |")
    if res.get("turns"):
        L += ["", "## 失误和找回（%）", "",
              "| 模型 | 档 | 余量 | 局数 | 成功率 | 有失误的局 | 逐回合失误率 | 找回率 | 回到原水平 | 有失误的成功率 | 无失误的成功率 | 查不到的回合 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, per in res["turns"].items():
            for kk, bands in per.items():
                for name in BAND_NAMES + ["pool", "all"]:
                    s = bands[name]
                    if not s["replayed"]:
                        continue
                    L.append(f"| {m} | {kk} | {name} | {s['replayed']} | {pct(s['success_rate'])} | "
                             f"{pct(s['episode_error_share'])} | {pct(s['error_rate'])} | {pct(s['recovery_rate'])} | "
                             f"{pct(s['back_to_original_level'])} | {pct(s['success_with_error'])} | "
                             f"{pct(s['success_without_error'])} | {s['unknown_d_turns']} |")
        checks = [b["replay_check"] for per in res["turns"].values() for b in per.values()]
        cov = res.get("turns_coverage") or {}
        L += ["", f"重放核对：重放 {sum(c['replayed'] for c in checks)} 局（应有 {cov.get('episodes_expected')}），"
                  f"和记录一致 {sum(c['replay_same'] for c in checks)} 局，回合数和记录相同 "
                  f"{sum(c['turns_match'] for c in checks)} 局，没重放 {sum(c['not_replayed'] for c in checks)} 局；"
                  f"缺的结果 {cov.get('missing')}，认不出的标签 {cov.get('unknown_labels')}"]
    if res.get("fail_end"):
        L += ["", "## 失败终点剩余步数（表内的值；精确的只是一部分，其余是上界）", "",
              "| 模型 | 档 | 余量 | 失败局 | 查得到 | 其中精确 | 均值 | 中位 | ≤ 2 的比例 % | 确定 ≤ 2 的局 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, per in res["fail_end"].items():
            for kk, bands in per.items():
                for name in BAND_NAMES + ["pool", "all"]:
                    s = bands[name]
                    if s["failed"]:
                        mean = "—" if s["d_end_mean"] is None else f"{s['d_end_mean']:.2f}"
                        L.append(f"| {m} | {kk} | {name} | {s['failed']} | {s['known']} | {s['exact']} | {mean} | "
                                 f"{s['d_end_median']} | {pct(s['d_end_le2_share'])} | {s['d_end_le2_exact']} |")
        cov = res.get("fail_end_coverage") or {}
        L += ["", f"覆盖核对：认不出的结果 {cov.get('unknown_paths')}，失败局数对不上 {cov.get('count_mismatch')}，"
                  f"没有表的题 {cov.get('no_table')}"]
    return "\n".join(L) + "\n"


def run(sets: Dict[str, Path], originals: Dict[str, List[Path]], turns: Optional[Dict[str, Any]],
        fail_end: Optional[Dict[str, Any]], models: Sequence[str] = MODELS) -> Dict[str, Any]:
    data = Data(sets, models)
    res = {"limit": LIMIT, "bands": BAND_NAMES, "sets": {s: str(p) for s, p in sets.items()},
           "main": main_verdict(data), "secondary": secondary(data),
           "success": success_table(data), "original": original_vs_perturbed(data, originals)}
    if turns is not None:
        res["turns"], res["turns_coverage"] = turn_side(data, turns)
    if fail_end is not None:
        res["fail_end"], res["fail_end_coverage"] = fail_end_side(data, fail_end)
    return res


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="{val|test}_k<k>=FOLDER")
    ap.add_argument("--original", action="append", default=[], help="MODEL=VALDIR,TESTDIR (earlier results)")
    ap.add_argument("--turns", default=None)
    ap.add_argument("--fail-end", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sets = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in a.set}
    originals = {o.split("=", 1)[0]: [Path(x) for x in o.split("=", 1)[1].split(",")] for o in a.original}
    load = lambda p: json.loads(Path(p).read_text()) if p else None  # noqa: E731
    res = run(sets, originals, load(a.turns), load(a.fail_end))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    md = markdown(res)
    Path(a.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
