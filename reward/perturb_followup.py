"""Reward line, the perturbed-start evaluation (docs/reward/reports/PERTURB_EVAL_20261106.md sections 0, 3 and 5): the
follow-up numbers the report review asked for, on this evaluation's fixed paths (reward_runs/eval_sets/pert_*, the
analysis/20_perturb and analysis/19_mix/turn_metrics outputs; stdlib only, one process, about a minute):

    nice -n 19 python3 reward/perturb_followup.py > reward_runs/analysis/20_perturb/followup.txt

(a) failed episodes: did they ever get back to the original start's level (remaining <= d*), and at k = 2, how many
    spare actions did they have then (12 - actions used - remaining), did they err again, end beyond that level;
(b) RL - start success by margin group: original start (margin 12 - d*) vs k = 2 / 4 (margin 12 - (d* + k));
(c) recovery within two turns: O-keep vs S0 (paired bootstrap over tasks), and each model perturbed vs original start
    on the same tasks (paired bootstrap);
(d) failure ends: how many in-table values are exact; certain <= 2 / certain >= 3 (lower bound min(d, 13 - depth));
(e) errors that leave the table: a turn from a known remaining number to an unknown one.
"""
import json
import random
import sys
from collections import defaultdict

U = "/mnt/umm/users/yinbaiqiao"
R = U + "/reward_runs"
M3 = U + "/VAGEN-Lite-matrix/exps/matrix_v3"
MODELS = ["rw_s0_150", "rw_s2s0_300", "rw_ok_250", "rw_s2ok_250", "rw_g_250", "rw_o_200",
          "sft_q3formal_e8", "sft_m3d_e9", "m3ds1d_300"]
ORIG_LABEL = {"rw_s0_150": "rw_s0@150", "rw_s2s0_300": "rw_s2s0@300", "rw_ok_250": "rw_ok@250",
              "rw_s2ok_250": "rw_s2ok@250", "rw_g_250": "rw_g@250", "rw_o_200": "rw_o@200",
              "sft_q3formal_e8": "c0_sft@0", "sft_m3d_e9": "m3d_sft@0", "m3ds1d_300": "m3ds1d@300"}
ORIG_DIRS = {m: [R + "/eval_sets/eval_val_v2/" + m, R + "/eval_sets/eval_test_v2/" + m] for m in MODELS[:6]}
ORIG_DIRS["sft_q3formal_e8"] = [U + "/models_runs/eval_sets/eval_val_v2/sft_q3formal_e8",
                                U + "/models_runs/eval_sets/eval_test_v2/sft_q3formal_e8"]
for m in ("sft_m3d_e9", "m3ds1d_300"):
    ORIG_DIRS[m] = [M3 + "/eval_val_v2_screened/" + m, M3 + "/eval_test_v2_screened/" + m]
PAIRS_C0 = [(m, "sft_q3formal_e8") for m in MODELS[:6]]
OK_S0 = [("rw_ok_250", "rw_s0_150"), ("rw_s2ok_250", "rw_s2s0_300")]


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def pct(x):
    return "-" if x is None else "%.1f" % (100 * x)


# ---- load
tasks = {}  # (k, split, task_id) -> d_orig
for k in (2, 4):
    for sp in ("val", "test"):
        for r in jl("%s/eval_sets/pert_%s_k%d/tasks.jsonl" % (R, sp, k)):
            tasks[(k, sp, r["task_id"])] = int(r["perturbation"]["d_star_original"])
pert_rows = {}  # (m, k, split, tid) -> results row
for m in MODELS:
    for k in (2, 4):
        for sp in ("val", "test"):
            for r in jl("%s/eval_sets/pert_%s_k%d/%s/results.jsonl" % (R, sp, k, m)):
                pert_rows[(m, k, sp, r["task_id"])] = r
tm = json.load(open(R + "/analysis/20_perturb/turn_metrics.json"))["episodes"]
pert_eps = {}  # (m, k, split, tid) -> episode
for lab, eps in tm.items():
    head, _, folder = lab.partition("|")
    m, _, kk = head.rpartition("@k")
    sp = folder.split("_")[1]
    for e in eps:
        pert_eps[(m, int(kk), sp, e["task_id"])] = e
tm0 = json.load(open(R + "/analysis/19_mix/turn_metrics/turn_metrics_v2.json"))["episodes"]
orig_eps = {}  # (m, split, tid) -> episode
for m, lab0 in ORIG_LABEL.items():
    for lab, eps in tm0.items():
        if lab.split("|")[0] != lab0:
            continue
        sp = "test" if "_test_" in lab.split("|")[1] else "val"  # "eval_test_v2" also contains "val" (in "eval")
        for e in eps:
            orig_eps[(m, sp, e["task_id"])] = e
orig_rows = {}  # (m, split, tid) -> row
for m, (vd, td) in ORIG_DIRS.items():
    for sp, d in (("val", vd), ("test", td)):
        for r in jl(d + "/results.jsonl"):
            if "error" not in r:
                orig_rows[(m, sp, r["task_id"])] = r


def margin_pert(k, sp, tid):
    return 12 - (tasks[(k, sp, tid)] + k)


def pool_keys(m, k):
    return [key for key in pert_eps if key[0] == m and key[1] == k and margin_pert(k, key[2], key[3]) >= 1]


# ---- (a) back to the original level
print("== (a) failed episodes and the original level (margin >= 1)")
for k in (2, 4):
    for m in MODELS:
        keys = pool_keys(m, k)
        n = len(keys)
        succ = [key for key in keys if pert_eps[key]["success"]]
        fail = [key for key in keys if not pert_eps[key]["success"]]
        back_fail = []
        for key in fail:
            e = pert_eps[key]
            d0 = tasks[(k, key[2], key[3])]
            if any(x is not None and x <= d0 for x in e["d"]):
                back_fail.append(key)
        line = "k%d %-16s n %d success %s | failed %d: got back %s, never %s | of all: never %s, back+failed %s" % (
            k, m, n, pct(len(succ) / n), len(fail), pct(len(back_fail) / len(fail)),
            pct(1 - len(back_fail) / len(fail)), pct((len(fail) - len(back_fail)) / n), pct(len(back_fail) / n))
        spare, erred, beyond = [], 0, 0
        for key in back_fail:
            e = pert_eps[key]
            d0 = tasks[(k, key[2], key[3])]
            t = next(i for i, x in enumerate(e["d"]) if x is not None and x <= d0)
            traj = pert_rows[key]["trajectory"]
            used = sum(len(traj[i].get("actions") or []) for i in range(t))
            spare.append((12 - used) - e["d"][t])
            d = e["d"]
            if any(d[j] is not None and d[j - 1] is not None and d[j] > d[j - 1] for j in range(t + 1, len(d))):
                erred += 1
            if d[-1] is None or d[-1] > d0:
                beyond += 1
        if back_fail:
            line += " | at return: spare mean %.2f, >=0 %s, >=2 %s; erred again %s; ended beyond level %s" % (
                sum(spare) / len(spare), pct(sum(s >= 0 for s in spare) / len(spare)),
                pct(sum(s >= 2 for s in spare) / len(spare)), pct(erred / len(back_fail)), pct(beyond / len(back_fail)))
        print(line)

# ---- (b) RL - start by margin group
print("== (b) RL - start success (points) by margin group; mean over the six C0 pairs, then m3ds1d - D e9")
GROUPS = [("0-1", 0, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6-7", 6, 7), ("8+", 8, 99)]


def by_margin(m, setting):
    out = defaultdict(list)
    if setting == "orig":
        for (mm, sp, tid), r in orig_rows.items():
            if mm == m:
                out[12 - int(r["d_star"])].append(bool(r["success"]))
    else:
        k = setting
        for (mm, kk, sp, tid), r in pert_rows.items():
            if mm == m and kk == k:
                out[margin_pert(k, sp, tid)].append(bool(r["success"]))
    return out


for label, pairs in (("C0 pairs", PAIRS_C0), ("m3ds1d - D e9", [("m3ds1d_300", "sft_m3d_e9")])):
    for setting in ("orig", 2, 4):
        cells = []
        for g, lo, hi in GROUPS:
            diffs, ns, base = [], [], []
            for a, b in pairs:
                A, B = by_margin(a, setting), by_margin(b, setting)
                sa = [x for mg in range(lo, hi + 1) for x in A.get(mg, [])]
                sb = [x for mg in range(lo, hi + 1) for x in B.get(mg, [])]
                if sa and sb:
                    diffs.append(100 * (sum(sa) / len(sa) - sum(sb) / len(sb)))
                    ns.append(len(sa))
                    base.append(100 * sum(sb) / len(sb))
            cells.append("%s: %s (n %s, start %s%%)" % (g, "%+.1f" % (sum(diffs) / len(diffs)) if diffs else "-",
                                                      ns[0] if ns else 0,
                                                      "%.1f" % (sum(base) / len(base)) if base else "-"))
        print("%-14s %-4s %s" % (label, setting, " | ".join(cells)))

# ---- (c) recovery
print("== (c) recovery within two turns, paired bootstrap over tasks (2000 draws, seed 20261106)")


def rec_counts(e):
    d = e["d"]
    rec = able = 0
    for t in range(1, len(d)):
        if d[t] is None or d[t - 1] is None or d[t] <= d[t - 1]:
            continue
        later = [x for x in d[t + 1:t + 3] if x is not None]
        back = any(x <= d[t - 1] for x in later)
        if len(d) - 1 - t >= 2 or back:
            able += 1
            rec += back
    return rec, able


def boot(pairs_of_counts, reps=2000, seed=20261106):
    """pairs_of_counts: [((rec_a, able_a), (rec_b, able_b)) per task] -> (diff points, lo, hi)."""
    rng = random.Random(seed)

    def diff(sample):
        ra = sum(x[0][0] for x in sample); aa = sum(x[0][1] for x in sample)
        rb = sum(x[1][0] for x in sample); ab = sum(x[1][1] for x in sample)
        return 100 * (ra / aa - rb / ab) if aa and ab else None

    point = diff(pairs_of_counts)
    vals = []
    n = len(pairs_of_counts)
    for _ in range(reps):
        v = diff([pairs_of_counts[rng.randrange(n)] for _ in range(n)])
        if v is not None:
            vals.append(v)
    vals.sort()
    return point, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


for a, b in OK_S0:
    for setting in ("orig", 2, 4):
        if setting == "orig":
            keys = [(sp, tid) for (mm, sp, tid) in orig_eps if mm == a and (b, sp, tid) in orig_eps]
            pc = [(rec_counts(orig_eps[(a, sp, tid)]), rec_counts(orig_eps[(b, sp, tid)])) for sp, tid in keys]
        else:
            keys = [(key[2], key[3]) for key in pool_keys(a, setting)]
            pc = [(rec_counts(pert_eps[(a, setting, sp, tid)]), rec_counts(pert_eps[(b, setting, sp, tid)]))
                  for sp, tid in keys]
        p, lo, hi = boot(pc)
        print("%s - %s %-4s %+.1f [%+.1f, %+.1f]  (tasks %d)" % (a, b, setting, p, lo, hi, len(pc)))
for m in MODELS:
    for k in (2, 4):
        keys = [(key[2], key[3]) for key in pool_keys(m, k) if (m, key[2], key[3]) in orig_eps]
        pc = [(rec_counts(pert_eps[(m, k, sp, tid)]), rec_counts(orig_eps[(m, sp, tid)])) for sp, tid in keys]
        p, lo, hi = boot(pc)
        print("%-16s k%d perturbed - original (same tasks) %+.1f [%+.1f, %+.1f]  (tasks %d)" % (m, k, p, lo, hi,
                                                                                          len(pc)))

# ---- (d) failure ends
print("== (d) failure ends (margin >= 1): in-table exact share; certain <= 2; certain >= 3 (of failed)")
fe = json.load(open(R + "/analysis/20_perturb/fail_end.json"))["episodes"]
for k in (2, 4):
    for m in MODELS:
        eps = []
        for sp in ("val", "test"):
            for e in fe.get("%s/eval_sets/pert_%s_k%d/%s" % (R, sp, k, m), []):
                if 12 - int(e["d_star"]) >= 1:
                    eps.append(e)
        ind = [e for e in eps if e["kind"] == "d"]
        exact = [e for e in ind if e["exact"]]
        le2 = sum(1 for e in exact if e["d_end"] <= 2)
        ge3 = sum(1 for e in ind if (e["d_end"] if e["exact"] else min(e["d_end"], 13 - e["depth_end"])) >= 3)
        lb = [e["d_end"] if e["exact"] else min(e["d_end"], 13 - e["depth_end"]) for e in ind]
        print("k%d %-16s failed %d, in table %d, exact %s of in-table; certain <=2 %s, certain >=3 %s of failed; "
              "mean in-table %.2f, mean lower bound %.2f" % (k, m, len(eps), len(ind), pct(len(exact) / len(ind)),
                                                             pct(le2 / len(eps)), pct(ge3 / len(eps)),
                                                             sum(e["d_end"] for e in ind) / len(ind), sum(lb) / len(lb)))

# ---- (e) errors that leave the table
print("== (e) turns from a known remaining number to an unknown one (margin >= 1), vs counted errors")
for k in (2, 4):
    for m in MODELS:
        out_turns = errs = 0
        for key in pool_keys(m, k):
            d = pert_eps[key]["d"]
            for t in range(1, len(d)):
                if d[t - 1] is not None and d[t] is None:
                    out_turns += 1
                if d[t - 1] is not None and d[t] is not None and d[t] > d[t - 1]:
                    errs += 1
        print("k%d %-16s left the table %d, counted errors %d (%s)" % (k, m, out_turns, errs, pct(out_turns / errs)))


# ---- (e2) recovery if every step off the table were an unrecovered error
def rec_counts_exit(e):
    rec, able = rec_counts(e)
    d = e["d"]
    exits = sum(1 for t in range(1, len(d)) if d[t - 1] is not None and d[t] is None)
    return rec, able + exits


print("== (e2) recovery as counted vs with every step off the table an unrecovered error (margin >= 1)")
for k in (2, 4):
    for m in MODELS:
        keys = pool_keys(m, k)
        a = [rec_counts(pert_eps[key]) for key in keys]
        b = [rec_counts_exit(pert_eps[key]) for key in keys]
        r1 = sum(x[0] for x in a) / sum(x[1] for x in a)
        r2 = sum(x[0] for x in b) / sum(x[1] for x in b)
        print("k%d %-16s recovery %s -> %s (%+.1f)" % (k, m, pct(r1), pct(r2), 100 * (r2 - r1)))

# ---- (f) RL - start on the same pool tasks: original start vs perturbed, per pair
print("== (f) RL - start success on the same tasks (margin >= 1 of the perturbed set): original start vs perturbed")
for a, b in PAIRS_C0 + [("m3ds1d_300", "sft_m3d_e9")]:
    cells = []
    for k in (2, 4):
        keys = [key for key in pool_keys(a, k)
                if (a, key[2], key[3]) in orig_rows and (b, key[2], key[3]) in orig_rows]
        oa = sum(bool(orig_rows[(a, key[2], key[3])]["success"]) for key in keys) / len(keys)
        ob = sum(bool(orig_rows[(b, key[2], key[3])]["success"]) for key in keys) / len(keys)
        pa = sum(bool(pert_rows[(a, k, key[2], key[3])]["success"]) for key in keys) / len(keys)
        pb = sum(bool(pert_rows[(b, k, key[2], key[3])]["success"]) for key in keys) / len(keys)
        cells.append("k%d (n %d): original %+.1f -> perturbed %+.1f; ratio %.2f -> %.2f; drops RL %.1f start %.1f" % (
            k, len(keys), 100 * (oa - ob), 100 * (pa - pb), oa / ob, pa / pb, 100 * (oa - pa), 100 * (ob - pb)))
    print("%-12s vs %-16s %s" % (a, b, " | ".join(cells)))

# ---- (g) per-turn error rate: original start (all its tasks) vs perturbed (margin >= 1)
print("== (g) per-turn error rate and episodes with an error: original start vs perturbed")


def err_rate(eps):
    turns = errs = with_err = 0
    for e in eps:
        d = e["d"]
        had = False
        for t in range(1, len(d)):
            if d[t] is None or d[t - 1] is None:
                continue
            turns += 1
            if d[t] > d[t - 1]:
                errs += 1
                had = True
        with_err += had
    return errs / turns, with_err / len(eps), turns / len(eps)


for m in MODELS:
    o = err_rate([e for (mm, sp, tid), e in orig_eps.items() if mm == m and "d" in e])
    p2 = err_rate([pert_eps[key] for key in pool_keys(m, 2)])
    p4 = err_rate([pert_eps[key] for key in pool_keys(m, 4)])
    print("%-16s per-turn error %s / %s / %s; episodes with an error %s / %s / %s; turns per episode %.1f / %.1f / %.1f"
          % (m, pct(o[0]), pct(p2[0]), pct(p4[0]), pct(o[1]), pct(p2[1]), pct(p4[1]), o[2], p2[2], p4[2]))
sys.stdout.flush()
