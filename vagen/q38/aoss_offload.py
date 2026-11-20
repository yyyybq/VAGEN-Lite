#!/usr/bin/env python3
"""q38 (the Qwen3.8-27B rerun line): archive evaluated checkpoints to AOSS, then delete the local copy.

P1_REPORT 6 "定 5", accepted by the user on 2026-11-20 ("其他的按你的推荐来"): a standing authorization for the q38
line only -- a checkpoint whose evaluation is finished is uploaded to AOSS and then deleted; locally each run keeps only
the checkpoint the decision uses (--keep-rule last, the default: PREREG_Q38 4 reads GRPO at its last
checkpoint; "best" or "best_and_last", the train_chain rule, on request), plus whatever another plan or an
unfinished job still names.  The disk line's iron rule (RETENTION_POLICY R9) is
followed: upload first, check every object's size, manifest locally and on AOSS, re-check that nobody reads the files,
print the full list and total, delete, keep a per-file record and a note.

DRAFT.  Every phase is a dry run unless given its flag: nothing is written without `select --write`, nothing is
uploaded without `archive --execute`, nothing is deleted without `delete --yes`.

    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38
    PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python          # has boto3
    ENV="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
         OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false"
    $ENV nice -n 19 $PY -I -B vagen/q38/aoss_offload.py select PLAN.json            # what would go (dry)
    $ENV nice -n 19 $PY -I -B vagen/q38/aoss_offload.py select PLAN.json --write    # spec + plan.tsv
    $ENV nice -n 19 $PY -I -B vagen/q38/aoss_offload.py archive SPEC.json [--execute] [--files 2 --threads 2]
    $ENV nice -n 19 $PY -I -B vagen/q38/aoss_offload.py delete SPEC.json [--yes]
    $ENV nice -n 19 $PY -I -B vagen/q38/aoss_offload.py status SPEC.json

What `select` takes (PLAN.json is the train_chain plan of one chain; the chain's own views are used, train_chain.Chain):
  * RL checkpoints `<run_dir>/checkpoints/global_step_<s>` of a chain whose evaluate step is done, when every
    checkpoint of the run has a complete result (done.json) in every evaluation set, the folder has COMPLETE, and it is
    not kept by --keep-rule (last: the run's last; best: its validation best, select_by set, ties the earlier;
    best_and_last: both), not named by --keep;
  * with --resume-state archive, also the resume state of each run's last checkpoint (everything but
    actor/huggingface and COMPLETE, as prune_to_hf), once rl_extend and evaluate are done;
  * SFT epoch checkpoints of the picked group other than the picked one and the RL start model (pick_sft done).
Every item is then refused when (checked again before upload and before delete):
  outside q38/runs or inside q38/models / q38/envs; below a PROTECTED_PATHS.tsv row (disk line); named by a plan,
  another chain's plan (progress.json "plan", rl_model_override, picked SFT), an RL yaml or an unfinished job's
  launch.sh / prepared.json; its run has an unfinished sco job; an unfinished job of the workspace mentions it; a
  login-node process has a file below it open or as its cwd; a file was changed in the last hour; a file is not ours.

Records (the disk line's layout, so disk/restore_from_aoss.py BATCH restores them):
  /mnt/umm/users/yinbaiqiao/disk_runs/batches/<batch>/plan.tsv, manifest.jsonl, deleted.tsv (copies on AOSS
  vagen_lite_archive/disk_line_manifests/<batch>/); objects at fj:s3://baiqiao/vagen_lite_archive/<path below
  /mnt/umm/users/yinbaiqiao>; per run <run_dir>/aoss_offload/LEDGER.tsv and one note per deleted folder.  The note is
  kept outside checkpoints/: V1 removes step folders without actor/ on its next save (TB:995-1003), and a deleted
  folder is removed entirely so train_chain sees no half folder.
Login node (the user's rule of 2026-10-28): at most 4 cores per task, files x threads <= 4 (default 2 x 2, about
0.2 GB/s by the disk line's measurement); larger batches belong in a one-card sco job (8 x 8, about 1.4 GB/s).
Credentials are read from petreloss.conf section [fj] inside the process and never printed.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import fnmatch
import glob
import json
import os
import re
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # run as a file under python -I (the repository is not on sys.path then)
    sys.path.insert(0, str(REPO))

U = "/mnt/umm/users/yinbaiqiao"
BUCKET = "baiqiao"
PREFIX = "vagen_lite_archive"
MANIFEST_PREFIX = PREFIX + "/disk_line_manifests"
CONF = U + "/aoss/petreloss.conf"
APPROVAL = ("q38 standing authorization: P1_REPORT 定 5 (archive evaluated checkpoints to AOSS, then delete; keep "
            "locally only the ones decisions use; q38 line only), accepted by the user 2026-11-20: "
            "\"其他的按你的推荐来\"")
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED", "NOT_FOUND"}
PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
SKIP_DIRS = {"checkpoints", "rollout_data", "wandb", "model", "torch_extensions_renderer", "logs"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_jobs_default() -> Optional[List[dict]]:
    from data_gen.interiorgs_v10 import job_sco

    return job_sco.list_jobs()


def s3_client(files: int = 2, threads: int = 2):
    import boto3
    from botocore.config import Config

    cp = configparser.ConfigParser()
    cp.read(CONF)
    s = cp["fj"]
    return boto3.client("s3", endpoint_url=s["host_base"], aws_access_key_id=s["access_key"],
                        aws_secret_access_key=s["secret_key"],
                        config=Config(retries={"max_attempts": 8}, max_pool_connections=max(8, files * threads * 2),
                                      s3={"addressing_style": "path"}))


@dataclass
class Ctx:
    """Where things are and how the outside world is reached (the tests give their own)."""
    home: str = U
    scope: Tuple[str, ...] = (U + "/q38/runs",)
    never: Tuple[str, ...] = (U + "/q38/models", U + "/q38/envs")
    protected_tsv: str = U + "/VAGEN-Lite-disk/docs/disk/PROTECTED_PATHS.tsv"
    spec_dir: str = U + "/q38/offload/specs"
    records_dir: str = U + "/disk_runs/batches"
    ref_globs: Tuple[str, ...] = (U + "/q38/plans/*.json", U + "/q38/plans/*/*.json", U + "/q38/runs/*/plan*.json",
                                  U + "/q38/runs/*/rl*/*.yaml", U + "/q38/runs/*/*/rl*/*.yaml")
    progress_globs: Tuple[str, ...] = (U + "/q38/runs/*/progress.json", U + "/q38/runs/*/*/progress.json")
    proc_root: str = "/proc"
    min_age_s: float = 3600.0
    uid: int = field(default_factory=os.getuid)
    jobs_fn: Callable[[], Optional[List[dict]]] = list_jobs_default
    client_fn: Callable[..., Any] = s3_client
    clock: Callable[[], float] = time.time

    def key_of(self, path: str) -> str:
        assert path.startswith(self.home + "/"), path
        return "%s/%s" % (PREFIX, path[len(self.home) + 1:])


class Refused(SystemExit):
    pass


# ------------------------------------------------------------------------------------------------ references
def mentions(text: str, path: str) -> bool:
    """`path` appears in `text` as a whole path (global_step_5 is not a mention of global_step_50)."""
    return re.search(re.escape(path.rstrip("/")) + r"(?![0-9A-Za-z_.-])", text) is not None


def reference_sources(ctx: Ctx) -> Dict[str, str]:
    """{source: text} of everything that may name a checkpoint as a model to use: plan files, RL yamls, and of every
    q38 chain's progress.json its plan, rl_model_override and picked SFT model (not the rest: progress.json also lists
    the chain's own SFT epochs, which are not uses)."""
    out: Dict[str, str] = {}
    for pat in ctx.ref_globs:
        for f in sorted(glob.glob(pat)):
            try:
                if os.path.getsize(f) <= 20_000_000:
                    out[f] = Path(f).read_text(errors="ignore")
            except OSError:
                pass
    for pat in ctx.progress_globs:
        for f in sorted(glob.glob(pat)):
            try:
                st = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue
            pick = ((st.get("steps") or {}).get("pick_sft") or {}).get("decision") or {}
            out[f + "#plan"] = json.dumps(st.get("plan"), ensure_ascii=False) + "\n" + str(st.get("rl_model_override"))
            out[f + "#pick"] = str(pick.get("model"))
    return out


def referenced_by(path: str, sources: Dict[str, str]) -> List[str]:
    return [src for src, text in sources.items() if mentions(text, path)]


# ------------------------------------------------------------------------------------------------ selection
def _result(name: str, s: int) -> str:
    return f"{name}_{s:03d}" if name else f"v10_rl_{s:03d}"


KEEP_RULES = ("last", "best", "best_and_last")


def select(plan_path: str, ctx: Ctx, keep: Sequence[str] = (), resume_state: str = "keep",
           scorer: Optional[Callable[[Path], Optional[float]]] = None,
           keep_rule: str = "last") -> Tuple[List[dict], dict]:
    """(items, report) of one chain; nothing is written.  See the module docstring for the rule."""
    from data_gen.interiorgs_v10 import train_chain as tc

    if keep_rule not in KEEP_RULES:
        raise ValueError(f"keep_rule {keep_rule!r} not in {KEEP_RULES}")
    plan = json.loads(Path(plan_path).read_text())
    chain = tc.Chain(plan, sleep=lambda _s: None)
    steps = chain.state["steps"]
    scorer = scorer or tc.success_rate
    keep_set = {str(Path(k).resolve()) for k in keep}
    report: Dict[str, Any] = {"chain": str(chain.tr), "plan": str(plan_path), "runs": {}, "sft": {}, "notes": []}
    items: List[dict] = []
    if (steps.get("evaluate") or {}).get("state") != "done":
        report["notes"].append("the chain's evaluate step is not done: nothing selected")
        return items, report
    sources = reference_sources(ctx)
    sets = chain.eval_sets()
    select_by = plan.get("select_by") or next(iter(sets), None)
    every = int((plan.get("eval") or {}).get("every", 50))

    def add(kind: str, root: Path, run_dir: Path, why: str, **extra: Any) -> None:
        r = str(root.resolve())
        if r in keep_set:
            report["notes"].append(f"{r}: kept (--keep)")
            return
        refs = referenced_by(r, sources)
        if refs and kind != "resume_state":
            report["notes"].append(f"{r}: kept, named by {refs[:3]}")
            return
        items.append({"id": "i%03d" % (len(items) + 1), "kind": kind, "root": r, "run_dir": str(run_dir.resolve()),
                      "marker_dir": str((run_dir / "aoss_offload").resolve()), "why": why, **extra})

    runs = {} if plan.get("eval_only") or plan.get("sft_only") else chain.rl_runs()
    for name, run in runs.items():
        ck = run["run_dir"] / "checkpoints"
        found = sorted(int(p.name.split("_")[-1]) for p in ck.glob("global_step_*") if p.name.split("_")[-1].isdigit())
        rep: Dict[str, Any] = {"checkpoints": found}
        report["runs"][name or "(single)"] = rep
        if not found:
            continue
        if select_by is None or any(s % every for s in found):
            rep["skipped"] = "a checkpoint off the evaluation grid, or no evaluation set"
            continue
        vals = {s: scorer(sets[select_by] / _result(name, s)) for s in found}
        incomplete = [s for s in found if vals[s] is None or not all((d / _result(name, s) / "done.json").is_file()
                                                                    for d in sets.values())]
        if incomplete:
            rep["skipped"] = f"no complete result in every evaluation set for steps {incomplete}"
            continue
        best = max(vals, key=lambda s: (vals[s], -s))
        last = max(found)
        kept = {"last": {last}, "best": {best}, "best_and_last": {best, last}}[keep_rule]
        rep.update(best=best, last=last, kept=sorted(kept), keep_rule=keep_rule, val=vals)
        for s in found:
            folder = ck / f"global_step_{s}"
            if s in kept:
                continue
            if not (folder / "COMPLETE").exists():
                rep.setdefault("not_complete", []).append(s)
                continue
            add("rl_ckpt", folder, run["run_dir"], f"evaluated (val {vals[s]}), not kept ({keep_rule}: {sorted(kept)})",
                run=name, step=s)
        if resume_state == "archive":
            lf = ck / f"global_step_{last}"
            extra = [p for p in lf.iterdir() if p.name not in ("actor", "COMPLETE")]
            if (lf / "actor").is_dir():
                extra += [p for p in (lf / "actor").iterdir() if p.name != "huggingface"]
            if extra and (steps.get("rl_extend") or {}).get("state") == "done" and (lf / "COMPLETE").exists() \
                    and (lf / "actor" / "huggingface").is_dir():
                add("resume_state", lf, run["run_dir"], "resume state of the last checkpoint (the run is over)",
                    run=name, step=last, exclude=[r"^actor/huggingface/", r"^COMPLETE$"])
    pick_step = steps.get("pick_sft") or {}
    pick = (pick_step.get("decision") or {}).get("model")
    if pick_step.get("state") == "done" and pick and (steps.get("sft_eval") or {}).get("state") == "done":
        start = str(chain.sft_model().resolve())
        for group in chain.sft_groups():
            model = chain.lf_dir(group) / "model"
            if not model.is_dir() or Path(pick).parent.resolve() != model.resolve():
                continue  # as train_chain.retain: only the picked group's epochs
            report["sft"][group or "(single)"] = {"pick": pick}
            for epoch, path in tc.epoch_checkpoints(model):
                r = str(path.resolve())
                if r in (str(Path(pick).resolve()), start):
                    continue
                add("sft_epoch", path, chain.lf_dir(group), f"SFT epoch {epoch}, not the picked one", epoch=epoch)
    return items, report


# ------------------------------------------------------------------------------------------------ files and guards
def enumerate_item(item: dict) -> List[dict]:
    root = item["root"].rstrip("/")
    exc = [re.compile(r) for r in item.get("exclude", [])]
    rows = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode) or any(r.search(rel) for r in exc):
                continue  # symlinks are neither archived nor deleted
            rows.append({"item": item["id"], "path": p, "bytes": st.st_size, "mtime": st.st_mtime,
                         "uid": st.st_uid, "nlink": st.st_nlink})
    return rows


def protected_patterns(ctx: Ctx) -> List[str]:
    try:
        with open(ctx.protected_tsv) as f:
            f.readline()
            return [line.split("\t")[0].rstrip("/") for line in f if line.strip()]
    except OSError:
        return []


def under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def open_below(ctx: Ctx, roots: Sequence[str]) -> Dict[str, List[str]]:
    """{root: ["pid N (cwd|fd)"]} of login-node processes with a file below a root open (only what /proc shows us)."""
    hits: Dict[str, List[str]] = {}
    for pid in os.listdir(ctx.proc_root) if os.path.isdir(ctx.proc_root) else []:
        if not pid.isdigit():
            continue
        base = os.path.join(ctx.proc_root, pid)
        links = [("cwd", os.path.join(base, "cwd"))]
        try:
            links += [("fd", os.path.join(base, "fd", n)) for n in os.listdir(os.path.join(base, "fd"))]
        except OSError:
            pass
        for kind, link in links:
            try:
                target = os.readlink(link)
            except OSError:
                continue
            for r in roots:
                if under(target, r):
                    hits.setdefault(r, []).append(f"pid {pid} ({kind})")
    return hits


def job_dirs(top: Path) -> List[Path]:
    """Folders holding a submission.json below `top` (job_sco's layout), not descending into bulky folders."""
    out = []
    for dp, dns, fns in os.walk(top):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        if "submission.json" in fns:
            out.append(Path(dp))
    return out


def guards(items: Sequence[dict], rows: Sequence[dict], ctx: Ctx, chain_dir: str) -> List[Tuple[str, str]]:
    """[(item id, reason)] of every item that must not be archived or deleted now."""
    bad: List[Tuple[str, str]] = []
    pats = protected_patterns(ctx)
    sources = reference_sources(ctx)
    for it in items:
        r = it["root"]
        if not any(under(r, s) for s in ctx.scope) or any(under(r, n) for n in ctx.never):
            bad.append((it["id"], f"outside the q38 scope {ctx.scope}"))
        if any(fnmatch.fnmatchcase(r, p) or fnmatch.fnmatchcase(r, p + "/*") for p in pats):
            bad.append((it["id"], "below a PROTECTED_PATHS.tsv row (ask the user)"))
        refs = referenced_by(r, sources)
        if refs and it["kind"] != "resume_state":
            bad.append((it["id"], f"named by {refs[:3]}"))
    jobs = ctx.jobs_fn()
    if jobs is None:
        return bad + [(it["id"], "the sco job list could not be read (nothing is decided on a partial view)")
                      for it in items]
    unfinished = {str(j.get("name")): j for j in jobs if str(j.get("state")) not in FINISHED}
    texts = {jid: json.dumps(j, ensure_ascii=False) for jid, j in unfinished.items()}
    tops = {Path(chain_dir)} | {Path(it["run_dir"]) for it in items}
    for top in tops:
        for d in job_dirs(top) if top.is_dir() else []:
            try:
                jid = json.loads((d / "submission.json").read_text()).get("job_id")
            except (OSError, ValueError):
                continue
            if jid not in unfinished:
                continue
            body = "".join((d / n).read_text(errors="ignore") for n in ("launch.sh", "prepared.json") if (d / n).is_file())
            for it in items:
                if under(str(d), it["run_dir"]):
                    bad.append((it["id"], f"its run has an unfinished job {jid} ({d})"))
                elif mentions(body, it["root"]):
                    bad.append((it["id"], f"unfinished job {jid} ({d}) names it"))
    for jid, text in texts.items():
        for it in items:
            if mentions(text, it["root"]) or mentions(text, it["run_dir"]):
                bad.append((it["id"], f"unfinished job {jid} mentions it"))
    for r, who in open_below(ctx, [it["root"] for it in items]).items():
        bad += [(it["id"], f"open on the login node: {who[:3]}") for it in items if it["root"] == r]
    cutoff = ctx.clock() - ctx.min_age_s
    by_item: Dict[str, List[dict]] = {}
    for row in rows:
        by_item.setdefault(row["item"], []).append(row)
    for it in items:
        rs = by_item.get(it["id"], [])
        if not rs:
            bad.append((it["id"], "no files"))
        if any(x["mtime"] > cutoff for x in rs):
            bad.append((it["id"], f"a file changed in the last {ctx.min_age_s / 3600:.1f} h"))
        if any(x["uid"] != ctx.uid for x in rs):
            bad.append((it["id"], "a file of another owner"))
    return bad


# ------------------------------------------------------------------------------------------------ records
def records(ctx: Ctx, batch: str) -> Path:
    return Path(ctx.records_dir) / batch


def write_plan(path: Path, rows: Sequence[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write("item\tpath\tbytes\tmtime\tuid\tnlink\n")
        for r in rows:
            f.write("%s\t%s\t%d\t%.6f\t%d\t%d\n" % (r["item"], r["path"], r["bytes"], r["mtime"], r["uid"], r["nlink"]))
    os.replace(tmp, path)


def read_plan(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        f.readline()
        for line in f:
            item, p, b, mt, uid, nl = line.rstrip("\n").split("\t")
            rows.append({"item": item, "path": p, "bytes": int(b), "mtime": float(mt), "uid": int(uid), "nlink": int(nl)})
    return rows


def show(rows: Sequence[dict], items: Sequence[dict], title: str, full: bool = True) -> None:
    total = sum(r["bytes"] for r in rows)
    print("== %s: %d files, %.2f GB" % (title, len(rows), total / 1e9))
    if full:
        for r in rows:
            print("%16d  %s" % (r["bytes"], r["path"]))
    for it in items:
        rs = [r for r in rows if r["item"] == it["id"]]
        print("   %s %-12s %6d files %9.2f GB  %s  (%s)" % (it["id"], it["kind"], len(rs), sum(r["bytes"] for r in rs) / 1e9,
                                                         it["root"], it["why"]))
    print("== total %d files, %.2f GB (%d bytes)" % (len(rows), total / 1e9, total), flush=True)


def load_spec(spec_path: str) -> dict:
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("line") != "q38":
        raise Refused(f"{spec_path} is not a q38 offload spec")
    return spec


def refuse_on(bad: Sequence[Tuple[str, str]]) -> None:
    if bad:
        for iid, why in bad[:60]:
            print("REFUSE", iid, why)
        raise Refused("refused: %d problems (nothing uploaded or deleted)" % len(bad))


# ------------------------------------------------------------------------------------------------ phases
def cmd_select(plan_path: str, ctx: Ctx, keep: Sequence[str], resume_state: str, write: bool,
               batch: Optional[str] = None, scorer=None, keep_rule: str = "last") -> Optional[str]:
    items, report = select(plan_path, ctx, keep, resume_state, scorer, keep_rule)
    for n in report["notes"]:
        print("NOTE", n)
    for name, rep in report["runs"].items():
        print("RUN", name, json.dumps({k: v for k, v in rep.items() if k != "val"}, default=str))
    if not items:
        print("nothing to offload")
        return None
    rows = [r for it in items for r in enumerate_item(it)]
    show(rows, items, "selected (dry run)" if not write else "selected", full=False)
    bad = guards(items, rows, ctx, report["chain"])
    for iid, why in bad:
        print("REFUSE", iid, why)
    if bad:
        refused = {i for i, _ in bad}
        items = [it for it in items if it["id"] not in refused]
        rows = [r for r in rows if r["item"] not in refused]
        print("%d items left after the guards" % len(items))
    if not items or not write:
        if not write:
            print("dry run: add --write to write the spec and plan.tsv")
        return None
    batch = batch or "q38_%s_%s" % (Path(report["chain"]).name, dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    spec = {"batch": batch, "line": "q38", "approved": APPROVAL, "created_utc": now(), "plan_file": str(plan_path),
            "chain": report["chain"], "keep": list(keep), "resume_state": resume_state, "keep_rule": keep_rule,
            "items": items,
            "report": json.loads(json.dumps(report, default=str))}
    run = records(ctx, batch)
    run.mkdir(parents=True, exist_ok=False)
    os.makedirs(ctx.spec_dir, exist_ok=True)
    spec_path = Path(ctx.spec_dir) / f"{batch}.json"
    spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    write_plan(run / "plan.tsv", rows)
    print("wrote %s and %s" % (spec_path, run / "plan.tsv"))
    return str(spec_path)


def remote_size(c, key: str) -> Optional[int]:
    try:
        return int(c.head_object(Bucket=BUCKET, Key=key)["ContentLength"])
    except Exception:  # absent or unreadable: treated as not archived
        return None


def put_meta(c, ctx: Ctx, batch: str, names: Iterable[str]) -> None:
    for n in names:
        p = records(ctx, batch) / n
        if p.exists():
            c.upload_file(str(p), BUCKET, "%s/%s/%s" % (MANIFEST_PREFIX, batch, n))


def read_manifest(path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                if m.get("ok"):
                    done[m["path"]] = m
    return done


def cmd_archive(spec_path: str, ctx: Ctx, execute: bool, files: int = 2, threads: int = 2) -> int:
    spec = load_spec(spec_path)
    run = records(ctx, spec["batch"])
    rows = read_plan(run / "plan.tsv")
    refuse_on(guards(spec["items"], rows, ctx, spec["chain"]))
    man_path = run / "manifest.jsonl"
    done = read_manifest(man_path)
    todo = [r for r in rows if r["path"] not in done or done[r["path"]]["bytes"] != r["bytes"]]
    print("%d planned, %d already archived, %d to upload (%.2f GB); files x threads = %d x %d"
          % (len(rows), len(rows) - len(todo), len(todo), sum(r["bytes"] for r in todo) / 1e9, files, threads))
    if not execute:
        print("dry run: add --execute to upload")
        return 0
    c = ctx.client_fn(files, threads)
    try:
        from boto3.s3.transfer import TransferConfig

        opts = {"Config": TransferConfig(max_concurrency=threads, multipart_chunksize=64 << 20,
                                         multipart_threshold=64 << 20)}
    except ImportError:  # only the tests' fake client gets here (the q38 environment has no boto3)
        opts = {}
    lock, moved, t0 = threading.Lock(), [0], time.time()

    def one(r: dict) -> dict:
        k = ctx.key_of(r["path"])
        err = None
        if remote_size(c, k) != r["bytes"]:
            for attempt in range(3):
                try:
                    c.upload_file(r["path"], BUCKET, k, **opts)
                    err = None
                    break
                except Exception as e:  # retried, then reported
                    err = repr(e)[:300]
                    time.sleep(10 * (attempt + 1))
        got = remote_size(c, k)
        with lock:
            moved[0] += r["bytes"]
        return {"path": r["path"], "key": k, "bytes": r["bytes"], "remote_bytes": got, "mtime": r["mtime"],
                "ok": err is None and got == r["bytes"], "error": err, "utc": now()}

    bad = 0
    with ThreadPoolExecutor(files) as pool, open(man_path, "a") as man:
        futs = [pool.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            m = f.result()
            man.write(json.dumps(m) + "\n")
            man.flush()
            bad += not m["ok"]
            if i % 10 == 0 or i == len(futs):
                print("  %d/%d files, %.1f GB, %.0f MB/s, %d not ok" % (
                    i, len(futs), moved[0] / 1e9, moved[0] / max(time.time() - t0, 1) / 1e6, bad), flush=True)
    put_meta(c, ctx, spec["batch"], ["plan.tsv", "manifest.jsonl"])
    print("archive done: %d not ok; manifest %s (copy on AOSS %s/%s/)" % (bad, man_path, MANIFEST_PREFIX, spec["batch"]))
    return bad


def cmd_delete(spec_path: str, ctx: Ctx, yes: bool, scorer=None) -> int:
    spec = load_spec(spec_path)
    run = records(ctx, spec["batch"])
    rows = read_plan(run / "plan.tsv")
    items = spec["items"]
    refuse_on(guards(items, rows, ctx, spec["chain"]))
    # the selection rule once more: an item another plan has started to use since, or a changed best, stays
    # the spec's own rule (a spec written before keep_rule existed used best_and_last)
    again, _ = select(spec["plan_file"], ctx, spec.get("keep") or (), spec.get("resume_state", "keep"), scorer,
                      spec.get("keep_rule", "best_and_last"))
    still = {(it["kind"], it["root"]) for it in again}
    refuse_on([(it["id"], "no longer selected by the rule (best, last, --keep or a new reference)")
               for it in items if (it["kind"], it["root"]) not in still])
    man = read_manifest(run / "manifest.jsonl")
    c = ctx.client_fn(1, 1)
    problems = []
    for r in rows:
        m = man.get(r["path"])
        if not m or m["remote_bytes"] != r["bytes"]:
            problems.append((r["item"], "not archived: " + r["path"]))
            continue
        if remote_size(c, m["key"]) != r["bytes"]:  # asked again right before the delete
            problems.append((r["item"], "AOSS object missing or of another size now: " + m["key"]))
            continue
        try:
            st = os.lstat(r["path"])
        except FileNotFoundError:
            problems.append((r["item"], "already gone (someone else deleted it): " + r["path"]))
            continue
        if st.st_size != r["bytes"] or abs(st.st_mtime - r["mtime"]) > 1e-3:
            problems.append((r["item"], "changed since the plan: " + r["path"]))
    refuse_on(problems)
    show(rows, items, "TO DELETE, batch %s" % spec["batch"])
    if not yes:
        print("dry run: add --yes to delete")
        return 0
    log_path = run / "deleted.tsv"
    touched = set()
    with open(log_path, "a") as log:
        if log_path.stat().st_size == 0:
            log.write("utc\titem\tpath\tbytes\taoss_key\n")
        for r in rows:
            os.remove(r["path"])
            log.write("%s\t%s\t%s\t%d\t%s\n" % (now(), r["item"], r["path"], r["bytes"], ctx.key_of(r["path"])))
            touched.add(os.path.dirname(r["path"]))
    by_id = {it["id"]: it for it in items}
    for d in sorted(touched, key=lambda x: -x.count("/")):  # emptied folders, deepest first
        it = by_id[next(r["item"] for r in rows if os.path.dirname(r["path"]) == d)]
        stop = os.path.dirname(it["root"]) if it["kind"] != "resume_state" else it["root"]
        while under(d, it["root"]) and d != stop:
            try:
                os.rmdir(d)
            except OSError:
                break
            d = os.path.dirname(d)
    for it in items:
        rs = [r for r in rows if r["item"] == it["id"]]
        os.makedirs(it["marker_dir"], exist_ok=True)
        what = "the resume state of" if it["kind"] == "resume_state" else "all of"
        note = (f"{what} {it['root']} ({it['kind']}, {it['why']}; {len(rs)} files, "
                f"{sum(r['bytes'] for r in rs) / 1e9:.2f} GB) archived to fj:s3://{BUCKET}/{PREFIX}/<same path below "
                f"{ctx.home}> and deleted {now()}, batch {spec['batch']} ({spec['approved']}).\n"
                f"Records: {run}/ (deleted.tsv) and fj:s3://{BUCKET}/{MANIFEST_PREFIX}/{spec['batch']}/.\n"
                f"Restore: python /mnt/umm/users/yinbaiqiao/VAGEN-Lite-disk/disk/restore_from_aoss.py {spec['batch']} "
                f"--match '^{re.escape(it['root'])}/' --yes (probe-spatial Python, proxies unset, nice -n 19)\n")
        name = "%s.%s.txt" % (Path(it["root"]).name + (".resume_state" if it["kind"] == "resume_state" else ""),
                              spec["batch"])
        Path(it["marker_dir"], name).write_text(note)
        ledger = Path(it["marker_dir"]) / "LEDGER.tsv"
        new = not ledger.exists()
        with open(ledger, "a") as f:
            if new:
                f.write("utc\tbatch\titem\tkind\troot\tfiles\tbytes\n")
            f.write("%s\t%s\t%s\t%s\t%s\t%d\t%d\n" % (now(), spec["batch"], it["id"], it["kind"], it["root"], len(rs),
                                                     sum(r["bytes"] for r in rs)))
    put_meta(c, ctx, spec["batch"], ["deleted.tsv"])
    print("deleted %d files, %.2f GB; record %s (copy on AOSS)" % (len(rows), sum(r["bytes"] for r in rows) / 1e9, log_path))
    return len(rows)


def cmd_status(spec_path: str, ctx: Ctx) -> None:
    spec = load_spec(spec_path)
    run = records(ctx, spec["batch"])
    rows = read_plan(run / "plan.tsv")
    man = read_manifest(run / "manifest.jsonl")
    deleted = (run / "deleted.tsv").read_text().count("\n") - 1 if (run / "deleted.tsv").exists() else 0
    print("batch %s: %d items, %d files %.2f GB planned; %d archived ok; %d deleted" % (
        spec["batch"], len(spec["items"]), len(rows), sum(r["bytes"] for r in rows) / 1e9, len(man), max(deleted, 0)))


def main(argv: Optional[Sequence[str]] = None, ctx: Optional[Ctx] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="phase", required=True)
    s = sub.add_parser("select")
    s.add_argument("plan")
    s.add_argument("--keep", action="append", default=[], help="a checkpoint folder to keep locally (repeatable)")
    s.add_argument("--resume-state", choices=["keep", "archive"], default="keep")
    s.add_argument("--keep-rule", choices=list(KEEP_RULES), default="last",
                   help="which RL checkpoints stay local (定 5: the one the decision uses, the last)")
    s.add_argument("--batch")
    s.add_argument("--write", action="store_true")
    a = sub.add_parser("archive")
    a.add_argument("spec")
    a.add_argument("--execute", action="store_true")
    a.add_argument("--files", type=int, default=2)
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--in-sco-job", action="store_true", help="running inside a one-card sco job: files x threads may exceed 4")
    d = sub.add_parser("delete")
    d.add_argument("spec")
    d.add_argument("--yes", action="store_true")
    t = sub.add_parser("status")
    t.add_argument("spec")
    args = ap.parse_args(argv)
    for k in PROXY_VARS:  # AOSS hangs through the login-node proxy
        os.environ.pop(k, None)
    ctx = ctx or Ctx()
    if args.phase == "select":
        cmd_select(args.plan, ctx, args.keep, args.resume_state, args.write, args.batch, keep_rule=args.keep_rule)
    elif args.phase == "archive":
        if args.files * args.threads > 4 and not args.in_sco_job:
            raise Refused("login node: files x threads must be <= 4 (the user's rule of 2026-10-28)")
        return 1 if cmd_archive(args.spec, ctx, args.execute, args.files, args.threads) else 0
    elif args.phase == "delete":
        cmd_delete(args.spec, ctx, args.yes)
    else:
        cmd_status(args.spec, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
