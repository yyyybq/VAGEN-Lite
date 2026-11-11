"""TVR pilot gate G0-G3, A3 (report only) and A4 (CONTRACT_wave2 section 8.1; PREREG_TVR_PILOT.md section 13;
mirrors vpd/walkerbench/pilot_gate.py).

    nice -n 19 PY_EVAL -m vpd.tvr.gate_tvr [--pilot-root OUT/tvr/runs/pilot] [--starts Q3B,Q3S,E2] [--power]
    -> PILOT_ROOT/GATE.json and GATE.md.  Exit 0 = every gate decided and passed (decision "ruler"), 1 = a gate
       failed, 3 = incomplete (missing evals / files, or A4 still pending).

Per start S the gated evals are P/S/evals/pilot__S__{0W,100,TOP}__c, pilot__S__TOP__u, pilot__S__TOP__blind
(+ pilot__Q3B__100s1002__c, the seed replicate: G0 mechanism items and power only).  TOP = pilot_cfg tiers.n_top.
G0 (all 17 items must hold; any missing input = incomplete):
   1 evals_valid           every listed eval valid (results.json), 500 tasks, none missing
   2 templates             smoke B5 (vllm_parity_S) + parity_check (hfparity_S) PASS; every image eval's prompt parity
   3 image_tokens          222 tokens / grid [1,22,40]: smoke dry runs and every image eval's token audit
   4 renderer              profile, font sha, L == render.* in the replay and every image eval; E == replay_envs
                           (replay) / eval_envs (image evals); renderer.workers_ok is True and every worker's own
                           ready reply has profile / L == render.*, the pinned font, the pinned system prompt and
                           TVRBench files (pilot_tvr.worker_problems)
   5 code                  the three starts' 0W share one code_sha256 starting with the frozen prefix; 100 / TOP from
                           their own 0W (100_s1002 from 0W_s1002; 0W_s1002 with the code of its start's 0W); one
                           harness code_sha256 (results.code_sha256 == code_fingerprint.code_sha256) across all evals
                           (the evals only: the data job's replay may come from older run_tvr code)
   6 kin                   kin_equal_all_kept (demos) AND per image eval 0 valid-list mismatches and kin-unequal tasks
                           <= gates.kin_eval.task_mismatch_max_frac of its tasks
   7 merge_parity          P/S/parity/merge_<tier>.json pass (and of the adapter.pt now on disk) for EVERY merged eval
   8 served_adapter        every merged eval served the adapter.pt now on disk
   9 attn_pixels           sdpa at every attention site of every adapter; max_pixels in training and in vLLM
  10 settings              results.settings == pilot_cfg protocol and the per-run values run_tvr records:
                           max_tokens_used == protocol.max_tokens[arm], regex_used (protocol.regex for c / blind, None
                           for u), blind_concurrency_used (protocol.blind_concurrency for blind, else None),
                           request_fields == the registered body keys, max_model_len_used == the judged max_model_len;
                           server block (results.server.manifest: cfg, cmd, extra_args) and versions; max_model_len
                           (== the frozen value with no fallback_used.json, or == the O9 fallback in EVERY eval when it
                           is registered: o9_fallback())
  11 reach_frozen          every eval task's trajectory reach_digest == reach/v1/index.json
  12 official_aggregate    TVRBench eval_va.aggregate() SR == ours (official_check.json equal) for every eval
  13 adapter_provenance    meta start / seed / phase / optimizer_steps == planned / data_sha256 == tiers manifest,
                           whose sha == data.tiers_manifest_sha256; tiers.n_top == CAL/cal_decision.json n_top (the
                           calibration rule's output; with_400 True for 400: pilot_tvr.n_top_problems)
  14 floors_file           floors_eval500.json sha == data.floors_eval500_sha256, warnings [], F == data.F_eval500
  15 render_consistency    the smoke's root sft2 replay and every eval job's canary pass the tolerance rule
  16 demos_replay          the replay valid with 1,600 trajectories; kept >= gates.min_kept; every drop listed with a
                           G10(a) reason
  17 independent_recompute k/n (and per category) recounted from trajectory.json (recheck_tvr.counts) == results.json
                           for every eval; recheck_tvr.floors == floors_eval500.json
  recorded, not judged: reach live == frozen counts, unstable reach tasks, render-parity PSNR / SSIM minima.
G1 learnable: some start has wilson_lb(k_TOP_c, 500) > max(F, that start's 0W c rate).
G2 format: every c and blind eval: 0 regex mismatches over > 0 replies, 500 trajectories, none missing.
G3 not saturated: every start's TOP c rate <= gates.saturation.
A3 report only: per start TOP c vs TOP blind on the same tasks: b (only with images), c (only without), exact
   two-sided binomial p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2); "images_better_p_lt_0.05" (b > c and
   p < gates.mcnemar_alpha).
A4: pending until G0-G3 pass; then POWER.json (vpd.common.power_sim tvr; --power runs it): pass / fail; a POWER.json
   whose input shas / pilot_cfg sha no longer match (power_stale) or that was not made at the registered settings
   (power_unregistered: registered.test, n_sims, tier_corr, fallback, persistence, backend == pilot_cfg power.*,
   no allow_no_replicate, the replicate present) is pending.
decision: ruler | not_a_ruler (G1 or G3 fail) | engineering (G0 or G2 fail) | power_insufficient | incomplete.
Every number is an absolute per start; nothing here computes or prints a difference between starts.
"""

from __future__ import annotations

import argparse
import fractions
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from vpd import paths
from vpd.tvr import pilot_cfg, pilot_tvr, recheck_tvr
from vpd.walkerbench.pilot_gate import wilson_lb, wilson_ub

SCHEMA = "vpd.tvr.gate/1"
STARTS = tuple(pilot_cfg.STARTS)
G0_ITEMS = ("evals_valid", "templates", "image_tokens", "renderer", "code", "kin", "merge_parity", "served_adapter",
            "attn_pixels", "settings", "reach_frozen", "official_aggregate", "adapter_provenance", "floors_file",
            "render_consistency", "demos_replay", "independent_recompute")
NOTE = "no between-start differences are computed"


def read_json(p) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def sha256_file(p) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def block(k: int, n: int, z: float) -> dict:
    return {"k": k, "n": n, "rate": None if not n else round(k / n, 4),
            "wilson95": None if not n else [round(wilson_lb(k, n, z), 4), round(wilson_ub(k, n, z), 4)]}


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial (McNemar) p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2); 1 if b + c = 0."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1))
    return min(1.0, float(fractions.Fraction(2 * tail, 2 ** n)))


def _item(name: str, bad: list, missing: list, detail=None) -> dict:
    ok = False if bad else (None if missing else True)
    return {"name": name, "ok": ok, "detail": {"problems": bad[:40], "missing": missing[:40],
                                               **({} if detail is None else {"facts": detail})}}


def _tri(oks) -> str:
    oks = list(oks)
    if any(o is False for o in oks):
        return "fail"
    if any(o is None for o in oks) or not oks:
        return "incomplete"
    return "pass"


def _plain(x):
    return json.loads(json.dumps(x))


# --- loading ----------------------------------------------------------------------------------------------------

class Eval:
    def __init__(self, root: Path, s: str, tier: str, arm: str):
        self.s, self.tier, self.arm = s, tier, arm
        self.key = f"{tier}__{arm}"
        self.dir = Path(root) / s / "evals" / f"pilot__{s}__{tier}__{arm}"
        self.results = read_json(self.dir / "results.json")
        self.valid = self.results is not None
        if self.results is None:
            self.results = read_json(self.dir / "results_invalid.json")
        self.present = self.results is not None
        self._traj = None

    @property
    def image(self) -> bool:
        return self.arm in pilot_tvr.IMAGE_ARMS

    @property
    def merged(self) -> bool:
        return True  # every pilot eval serves merged weights (no raw evals, O11)

    def trajectories(self) -> Dict[str, dict]:
        if self._traj is None:
            self._traj = {}
            td = self.dir / "tasks"
            for p in sorted(td.glob("*/trajectory.json")) if td.is_dir() else []:
                t = read_json(p)
                if t is not None:
                    self._traj[t.get("task_id")] = t
        return self._traj

    def r(self, *keys, default=None):
        cur = self.results or {}
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return default if cur is None else cur

    def manifest(self) -> Optional[dict]:
        """results.server.manifest (VLLMServer.manifest(): cfg, cmd, extra_args, mm_processor_kwargs); None if absent."""
        m = self.r("server", "manifest")
        return m if isinstance(m, dict) else None


def o9_fallback(L: pilot_tvr.Layout, value) -> dict:
    """Is the O9 fallback (protocol.max_model_len_fallback) registered?  It is iff (a) CAL/shake/fallback_used.json
    exists and records this value (the cal shakedown's retry wrote it), (b) the prereg as it is now quotes that file's
    sha256 (the section-17 deviation entry), and (c) every msmoke / train / eval run was prepared with a prereg commit
    whose text already quotes it (registered before the freeze; RUN/prepared.json["prereg"]["commit"])."""
    f = L.fallback
    sha = sha256_file(f)
    out = {"file": str(f), "file_sha256": sha, "registered": False, "problems": []}
    if sha is None:
        out["problems"].append(f"max_model_len {value} (the O9 fallback) but no {f}")
        return out
    doc = read_json(f) or {}
    if doc.get("max_model_len") != value:
        out["problems"].append(f"{f}: max_model_len {doc.get('max_model_len')!r} != {value}")
    try:
        text = pilot_tvr.prereg_path().read_text(encoding="utf-8")
    except OSError:
        text = ""
    if sha not in text:
        out["problems"].append(f"the prereg has no O9 deviation entry quoting the sha256 of {f.name} ({sha})")
    commits = {}
    for job, spec in pilot_tvr.JOBS.items():
        if spec["kind"] not in ("msmoke", "train", "eval"):
            continue
        for run in pilot_tvr.runs_of(job, L):
            c = ((read_json(run / "prepared.json") or {}).get("prereg") or {}).get("commit")
            if c:
                commits[run.name] = c
    if not commits:
        out["problems"].append("no msmoke / train / eval run records the prereg commit it was prepared with")
    for name, c in sorted(commits.items()):
        t = pilot_tvr.prereg_text_at(c)
        if t is None or sha not in t:
            out["problems"].append(f"{name} was prepared with prereg commit {c[:12]}, whose text does not register "
                                   f"the O9 fallback (sha256 {sha[:12]})")
    out["registered"] = not out["problems"]
    return out


# the request body keys (CONTRACT 4.6; run_tvr's docstring and Convo): + structured_outputs on the regex arms
REQUEST_FIELDS = ("model", "messages", "temperature", "top_p", "max_tokens", "top_k")
REGEX_ARMS = ("c", "blind")


def used_settings(arm: str, proto: dict) -> dict:
    """The per-run values run_tvr records in results.settings (Harness.settings: arm, *_used, request_fields) as the
    protocol registers them for `arm`; max_model_len_used is judged against the served value in G0(10)."""
    rx = arm in REGEX_ARMS
    return {"arm": arm, "max_tokens_used": proto["max_tokens"][arm], "regex_used": proto["regex"] if rx else None,
            "blind_concurrency_used": proto["blind_concurrency"] if arm == "blind" else None,
            "request_fields": sorted(REQUEST_FIELDS + (("structured_outputs",) if rx else ()))}


def eval_list(s: str, cfg=None) -> List[tuple]:
    top = pilot_cfg.n_top(cfg)
    out = [("0W", "c"), ("100", "c"), (top, "c"), (top, "u"), (top, "blind")]
    rep = pilot_tvr.replicate(cfg)
    if s == rep["start"]:
        out.append((f"{rep['tier']}s{rep['seed']}", "c"))
    return out


# --- G0 -----------------------------------------------------------------------------------------------------------

def g0_items(root: Path, evals: Dict[str, Dict[str, Eval]], starts, L: pilot_tvr.Layout, cfg, sv: Optional[dict],
             sv_path: Optional[Path], recheck_args: dict) -> tuple:
    proto = pilot_cfg.get("protocol", cfg)
    render = pilot_cfg.get("render", cfg)
    all_ev = [e for s in starts for e in evals[s].values()]
    items, recorded = [], {}
    sv_checks = {c["name"]: c for c in (sv or {}).get("checks", [])}

    # 1 evals_valid
    bad, miss = [], []
    for e in all_ev:
        if not e.present:
            miss.append(f"{e.s}/{e.key}")
        elif not e.valid or e.r("n_tasks") != pilot_tvr.EVAL500_N or e.r("tasks_missing", default=[]):
            bad.append(f"{e.s}/{e.key}: valid {e.valid}, n_tasks {e.r('n_tasks')}, missing "
                       f"{e.r('tasks_missing', default=[])[:3]}")
    items.append(_item("evals_valid", bad, miss))

    # 2 templates, 3 image tokens
    bad, miss, bad3, miss3 = [], [], [], []
    for s in starts:
        for name, tgt in ((f"hfparity_{s}", bad), (f"vllm_parity_{s}", bad), (f"dry_{s}", bad3)):
            c = sv_checks.get(name)
            if c is None:
                (miss if tgt is bad else miss3).append(f"smoke {name}")
            elif c.get("ok") is not True:
                tgt.append(f"smoke {name}: {json.dumps(c.get('detail'), default=str)[:160]}")
    grid = list(proto["image_grid"])
    for e in all_ev:
        if not e.present or not e.image:
            continue
        if e.r("audit", "prompt_parity", "equal") is not True:
            bad.append(f"{e.s}/{e.key}: prompt parity {e.r('audit', 'prompt_parity', 'equal')}")
        tpi = e.r("audit", "image_tokens", "tokens_per_image")
        tpi = list(tpi) if isinstance(tpi, (list, tuple)) else [tpi]
        gs = [list(g) for g in e.r("audit", "image_tokens", "grid_set", default=[])]
        if tpi != [proto["image_tokens"]] or gs != [grid]:
            bad3.append(f"{e.s}/{e.key}: tokens per image {tpi}, grids {gs}")
    items.append(_item("templates", bad, miss))
    items.append(_item("image_tokens", bad3, miss3))

    # 4 renderer
    bad, miss = [], []
    rp = read_json(L.replay / "results.json")

    def rend(where, rr, envs_key):
        want = {"profile": render["lvp_profile"], "font_sha256": render["badge_font_sha256"],
                "lp": render["lp_threads"], "envs": render[envs_key]}
        got = {k: (rr or {}).get(k) for k in want}
        if got != want or None in want.values():
            bad.append(f"{where}: renderer {got} != {want}")
        # the profile / lp above are the harness's request; the workers' own ready replies are judged here
        wp = pilot_tvr.worker_problems(rr, cfg)
        if wp:
            bad.append(f"{where}: renderer workers_ok {(rr or {}).get('workers_ok')!r}; {wp}"[:600])

    if rp is None:
        miss.append(f"{L.replay}/results.json")
    else:
        rend("replay", rp.get("renderer"), "replay_envs")
    for e in all_ev:
        if e.present and e.image:
            rend(f"{e.s}/{e.key}", e.r("renderer"), "eval_envs")
    items.append(_item("renderer", bad, miss))

    # 5 code
    bad, miss = [], []
    prefix = pilot_cfg.get("recipe.trainer_code_sha256_prefix", cfg)
    zero = {}
    for s in starts:
        m0 = read_json(L.adapter(s, "0W") / "meta.json")
        if m0 is None:
            miss.append(f"{s}/adapters/0W/meta.json")
            continue
        zero[s] = m0.get("code_sha256")
        for tier, init in pilot_tvr.train_tiers(s, cfg):
            if tier == "0W":
                continue
            m = read_json(L.adapter(s, tier) / "meta.json")
            sha0 = None if init is None else sha256_file(L.adapter(s, init) / "adapter.pt")
            if m is None:
                miss.append(f"{s}/adapters/{tier}/meta.json")
            elif init is not None and (sha0 is None or (m.get("init_adapter") or {}).get("sha256") != sha0):
                bad.append(f"{s}/{tier}: init {str((m.get('init_adapter') or {}).get('sha256'))[:12]} != "
                           f"{init}/adapter.pt {str(sha0)[:12]}")
            elif m.get("code_sha256") != zero[s]:  # 100 / TOP and the replicate's own 0W_s1002
                bad.append(f"{s}/{tier}: code sha differs from this start's 0W")
    if len(zero) == len(starts):
        vals = set(zero.values())
        if len(vals) != 1 or None in vals or not str(next(iter(vals))).startswith(prefix):
            bad.append(f"0W code_sha256 {sorted(str(v)[:12] for v in vals)} (want one value starting {prefix})")
    # the harness fingerprint: run_tvr's top-level results.code_sha256 (== code_fingerprint.code_sha256); results.code
    # is the served start code (a string), not a fingerprint.  The evals only (PREREG 13 G0(5)): the data job's
    # replay was made before the evals and may carry an older run_tvr fingerprint.
    harness = set()
    for e in all_ev:
        if not e.present:
            continue
        h, hf = e.r("code_sha256"), e.r("code_fingerprint", "code_sha256")
        if h is None or hf is None:
            miss.append(f"{e.s}/{e.key}: results." + ("code_sha256" if h is None else "code_fingerprint.code_sha256"))
            continue
        if hf != h:
            bad.append(f"{e.s}/{e.key}: code_sha256 {str(h)[:12]} != code_fingerprint {str(hf)[:12]}")
        harness.add(h)
    if len(harness) > 1:
        bad.append(f"harness code_sha256 across evals: {sorted(str(h)[:12] for h in harness)}")
    items.append(_item("code", bad, miss, {"zero_w_code": {s: str(v)[:12] for s, v in zero.items()}}))

    # 6 kin
    bad, miss = [], []
    man = read_json(L.demos / "manifest.json")
    if man is None:
        miss.append(f"{L.demos}/manifest.json")
    elif man.get("kin_equal_all_kept") is not True:
        bad.append(f"demos kin_equal_all_kept {man.get('kin_equal_all_kept')!r}")
    frac = float(pilot_cfg.get("gates.kin_eval.task_mismatch_max_frac", cfg))
    vmax = int(pilot_cfg.get("gates.kin_eval.valid_list_mismatch_max", cfg))
    kin_facts = {}
    for e in all_ev:
        if not e.present or not e.image:
            continue
        k = e.r("kin", default={})
        vl, un, n = k.get("valid_list_mismatches"), k.get("tasks_unequal"), e.r("n_tasks")
        kin_facts[f"{e.s}/{e.key}"] = {"valid_list_mismatches": vl, "tasks_unequal": un,
                                       "pose_or_success_mismatches": k.get("pose_or_success_mismatches"),
                                       "first": (k.get("first") or [])[:3]}
        if not isinstance(vl, int) or not isinstance(un, int) or not isinstance(n, int) or not n:
            miss.append(f"{e.s}/{e.key}: kin counts")
        elif vl > vmax:
            bad.append(f"{e.s}/{e.key}: {vl} valid-list mismatches (broken injection; max {vmax})")
        elif fractions.Fraction(un, n) > fractions.Fraction(str(frac)):
            bad.append(f"{e.s}/{e.key}: {un}/{n} kin-unequal tasks > {frac}")
    items.append(_item("kin", bad, miss, kin_facts))

    # 7 merge parity, 8 served adapter
    bad, miss, bad8, miss8 = [], [], [], []
    for s in starts:
        seen = set()
        for e in evals[s].values():
            ad = L.adapter(s, pilot_tvr.EVAL_ADAPTER[e.tier])
            if e.tier not in seen:
                seen.add(e.tier)
                p = L.parity(s, e.tier)
                rc, msg = pilot_tvr.parity_state(p, ad)
                if not p.is_file():
                    miss.append(f"{s}: {p.name}")
                elif rc != 0:
                    bad.append(f"{s}: {msg}")
            if not e.present:
                continue
            disk = sha256_file(ad / "adapter.pt")
            served = e.r("adapter", "adapter_sha256")
            if disk is None:
                miss8.append(f"{s}/{e.key}: {ad}/adapter.pt")
            elif served != disk:
                bad8.append(f"{s}/{e.key}: served {str(served)[:12]} != {ad.name}/adapter.pt {disk[:12]}")
    items.append(_item("merge_parity", bad, miss))
    items.append(_item("served_adapter", bad8, miss8))

    # 9 attention and pixels
    bad, miss = [], []
    mp = int(proto["max_pixels"])
    for s in starts:
        for tier, _ in pilot_tvr.train_tiers(s, cfg):
            m = read_json(L.adapter(s, tier) / "meta.json")
            if m is None:
                miss.append(f"{s}/adapters/{tier}/meta.json")
                continue
            sites = m.get("attn_sites") or {}
            if m.get("attn") != pilot_cfg.get("recipe.attn", cfg) or not sites or \
                    any(v != pilot_cfg.get("recipe.attn", cfg) for v in sites.values()):
                bad.append(f"{s}/{tier}: attention {m.get('attn')} {sites}")
            if m.get("max_pixels") != mp:
                bad.append(f"{s}/{tier}: training max_pixels {m.get('max_pixels')}")
    for e in all_ev:
        if not e.present:
            continue
        smf = e.manifest()
        if smf is None:
            miss.append(f"{e.s}/{e.key}: results.server.manifest")
            continue
        mm = smf.get("mm_processor_kwargs") or {}
        if mm.get("max_pixels") != mp or (mm.get("size") or {}).get("longest_edge") != mp:
            bad.append(f"{e.s}/{e.key}: vLLM mm_processor_kwargs {mm}")
    items.append(_item("attn_pixels", bad, miss))

    # 10 settings
    bad, miss = [], []
    pro = _plain(proto)
    srv = pro["server"]
    vers = pro["versions"]
    mml_used, mml_set = {}, {}
    for e in all_ev:
        if not e.present:
            continue
        st = e.r("settings")
        if not isinstance(st, dict):
            miss.append(f"{e.s}/{e.key}: results.settings")
            continue
        st = _plain(st)
        for k, v in pro.items():
            if k in ("max_model_len", "max_model_len_fallback"):
                continue
            if st.get(k) != v:
                bad.append(f"{e.s}/{e.key}: settings.{k} {st.get(k)!r} != {v!r}")
        # results.settings starts as a copy of protocol (run_tvr.settings), so the keys above are equal by
        # construction; the values this run actually used are run_tvr's *_used fields and request_fields
        for k, v in used_settings(e.arm, pro).items():
            if k not in st:
                miss.append(f"{e.s}/{e.key}: settings.{k}")
            elif st[k] != v:
                bad.append(f"{e.s}/{e.key}: settings.{k} {st[k]!r} != {v!r}")
        mml_set[f"{e.s}/{e.key}"] = st.get("max_model_len_used")
        # run_tvr: results.server = {manifest: VLLMServer.manifest() (cfg, cmd, extra_args, mm_processor_kwargs, ...),
        # versions, max_model_len, ready_s, served_model_name}
        sv_ = e.r("server", default={})
        smf = e.manifest()
        scfg = (smf or {}).get("cfg") or {}
        if smf is None:
            miss.append(f"{e.s}/{e.key}: results.server.manifest")
        else:
            cmd = [str(x) for x in smf.get("cmd") or []]
            if scfg.get("enforce_eager") is not srv["enforce_eager"] or scfg.get("seed") != srv["seed"]:
                bad.append(f"{e.s}/{e.key}: server enforce_eager {scfg.get('enforce_eager')} seed {scfg.get('seed')}")
            if scfg.get("max_images") != pro["max_images"]:
                bad.append(f"{e.s}/{e.key}: server max_images {scfg.get('max_images')}")
            if "--generation-config" not in cmd or cmd[cmd.index("--generation-config") + 1:][:1] != [
                    srv["generation_config"]]:
                bad.append(f"{e.s}/{e.key}: server --generation-config not {srv['generation_config']}")
            if srv["structured_outputs_backend"] not in json.dumps(smf.get("extra_args") or []):
                bad.append(f"{e.s}/{e.key}: server extra_args without {srv['structured_outputs_backend']}")
        v = sv_.get("versions") or {}
        got = {"vllm": v.get("vllm"), "xgrammar": v.get("xgrammar"), "transformers": v.get("transformers"),
               "ai2thor": v.get("ai2thor")}
        want = {"vllm": vers["vllm"], "xgrammar": vers["xgrammar"], "transformers": vers["transformers_serve"],
                "ai2thor": vers["ai2thor"]}
        if e.arm == "blind":
            got.pop("ai2thor")
            want.pop("ai2thor")
        if got != want:
            bad.append(f"{e.s}/{e.key}: versions {got} != {want}")
        mml_used[f"{e.s}/{e.key}"] = sv_.get("max_model_len", scfg.get("max_model_len"))
    vals = set(mml_used.values())
    fb = o9_fallback(L, pro["max_model_len_fallback"])
    if vals:
        if len(vals) != 1:
            bad.append(f"max_model_len differs between evals: {sorted(map(str, vals))}")
        else:
            v = next(iter(vals))
            if v == pro["max_model_len_fallback"]:
                bad += fb["problems"]
            elif v != pro["max_model_len"]:
                bad.append(f"max_model_len {v} != {pro['max_model_len']}")
            elif fb["file_sha256"] is not None:
                bad.append(f"{fb['file']} exists (the cal used the O9 fallback) but the evals ran at {v}")
            for key, u in sorted(mml_set.items()):  # the harness's own record == the judged (served) value
                if u is None:
                    miss.append(f"{key}: settings.max_model_len_used")
                elif u != v:
                    bad.append(f"{key}: settings.max_model_len_used {u!r} != the served max_model_len {v!r}")
    items.append(_item("settings", bad, miss, {"max_model_len": sorted(map(str, vals)),
                                               "o9_fallback": {k: fb[k] for k in ("file", "file_sha256", "registered")}}))

    # 11 reach frozen, 17a recount
    bad, miss, bad17, miss17 = [], [], [], []
    index = read_json(L.reach / "index.json")
    if index is None:
        miss.append(f"{L.reach}/index.json")
    live = {}
    for e in all_ev:
        if not e.present:
            continue
        tr = e.trajectories()
        live[f"{e.s}/{e.key}"] = e.r("reach", "live_equal")
        if index is not None:
            wrong = [t for t, x in tr.items() if (x.get("vpd") or {}).get("reach_digest") !=
                     ((index.get("tasks") or {}).get(t) or {}).get("digest")]
            if wrong or not tr:
                bad.append(f"{e.s}/{e.key}: {len(wrong)} of {len(tr)} tasks off the frozen set (e.g. {wrong[:2]})")
        rc = recheck_tvr.counts(e.dir)
        sr = e.r("sr", default={})
        if rc["n"] == 0:
            miss17.append(f"{e.s}/{e.key}: no trajectories")
        elif (rc["k"], rc["n"]) != (sr.get("k"), sr.get("n")):
            bad17.append(f"{e.s}/{e.key}: recount {rc['k']}/{rc['n']} != results.json {sr.get('k')}/{sr.get('n')}")
        else:
            for cat, c in rc["by_category"].items():
                got = e.r("by_category", cat, "sr", default={})
                if isinstance(got, dict) and "k" in got:
                    if (got.get("k"), got.get("n")) != (c["k"], c["n"]):
                        bad17.append(f"{e.s}/{e.key} {cat}: recount {c['k']}/{c['n']} != {got.get('k')}/"
                                     f"{got.get('n')}")
                else:
                    miss17.append(f"{e.s}/{e.key}: results.by_category.{cat}.sr")
    recorded["reach_live_equal"] = live
    recorded["unstable_reach_tasks"] = (index or {}).get("unstable_tasks")
    items.append(_item("reach_frozen", bad, miss))

    # 12 official
    bad, miss = [], []
    for e in all_ev:
        if not e.present:
            continue
        oc = read_json(e.dir / "official_check.json")
        if oc is None:
            miss.append(f"{e.s}/{e.key}: official_check.json")
        elif oc.get("equal") is not True:
            bad.append(f"{e.s}/{e.key}: official SR {oc.get('official_sr')} vs ours {oc.get('our_sr')}")
    items.append(_item("official_aggregate", bad, miss))

    # 13 adapter provenance
    bad, miss = [], []
    tman_p = L.tiers / "manifest.json"
    tman = read_json(tman_p)
    want_m = pilot_cfg.get("data.tiers_manifest_sha256", cfg)
    if tman is None:
        miss.append(str(tman_p))
    elif sha256_file(tman_p) != want_m:
        bad.append(f"tiers manifest sha {str(sha256_file(tman_p))[:12]} != data.tiers_manifest_sha256 "
                   f"{str(want_m)[:12]}")
    # the top tier trained and judged (pilot_cfg tiers.n_top, filled by hand) is the calibration rule's output
    if not L.cal_decision.is_file():
        miss.append(str(L.cal_decision))
    else:
        bad += pilot_tvr.n_top_problems(L, cfg)
    seed0, rep = int(pilot_cfg.SEED), pilot_tvr.replicate(cfg)
    for s in starts:
        for tier, _ in pilot_tvr.train_tiers(s, cfg):
            m = read_json(L.adapter(s, tier) / "meta.json")
            if m is None:
                miss.append(f"{s}/adapters/{tier}/meta.json")
                continue
            seed = int(rep["seed"]) if tier.endswith(f"_s{rep['seed']}") else seed0
            fname = pilot_tvr.TIER_FILE[tier]
            want_sha = None if tman is None else (tman.get(fname) or (tman.get("files") or {}).get(fname))
            probs = []
            if m.get("start") != s:
                probs.append(f"start {m.get('start')}")
            if m.get("seed") != seed:
                probs.append(f"seed {m.get('seed')} != {seed}")
            if m.get("phase") != pilot_tvr.PHASE[tier]:
                probs.append(f"phase {m.get('phase')} != {pilot_tvr.PHASE[tier]}")
            if not m.get("optimizer_steps") or m.get("optimizer_steps") != m.get("optimizer_steps_planned"):
                probs.append(f"optimizer_steps {m.get('optimizer_steps')} of {m.get('optimizer_steps_planned')}")
            if tman is not None and (want_sha is None or m.get("data_sha256") != want_sha):
                probs.append(f"data_sha256 {str(m.get('data_sha256'))[:12]} != manifest {fname} {str(want_sha)[:12]}")
            if probs:
                bad.append(f"{s}/{tier}: " + "; ".join(probs))
    items.append(_item("adapter_provenance", bad, miss))

    # 14 floors file
    bad, miss = [], []
    fp = L.floors / "floors_eval500.json"
    fl = read_json(fp)
    if fl is None:
        miss.append(str(fp))
    else:
        if sha256_file(fp) != pilot_cfg.get("data.floors_eval500_sha256", cfg):
            bad.append(f"floors_eval500.json sha {str(sha256_file(fp))[:12]} != data.floors_eval500_sha256")
        if fl.get("warnings") != []:
            bad.append(f"floors warnings {fl.get('warnings')}")
        if (fl.get("F") or {}).get("rate") != pilot_cfg.get("data.F_eval500", cfg):
            bad.append(f"F {(fl.get('F') or {}).get('rate')} != data.F_eval500 {pilot_cfg.get('data.F_eval500', cfg)}")
    items.append(_item("floors_file", bad, miss))

    # 15 render consistency
    bad, miss, rpf = [], [], {}
    if sv_path is None:
        miss.append("smoke verdict")
    else:
        p = Path(sv_path).parent / "replay2" / "render_parity.json"
        r = read_json(p)
        rpf["smoke"] = None if r is None else {k: r.get(k) for k in ("min_psnr_db", "min_ssim", "n_frames")}
        if r is None:
            miss.append(str(p))
        elif r.get("pass") is not True:
            bad.append(f"{p}: pass {r.get('pass')}")
    for s in starts:
        # the canary of the newest run of each eval job (an earlier one is moved aside to <part>.prev-<UTC> by the
        # payload before it renders again, so only canary/top and canary/low are read)
        cdir = root / s / "canary"
        files = []
        for part in ("top", "low"):
            p = cdir / part / "render_parity.json"
            if p.is_file():
                files.append(p)
            else:
                miss.append(f"{s}/canary/{part}/render_parity.json")
        for p in files:
            r = read_json(p)
            rpf[f"{s}/{p.parent.name}"] = None if r is None else {k: r.get(k) for k in ("min_psnr_db", "min_ssim",
                                                                                        "n_frames")}
            if r is None or r.get("pass") is not True:
                bad.append(f"{p}: pass {None if r is None else r.get('pass')}")
    recorded["render_parity"] = rpf
    items.append(_item("render_consistency", bad, miss))

    # 16 demos replay
    bad, miss = [], []
    n_pub = pilot_tvr.PUBLIC_N
    if rp is None:
        miss.append(f"{L.replay}/results.json")
    elif rp.get("valid") is not True or rp.get("n_tasks") != n_pub or rp.get("tasks_missing"):
        bad.append(f"replay valid {rp.get('valid')}, n_tasks {rp.get('n_tasks')}, missing "
                   f"{(rp.get('tasks_missing') or [])[:3]}")
    dropped = read_json(L.demos / "dropped.json")
    if man is None or dropped is None:
        miss.append("demos manifest.json / dropped.json")
    else:
        kept, nd = man.get("kept"), man.get("dropped")
        if not isinstance(kept, int) or kept < int(pilot_cfg.get("gates.min_kept", cfg)):
            bad.append(f"kept {kept} < {pilot_cfg.get('gates.min_kept', cfg)}")
        if not isinstance(dropped, list) or len(dropped) != nd or (isinstance(kept, int) and isinstance(nd, int)
                                                                   and kept + nd != n_pub):
            bad.append(f"dropped list {len(dropped) if isinstance(dropped, list) else dropped!r} vs manifest "
                       f"dropped {nd}, kept {kept} (want kept + dropped = {n_pub})")
        else:
            badr = [d for d in dropped if str(d.get("reason", "")).split("@")[0] not in pilot_tvr.DROP_REASONS]
            if badr:
                bad.append(f"{len(badr)} drops without a G10(a) reason, e.g. {badr[:2]}")
    items.append(_item("demos_replay", bad, miss, None if man is None else {"kept": man.get("kept"),
                                                                            "dropped": man.get("dropped")}))

    # 17 independent recompute (17a above) + floors
    floors_equal, fdetail = None, None
    if fl is None:
        miss17.append(str(fp))
    else:
        try:
            mine = recheck_tvr.floors(**recheck_args)
            probs = recheck_tvr.compare_floors(mine, fl)
            floors_equal = not probs
            fdetail = {"F": mine["F"], "problems": probs}
            bad17 += [f"floors recheck: {p}" for p in probs]
        except (OSError, ValueError, KeyError) as err:
            miss17.append(f"recheck_tvr floors: {type(err).__name__}: {err}"[:300])
    items.append(_item("independent_recompute", bad17, miss17))
    # counts_equal from the recount entries only (a missing floors file says nothing about the counts)
    count_bad = [b for b in bad17 if not b.startswith("floors recheck:")]
    count_miss = [m for m in miss17 if m != str(fp) and not m.startswith("recheck_tvr floors:")]
    recheck = {"counts_equal": False if count_bad else (None if count_miss else True),
               "floors_equal": floors_equal, "detail": fdetail}
    assert [i["name"] for i in items] == list(G0_ITEMS), [i["name"] for i in items]
    return items, recorded, recheck


def power_stale(pw: dict, cfg, expected_inputs: List[str]) -> List[str]:
    """Why an existing POWER.json does not describe the pilot as it is now ([] = current): every per_task.jsonl it
    read still has the recorded sha256, it read every per_task.jsonl A4 needs, and it ran under this pilot_cfg."""
    inp = pw.get("inputs") or {}
    files = inp.get("files_sha256")
    if not isinstance(files, dict) or not files:
        return ["no inputs.files_sha256"]
    out = [f"{p}: sha256 changed" for p, want in sorted(files.items()) if sha256_file(p) != want]
    out += [f"{p}: not among its inputs" for p in expected_inputs if Path(p).is_file() and p not in files]
    # power_sim records sha256(pilot_cfg.as_json(cfg)) as inputs.pilot_cfg_json_sha256 (the cfg it ran under)
    want_cfg = hashlib.sha256(pilot_cfg.as_json(cfg).encode()).hexdigest()
    got_cfg = inp.get("pilot_cfg_json_sha256")
    if got_cfg != want_cfg:
        out.append(f"pilot_cfg json sha256 {str(got_cfg)[:12]} != {want_cfg[:12]}")
    return out


def _js(x) -> str:
    return json.dumps(x, sort_keys=True)


def power_unregistered(pw: dict, cfg) -> List[str]:
    """Why a POWER.json was not made at the registered settings ([] = it was): power_sim's registered block is
    present and records the registered test, n_sims (registered and used; the CLI's --n-sims is for tests only),
    tier_corr, fallback, persistence and backend of pilot_cfg power.*, no allow_no_replicate, and the seed
    replicate was read (sigma.replicate_missing and persistence.replicate_missing False)."""
    p = _plain(pilot_cfg.get("power", cfg))
    reg = pw.get("registered")
    if not isinstance(reg, dict):
        return ["no registered block"]
    out = []
    if reg.get("test") != p["test"]:
        out.append(f"registered.test {reg.get('test')!r} != power.test {p['test']!r}")
    for k in ("n_sims_registered", "n_sims_used"):
        if _js(reg.get(k)) != _js(p["n_sims"]):
            out.append(f"registered.{k} {reg.get(k)!r} != power.n_sims {p['n_sims']}")
    if reg.get("allow_no_replicate") is not False:
        out.append(f"registered.allow_no_replicate {reg.get('allow_no_replicate')!r} (want false)")
    for blk in ("sigma", "persistence"):
        if (pw.get(blk) or {}).get("replicate_missing") is not False:
            out.append(f"{blk}.replicate_missing {(pw.get(blk) or {}).get('replicate_missing')!r} (want false)")
    for k in ("tier_corr", "fallback", "persistence"):
        if _js(reg.get(k)) != _js(p.get(k)):
            out.append(f"registered.{k} != power.{k}")
    # the backend this run used (PREREG 12.6: seed and n_sims pin the result only together with the backend)
    if reg.get("backend") != p.get("backend"):
        out.append(f"registered.backend {reg.get('backend')!r} != power.backend {p.get('backend')!r}")
    return out


def a4_rule(cfg) -> str:
    p = pilot_cfg.get("power", cfg)
    tc = p["tier_corr"]
    return (f"after G0-G3 pass: power_sim ({p['test']}, Holm): power >= {p['target_power']} at "
            f"{100 * float(p['primary_delta']):g} pp for both contrasts at tier_corr {tc['power']}, sigma_used, "
            f"lambda_used; family-wise type-I <= {p['type1_max']} at tier_corr "
            f"{' and '.join(str(r) for r in tc['type1'])} (the max)")


def a4_summary(pw: dict) -> dict:
    """The main design's verdict as GATE.json records it (no per-contrast numbers)."""
    v = (pw.get("verdicts") or {}).get("main") or {}
    return {"pass": pw.get("pass"), "fallback_pass": pw.get("fallback_pass"), "type1": v.get("type1"),
            "type1_ok": v.get("type1_ok"), "power_ok": v.get("power_ok"), "power_tier_corr": v.get("power_tier_corr"),
            "sigma_used": (pw.get("sigma") or {}).get("sigma_used"),
            "lambda_used": (pw.get("persistence") or {}).get("lambda_used"),
            "n_sims_used": (pw.get("registered") or {}).get("n_sims_used")}


# --- evaluate ---------------------------------------------------------------------------------------------------

def evaluate(pilot_root: Path = None, starts=STARTS, L: pilot_tvr.Layout = None, cfg=None,
             smoke_verdict_path: Optional[Path] = None, power: bool = False, recheck_args: Optional[dict] = None
             ) -> dict:
    L = L or pilot_tvr.LAYOUT
    root = Path(pilot_root or L.pilot_root)
    z = float(pilot_cfg.get("gates.z", cfg))
    top = pilot_cfg.n_top(cfg)
    if smoke_verdict_path is None:
        smoke_verdict_path = pilot_tvr.latest_verdict(L)[0]
    sv = read_json(smoke_verdict_path) if smoke_verdict_path else None
    pub = str(pilot_cfg.get("public.sft_train_json", cfg)).replace("OUT/", str(L.root) + "/", 1)
    rargs = {"reach_dir": L.reach, "tasks_dir": paths.TVR_REPO / "data" / "tasks", "public_path": Path(pub),
             "task_set": "eval500"}
    rargs.update(recheck_args or {})
    evals = {s: {} for s in starts}
    for s in starts:
        for tier, arm in eval_list(s, cfg):
            e = Eval(root, s, tier, arm)
            evals[s][e.key] = e
    fl = read_json(L.floors / "floors_eval500.json") or {}
    F = (fl.get("F") or {}).get("rate")
    items, recorded, recheck = g0_items(root, evals, starts, L, cfg, sv, smoke_verdict_path, rargs)
    g0 = _tri(i["ok"] for i in items)

    # per-start absolute numbers (recounted from trajectories)
    out_starts = {}
    counts = {}
    for s in starts:
        rows = {}
        for key, e in evals[s].items():
            if not e.present:
                rows[key] = {"present": False}
                continue
            rc = recheck_tvr.counts(e.dir)
            counts[(s, key)] = rc
            rows[key] = {"present": True, "valid": e.valid, "n_tasks": e.r("n_tasks"),
                         "sr": block(rc["k"], rc["n"], z),
                         "by_category": {c: block(v["k"], v["n"], z) for c, v in rc["by_category"].items()},
                         "spl": e.r("spl"), "spl_official": e.r("spl_official"), "within1": e.r("within1"),
                         "stop": e.r("stop"), "invalid": e.r("invalid"), "steps": e.r("steps"),
                         "errors_final": e.r("errors_final"), "bootstrap_scene": e.r("bootstrap_scene")}
        out_starts[s] = {"evals": rows}

    # G1
    per, oks = {}, []
    for s in starts:
        t, z0 = counts.get((s, f"{top}__c")), counts.get((s, "0W__c"))
        if t is None or z0 is None or F is None or t["n"] != pilot_tvr.EVAL500_N or z0["n"] != pilot_tvr.EVAL500_N:
            per[s] = {"top": top, "k": None if t is None else t["k"], "n": None if t is None else t["n"], "lb": None,
                      "F": F, "zero_w_rate": None if z0 is None else z0["rate"], "threshold": None, "pass": None}
            oks.append(None)
            continue
        lb = wilson_lb(t["k"], t["n"], z)
        thr = max(float(F), z0["k"] / z0["n"])
        ok = lb > thr
        per[s] = {"top": top, "k": t["k"], "n": t["n"], "lb": round(lb, 4), "F": F, "zero_w_rate": z0["k"] / z0["n"],
                  "threshold": thr, "pass": ok}
        oks.append(ok)
    g1 = "pass" if any(o is True for o in oks) else ("fail" if oks and all(o is False for o in oks) else "incomplete")
    G1 = {"ok": {"pass": True, "fail": False}.get(g1), "status": g1,
          "rule": "some start: wilson_lb(k_TOP_c, 500) > max(F, SR_0W_c of that start)", "per_start": per}

    # G2
    checks = []
    for s in starts:
        for key, e in evals[s].items():
            if e.arm not in ("c", "blind"):
                continue
            if not e.present:
                checks.append({"name": f"{s}/{key}", "ok": None, "detail": "missing"})
                continue
            inv = e.r("invalid", default={})
            rc = counts[(s, key)]
            ok = (inv.get("regex_mismatch") == 0 and (inv.get("replies") or 0) > 0 and not e.r("tasks_missing", default=[])
                  and rc["n"] == pilot_tvr.EVAL500_N)
            checks.append({"name": f"{s}/{key}", "ok": bool(ok),
                           "detail": {"regex_mismatch": inv.get("regex_mismatch"), "replies": inv.get("replies"),
                                      "trajectories": rc["n"]}})
    g2 = _tri(c["ok"] for c in checks)
    G2 = {"ok": {"pass": True, "fail": False}.get(g2), "status": g2,
          "rule": "every c and blind eval: 100% regex fullmatch, 0 tasks without a log", "checks": checks}

    # G3
    sat = fractions.Fraction(str(pilot_cfg.get("gates.saturation", cfg)))
    checks = []
    for s in starts:
        t = counts.get((s, f"{top}__c"))
        if t is None or t["n"] != pilot_tvr.EVAL500_N:
            checks.append({"name": s, "ok": None, "detail": None if t is None else {"k": t["k"], "n": t["n"]}})
        else:
            checks.append({"name": s, "ok": fractions.Fraction(t["k"], t["n"]) <= sat,
                           "detail": {"k": t["k"], "n": t["n"], "rate": t["rate"], "max": float(sat)}})
    g3 = _tri(c["ok"] for c in checks)
    G3 = {"ok": {"pass": True, "fail": False}.get(g3), "status": g3,
          "rule": f"every start's TOP c rate <= {float(sat)}", "checks": checks}

    # A3 (report only): images vs no images, same adapter, same tasks
    a3 = {}
    alpha3 = float(pilot_cfg.get("gates.mcnemar_alpha", cfg))
    for s in starts:
        ec, eb = evals[s].get(f"{top}__c"), evals[s].get(f"{top}__blind")
        if not (ec and eb and ec.present and eb.present):
            a3[s] = {"b": None, "c": None, "p": None, "images_better_p_lt_0.05": None}
            continue
        sc, sb = counts[(s, f"{top}__c")]["success"], counts[(s, f"{top}__blind")]["success"]
        common = sorted(set(sc) & set(sb))
        b = sum(1 for t in common if sc[t] and not sb[t])
        c = sum(1 for t in common if sb[t] and not sc[t])
        p = binom_two_sided(b, c)
        a3[s] = {"b": b, "c": c, "n_paired": len(common), "p": p,
                 "images_better_p_lt_0.05": bool(b > c and p < alpha3)}
    A3 = {"report_only": True, "rule": "exact two-sided binomial on the discordant pairs (TOP c vs TOP blind); "
          f"images better: b > c and p < {alpha3} (gates.mcnemar_alpha)", "per_start": a3}

    # A4
    pf = root / "POWER.json"
    a4_note = None
    power_inputs = [str(Eval(root, s, tier, "c").dir / "per_task.jsonl") for s in starts for tier in ("100", top)]
    rep_ = pilot_tvr.replicate(cfg)
    if rep_["start"] in starts:
        power_inputs.append(str(Eval(root, rep_["start"], f"{rep_['tier']}s{rep_['seed']}", "c").dir /
                                "per_task.jsonl"))
    if g0 == g1 == g2 == g3 == "pass":
        pw = None
        if power:
            from vpd.common import power_sim

            try:
                pw = power_sim.run_tvr(root, cfg=pilot_cfg.FROZEN if cfg is None else cfg, out=pf)
            except (OSError, ValueError, KeyError) as err:
                a4_note = f"power_sim failed: {type(err).__name__}: {err}"[:300]
        else:
            pw = read_json(pf)
            stale = power_stale(pw, cfg, power_inputs) if pw is not None else []
            if stale:
                a4_note = "stale POWER.json: " + "; ".join(stale[:3])
                pw = None
        unreg = power_unregistered(pw, cfg) if pw is not None else []
        if unreg:
            a4_note = "POWER.json not made at the registered settings: " + "; ".join(unreg[:3])
            pw = None
        a4 = "pending" if pw is None else ("pass" if pw.get("pass") is True else "fail")
    else:
        pw = None
        a4 = "pending"
    A4 = {"status": a4, "power_file": str(pf), "note": a4_note, "rule": a4_rule(cfg),
          "power": None if pw is None else a4_summary(pw)}

    if g0 == "fail" or g2 == "fail":
        decision = "engineering"
    elif g0 == "incomplete":
        decision = "incomplete"
    elif g1 == "fail" or g3 == "fail":
        decision = "not_a_ruler"
    elif "incomplete" in (g1, g2, g3):
        decision = "incomplete"
    elif a4 == "pass":
        decision = "ruler"
    elif a4 == "fail":
        decision = "power_insufficient"
    else:
        decision = "incomplete"
    ps = pilot_tvr.prereg_state()
    cfgp = paths.VPD_PKG / "tvr" / "pilot_cfg.py"
    return {"schema": SCHEMA, "utc": pilot_tvr.utc(), "pilot_root": str(root), "prereg_commit": ps.get("commit"),
            "pilot_cfg_sha256": sha256_file(cfgp), "z": z, "top": top,
            "floors": {"F": fl.get("F"), "file": str(L.floors / "floors_eval500.json"),
                       "components": {"stop0": (fl.get("stop0") or {}).get("rate"),
                                      "fixed": ((fl.get("fixed") or {}).get("eval") or {}).get("rate"),
                                      "random": (fl.get("random") or {}).get("mean_rate")}},
            "smoke_verdict": None if smoke_verdict_path is None else str(smoke_verdict_path),
            "starts": out_starts, "G0": {"ok": {"pass": True, "fail": False}.get(g0), "status": g0, "items": items},
            "G1": G1, "G2": G2, "G3": G3, "recheck": recheck, "A3": A3, "A4": A4, "recorded": recorded,
            "decision": decision, "note": NOTE}


# --- GATE.md ----------------------------------------------------------------------------------------------------

DECISION_ZH = {"ruler": "TVR 是合格的尺子（G0-G3 过，A4 过）：按预注册进确认轮",
               "not_a_ruler": "TVR 当不了尺子（G1 或 G3 不过）：不进确认轮，写进总报告",
               "engineering": "工程问题（G0 或 G2 不过）：修好重跑受影响的步骤，不判科学结论",
               "power_insufficient": "G0-G3 过，但功效模拟不够（A4 不过）：不按现方案进确认轮，交用户",
               "incomplete": "不完整：缺评测或文件（或 A4 还没算），补齐后再判"}


def _pct(b: Optional[dict]) -> str:
    if not b or not b.get("n"):
        return "—"
    lo, hi = b["wilson95"]
    return f"{b['k']}/{b['n']} = {100 * b['rate']:.1f}%（{100 * lo:.1f}–{100 * hi:.1f}）"


def _st(x) -> str:
    return {"pass": "过", "fail": "不过", "incomplete": "不完整", "pending": "待算"}.get(x, str(x))


def render_md(rep: dict) -> str:
    top = rep["top"]
    L = ["# TVR 试点闸门结果（GATE）", "",
         f"- 结论：**{DECISION_ZH.get(rep['decision'], rep['decision'])}**（decision = {rep['decision']}）",
         f"- G0 {_st(rep['G0']['status'])}，G1 {_st(rep['G1']['status'])}，G2 {_st(rep['G2']['status'])}，"
         f"G3 {_st(rep['G3']['status'])}，A3 只报告，A4 {_st(rep['A4']['status'])}（规则见 PREREG_TVR_PILOT.md 第 13 节）",
         f"- 生成时间：{rep['utc']}（UTC）；Wilson z = {rep['z']}；最高档 TOP = {top}；预注册提交 {rep['prereg_commit']}。",
         "- 每个起点只列绝对数字（k/n 和 95% Wilson 区间），不算起点之间的差。", "",
         "## 盲走下限（floors_eval500.json）", "", "| 策略 | 率 |", "|---|---|"]
    comp = rep["floors"]["components"]
    for name, zh in (("stop0", "开局即停"), ("fixed", "固定序列"), ("random", "随机有效动作（3 种子均值）")):
        v = comp.get(name)
        L.append(f"| {zh} | {'—' if v is None else f'{100 * v:.2f}%'} |")
    F = rep["floors"]["F"] or {}
    frate = "—" if F.get("rate") is None else f"{100 * F['rate']:.2f}%"
    L.append(f"| **F**（{F.get('source')}） | {frate} |")
    L += ["", "## G1 / G3：最高档约束解码（500 题）", "",
          "| 起点 | TOP c 成功（95% CI） | Wilson 下界 | 门槛 max(F, 0W) | G1 | ≤ 饱和线？ |", "|---|---|---|---|---|---|"]
    g3 = {c["name"]: c for c in rep["G3"]["checks"]}
    for s, r in rep["G1"]["per_start"].items():
        ev = rep["starts"][s]["evals"].get(f"{top}__c") or {}
        thr = "—" if r["threshold"] is None else f"{100 * r['threshold']:.2f}%"
        lb = "—" if r["lb"] is None else f"{100 * r['lb']:.2f}%"
        g = {True: "过", False: "不过", None: "?"}[r["pass"]]
        ok3 = {True: "是", False: "否", None: "?"}[g3.get(s, {}).get("ok")]
        L.append(f"| {s} | {_pct(ev.get('sr'))} | {lb} | {thr} | {g} | {ok3} |")
    L += ["", "## 每次评测（绝对数字）", "",
          "| 起点 | 档 | 臂 | 有效 | 成功（95% CI） | SR-easy | SR-hard | LR-easy | LR-hard | SPL | 一步之内 | 停止率 | 停错/停止 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, st in rep["starts"].items():
        for key, ev in st["evals"].items():
            tier, arm = key.split("__")
            if not ev.get("present"):
                L.append(f"| {s} | {tier} | {arm} | 缺 | | | | | | | | | |")
                continue
            bc = ev.get("by_category") or {}
            cats = " | ".join(f"{(bc.get(c) or {}).get('k')}/{(bc.get(c) or {}).get('n')}" for c in pilot_cfg.CATEGORIES)
            w1 = ev.get("within1") or {}
            stp = ev.get("stop") or {}
            spl = ev.get("spl")
            L.append(f"| {s} | {tier} | {arm} | {'是' if ev.get('valid') else '否'} | {_pct(ev.get('sr'))} | {cats} | "
                     f"{'—' if spl is None else f'{spl:.3f}'} | {w1.get('k')}/{w1.get('n')} | "
                     f"{stp.get('stop_rate')} | {stp.get('fstop_given_stop')} |")
    L += ["", "## A3（只报告）：同一 TOP 适配器给图 vs 不给图，同一批 500 题", "",
          "| 起点 | 只有给图成功 b | 只有不给图成功 c | 精确双侧 p | 给图更好且 p < 0.05 |", "|---|---|---|---|---|"]
    for s, a in rep["A3"]["per_start"].items():
        p = a.get("p")
        ptxt = "—" if p is None else f"{p:.4g}"
        better = {True: "是", False: "否", None: "?"}[a.get("images_better_p_lt_0.05")]
        L.append(f"| {s} | {a.get('b')} | {a.get('c')} | {ptxt} | {better} |")
    L += ["", "## G0 机制（17 项）", "", "| # | 项 | 结果 | 问题 / 缺 |", "|---|---|---|---|"]
    for i, it in enumerate(rep["G0"]["items"], 1):
        res = {True: "过", False: "不过", None: "不完整"}[it["ok"]]
        d = it["detail"]
        txt = "；".join((d.get("problems") or [])[:3] + [f"缺 {m}" for m in (d.get("missing") or [])[:3]])
        L.append(f"| {i} | {it['name']} | {res} | {txt[:300]} |")
    L += ["", "## G2 格式", ""]
    badg2 = [c for c in rep["G2"]["checks"] if c["ok"] is not True]
    L.append("全部约束解码和不给图的评测：正则 100% 匹配，无缺日志的题。" if not badg2 else
             "；".join(f"{c['name']} {'不过' if c['ok'] is False else '缺'}" for c in badg2))
    a4, pw = rep["A4"], rep["A4"].get("power") or {}
    extra = "" if not pw else (f"；族错误率 {pw.get('type1')}，σ_used {pw.get('sigma_used')}，"
                               f"λ_used {pw.get('lambda_used')}，备用设计 fallback_pass {pw.get('fallback_pass')}")
    L += ["", f"A4 功效：{_st(a4['status'])}（{a4['power_file']}）{extra}"
          f"{'；' + str(a4['note']) if a4.get('note') else ''}。", "",
          "说明：试点数据（种子 1001）不进确认分析，只用于功效决定；不约束解码（u 臂）只作描述。", ""]
    return "\n".join(L)


def write_json(p: Path, obj) -> None:
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def main(argv=None) -> int:
    # login node (R5): one thread per numeric library before power_sim may import numpy; run under nice -n 19
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-root", type=Path, default=None, help="default OUT/tvr/runs/pilot")
    ap.add_argument("--starts", default=",".join(STARTS))
    ap.add_argument("--smoke-verdict", type=Path, default=None, help="default: the newest smoke_verdict.json")
    ap.add_argument("--power", action="store_true", help="run power_sim when G0-G3 pass (else read POWER.json)")
    args = ap.parse_args(argv)
    starts = tuple(s for s in args.starts.split(",") if s)
    rep = evaluate(args.pilot_root, starts, smoke_verdict_path=args.smoke_verdict, power=args.power)
    out = Path(rep["pilot_root"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "GATE.json", rep)
    tmp = out / f"GATE.md.tmp{os.getpid()}"
    tmp.write_text(render_md(rep), encoding="utf-8")
    os.replace(tmp, out / "GATE.md")
    print(json.dumps({"G0": rep["G0"]["status"], "G1": rep["G1"]["status"], "G2": rep["G2"]["status"],
                      "G3": rep["G3"]["status"], "A4": rep["A4"]["status"], "decision": rep["decision"]}))
    if rep["decision"] == "ruler":
        return 0
    if rep["decision"] in ("engineering", "not_a_ruler", "power_insufficient"):
        return 1
    return 3


if __name__ == "__main__":
    sys.exit(main())
