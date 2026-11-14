"""E3VS SFT data: the frozen single-frame prompt, BFS demo decision records, the format warm-up and tier orders.

Prompt (PREREG_E3VS_PILOT section 8; shared with vpd/e3vs/run.py through ``Prompts``).  The official
Qwen3-VL-8B-Instruct single_frame configuration (e3vs/configs/qwen_3.0_vl_8B_instruct/single_frame.yaml, merged
with agents/base_agent.py DEFAULT_CONFIG and worker.py as the official runner does) rendered with the official
jinja templates of the pinned repo (FileSystemLoader, trim_blocks, lstrip_blocks, as prompts/navigation_builder.py):
  - system = navigation_system.j2 with the observation description, mention_step0_target_centered true and
    action_instruction.j2 (use_cot false, action_chunk 0, restrict_collision: the "AVAILABLE ACTIONS (for this
    step)" list = all 11 actions, 0.25 m / 30.0 deg, the camera-mode movement text);
  - only change: the output-format tail ``action: move_forward`` (format line and example) is replaced by the
    neutral format of PREREG section 7 (OFFICIAL_TAIL -> NEUTRAL_TAIL, asserted to hit exactly once);
  - user = input_observation_instruction.j2 (question, ``Current step: t``, position with 2 decimals, last action
    and the collision line), followed by the current image; no history.  The three template files must have the
    pinned sha256s (TEMPLATE_SHA256).
Reply: ``THOUGHT: \\nACTION: <name>`` (THOUGHT empty), regex REGEX.  Images: lossless PNG at 512x512.

Records (n2_train_vpd sharegpt): {"id", "messages": [system, user "<text><image>", assistant reply], "images":
[png path], "meta": {...}}: one decision per record (single frame), loss on the one reply.

Subcommands (outputs under OUT/e3vs/sft unless --out):
  warmup  20 records (PREREG 9.2): one episode per train scene, the scene's smallest md5("<scene>:<episode>"),
          scenes in that md5 order, first 20; image = the start view; record i takes ACTIONS[i mod 11] (official
          order); question = "Format practice, not a viewpoint-search task: take the action <name>."; step 0.
          -> warmup20.jsonl, warmup/images/, warmup_manifest.json
  order   order_seed<S>.json from the collision-aware train demo pool (PREREG 9.3): one episode per (scene,
          question) (smallest md5), groups by question type sorted by md5 then shuffled with one
          random.Random(S) in type-name order, then a deficit interleave (pick the type maximising
          share * (placed + 1) - placed_type, ties by type name), so every prefix is stratified and tiers nest.
  tier    seed<S>/demos_<N>.jsonl: the first N episodes of the order, all their decision records (incl. stop);
          images under images/train/<episode>/t<step>.png (shared by all tiers).  N = ALL (round 4,
          OUT/plan/ROUND4.md B): every reached pool record, i.e. the order's episodes, then the other starts of
          the same (scene, question) in a stratified_order of their own (same seed) -> demos_ALL.jsonl.
  all     warmup + order + tier for --seed / --tiers.
Rendering: --renderer gsplat (one PLY at a time) or fake (tests).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import (ACTIONS, ANGLE_DEG, IMAGE_H, IMAGE_W, LOGIN_TMP_REPO, STEP_M, check_repo, code_meta,
                      load_episodes, repo_dir, sha256_file)

# ------------------------------------------------------------------------------------------------------------
# frozen prompt
# ------------------------------------------------------------------------------------------------------------
TEMPLATE_DIR_REL = "e3vs/prompts/templates"
TEMPLATE_SHA256 = {  # pinned repo bb7e603c, checked 2026-11-08 on the plan/login_tmp copy
    "navigation_system.j2": "8d683c7bf21ac203a24b1b1d25d89d8fcf8e5b0867d7c90d252791f198a6f634",
    "action_instruction.j2": "b23d88f8610e909ed1b9293e5a4e70da1f240044a9e4cec26a56bb0ca50f9f08",
    "input_observation_instruction.j2": "439fdbaefeb7ec6dcb6a28b49722cab1fa8f6a3dac166c87c7c3fb81c3790e7d",
}
TEMPLATES_PY_SHA256 = "ae69f2fe955ede06f5930418a15205a3727610a1ee87392e1ed791a0de3fdd8e"  # e3vs/prompts/templates.py
# e3vs/prompts/templates.py CAMERA_MOVEMENT_DESC (tests compare it with the repo file)
CAMERA_MOVEMENT_DESC = (
    "- **Forward/Backward Movement (`move_forward`, `move_backward`):** "
    "These actions move you along the camera's view direction (camera coordinate axis), "
    "which means you will move up or down if the camera is tilted. "
)
# _ObservationDescriptionBuilder with include_translation_info true, rotation false, no past observations
OBSERVATION_DESC = "You will be given your current position, the current observation image only."
# single_frame.yaml (Qwen3-VL-8B-Instruct) + DEFAULT_CONFIG + worker.py's step sizes
OFFICIAL_CONFIG = {
    "use_cot": False, "action_chunk": 0, "uses_think_tags": False, "combined_thinking": False,
    "thinking_process": ["SceneUnderstanding", "TargetIdentification", "CommonsenseReasoning", "SpatialPlanning",
                         "PerceptualForecasting"],
    "available_actions": list(ACTIONS), "action_setting": "restrict_collision",
    "include_translation_info": True, "include_rotation_info": False,
    "num_past_observations": 0, "num_past_actions": 0, "num_past_reasoning": 0,
    "mention_step0_target_centered": True, "action_step_size": STEP_M, "rotation_step_size": ANGLE_DEG,
    "max_output_tokens": 128, "temperature": 0.0,
}
OFFICIAL_TAIL = ("**OUTPUT FORMAT:**\nOutput exactly one action in this format:\naction: move_forward\n\n"
                 "Example:action: move_forward")
NEUTRAL_TAIL = ("**OUTPUT FORMAT:**\nOutput exactly one action in this format:\n"
                "THOUGHT: <your reasoning; may be empty>\nACTION: <action name>\n\n"
                "Example:\nTHOUGHT: \nACTION: move_forward")
ACTION_ALT = ("move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|look_up|"
              "look_down|stop")
REGEX = r"THOUGHT: \nACTION: (" + ACTION_ALT + ")"  # PREREG section 7, verbatim
assert sorted(ACTION_ALT.split("|")) == sorted(ACTIONS)
IMAGE = "<image>"
WARMUP_QUESTION = "Format practice, not a viewpoint-search task: take the action {name}."
N_WARMUP = 20
DEFAULT_SEED = 1001
DEFAULT_TIERS = (100,)
ALL_TIER = "ALL"  # every reached pool record (tier_records; round 4)


def reply(action: str) -> str:
    """The demo / warm-up reply: empty THOUGHT, one action."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    return f"THOUGHT: \nACTION: {action}"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Prompts:
    """The frozen prompt (module docstring), rendered from the pinned repo's jinja templates."""

    def __init__(self, repo: Path | None = None):
        import jinja2

        self.repo = Path(repo) if repo else repo_dir()
        tdir = self.repo / TEMPLATE_DIR_REL
        got = {f: sha256_file(tdir / f) for f in TEMPLATE_SHA256}
        bad = {f: g for f, g in got.items() if g != TEMPLATE_SHA256[f]}
        if bad:
            raise AssertionError(f"E3VS prompt templates differ from the pinned commit: {bad}")
        self.template_sha256 = got
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(tdir)), trim_blocks=True, lstrip_blocks=True)
        cfg = OFFICIAL_CONFIG
        ai = env.get_template("action_instruction.j2").render({
            "use_cot": cfg["use_cot"], "action_chunk": cfg["action_chunk"], "uses_think_tags": cfg["uses_think_tags"],
            "action_step_size": cfg["action_step_size"], "angle_step_size": cfg["rotation_step_size"],
            "available_actions": cfg["available_actions"], "possible_actions": list(cfg["available_actions"]),
            "enabled_steps": cfg["thinking_process"], "all_steps_details": {},
            "thinking_process": cfg["thinking_process"], "action_setting": cfg["action_setting"],
            "movement_description": CAMERA_MOVEMENT_DESC, "combined_thinking": cfg["combined_thinking"]})
        official = env.get_template("navigation_system.j2").render(
            observation_description=OBSERVATION_DESC, action_instruction=ai,
            mention_step0_target_centered=cfg["mention_step0_target_centered"])
        if official.count(OFFICIAL_TAIL) != 1 or not official.endswith(OFFICIAL_TAIL):
            raise AssertionError("the official system prompt does not end with the expected output-format tail")
        self.official_system = official
        self._system = official[: -len(OFFICIAL_TAIL)] + NEUTRAL_TAIL
        self._user_t = env.get_template("input_observation_instruction.j2")

    def system(self) -> str:
        return self._system

    def user(self, question: str, step: int, position, last_action: str | None = None,
             collision: bool = False) -> str:
        """The official user text of one decision (followed by the image)."""
        pos = [float(x) for x in np.asarray(position, dtype=np.float64).reshape(-1)[:3]]
        hist = [] if last_action is None else [{"action": last_action, "collision": bool(collision)}]
        cfg = OFFICIAL_CONFIG
        text = self._user_t.render({
            "question": question, "pos": pos, "rot": [0.0, 0.0, 0.0],
            "include_translation_info": cfg["include_translation_info"],
            "include_rotation_info": cfg["include_rotation_info"], "history": hist,
            "num_past_reasoning": cfg["num_past_reasoning"], "num_past_actions": cfg["num_past_actions"],
            "step": int(step), "target_pixel_coords": None, "action_setting": cfg["action_setting"],
            "last_action": last_action, "collision_occurred": bool(collision), "action_to_alphabet": {}})
        if IMAGE in text:
            raise ValueError("the user text contains the image placeholder")
        return text

    def meta(self) -> dict:
        return {"template_sha256": self.template_sha256, "system_sha256": sha256_text(self._system),
                "official_system_sha256": sha256_text(self.official_system), "official_tail": OFFICIAL_TAIL,
                "neutral_tail": NEUTRAL_TAIL, "regex": REGEX, "config": OFFICIAL_CONFIG,
                "repo": str(self.repo), "image": f"PNG {IMAGE_W}x{IMAGE_H}, after the user text"}


def sft_record(rid: str, system: str, user_text: str, image: Path | str, action: str, meta: dict) -> dict:
    return {"id": rid, "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user_text + IMAGE},
                                    {"role": "assistant", "content": reply(action)}],
            "images": [str(image)], "meta": meta}


# ------------------------------------------------------------------------------------------------------------
# selections
# ------------------------------------------------------------------------------------------------------------
def md5key(scene_id: str, episode_index: int) -> str:
    return hashlib.md5(f"{scene_id}:{int(episode_index)}".encode()).hexdigest()


def warmup_selection(train_episodes: list[dict], n: int = N_WARMUP) -> list[dict]:
    """One episode per train scene (smallest md5), scenes ordered by that md5, the first n."""
    best = {}
    for ep in train_episodes:
        k = md5key(ep["scene_id"], ep["episode_index"])
        if ep["scene_id"] not in best or k < best[ep["scene_id"]][0]:
            best[ep["scene_id"]] = (k, ep)
    chosen = [ep for _, ep in sorted(best.values(), key=lambda x: x[0])][:n]
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} train scenes for a warm-up of {n}")
    return chosen


def dedup_pool(pool: list[dict]) -> list[dict]:
    """One record per (scene_id, question): the smallest md5(scene:episode); only reached, non-extra starts."""
    best = {}
    for r in pool:
        if r.get("status") != "reached" or r.get("extra"):
            continue
        key = (r["scene_id"], r["question"])
        k = md5key(r["scene_id"], r["episode_index"])
        if key not in best or k < best[key][0]:
            best[key] = (k, r)
    return [r for _, r in sorted(best.values(), key=lambda x: x[0])]


def stratified_order(records: list[dict], seed: int = DEFAULT_SEED) -> list[dict]:
    """PREREG 9.3 steps 2-3 on deduplicated records (see module docstring)."""
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["question_type"]].append(r)
    rng = random.Random(seed)
    for qt in sorted(groups):
        groups[qt].sort(key=lambda r: md5key(r["scene_id"], r["episode_index"]))
        rng.shuffle(groups[qt])
    total = sum(len(g) for g in groups.values())
    share = {qt: len(g) / total for qt, g in groups.items()}
    placed = collections.Counter()
    out = []
    while len(out) < total:
        n = len(out)
        qt, best = None, None
        for q in sorted(groups):  # ties -> the first type name
            if placed[q] >= len(groups[q]):
                continue
            v = share[q] * (n + 1) - placed[q]
            if best is None or v > best + 1e-12:
                qt, best = q, v
        out.append(groups[qt][placed[qt]])
        placed[qt] += 1
    return out


def decision_steps(rec: dict) -> list[dict]:
    """[{step, pose, last_action, collision, action}] of one BFS record (actions include the final stop)."""
    acts, poses = rec["actions"], rec["poses"]
    if not acts or acts[-1] != "stop" or len(poses) != len(acts) + 1:
        raise ValueError(f"episode {rec.get('episode_index')}: malformed BFS record")
    return [{"step": t, "pose": np.asarray(poses[t], dtype=np.float64), "last_action": acts[t - 1] if t else None,
             "collision": False, "action": a} for t, a in enumerate(acts)]


# ------------------------------------------------------------------------------------------------------------
# rendering
# ------------------------------------------------------------------------------------------------------------
class SceneRenderers:
    """One renderer at a time (scenes are processed in order); kind gsplat | fake | a callable scene -> renderer."""

    def __init__(self, kind="gsplat", device: str = "cuda"):
        self.kind, self.device = kind, device
        self.scene, self.r = None, None
        self.meta = {}

    def get(self, scene_id: str):
        if scene_id != self.scene:
            from vpd.e3vs import render as RD

            self.r = None
            self.r = self.kind(scene_id) if callable(self.kind) else RD.make_renderer(self.kind, scene_id,
                                                                                      self.device)
            self.scene = scene_id
            m = dict(getattr(self.r, "meta", {}))
            if getattr(self.r, "name", "") == "gsplat":
                m["ply_sha256"] = RD.ply_sha256(scene_id)
            self.meta[scene_id] = m
        return self.r


def render_to(renderers: SceneRenderers, scene_id: str, c2w, path: Path, overwrite: bool = False) -> str:
    from vpd.e3vs import render as RD

    path = Path(path)
    if path.exists() and not overwrite:
        return sha256_file(path)
    return RD.write_png(renderers.get(scene_id).render_rgb(c2w, IMAGE_W, IMAGE_H), path)


def _write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return sha256_file(path)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _code() -> dict:
    from vpd.e3vs import render as RD

    return code_meta(Path(__file__), Path(RD.__file__), Path(sys.modules["vpd.e3vs"].__file__))


def build_warmup(out: Path, prompts: Prompts, renderers: SceneRenderers, repo: Path, overwrite: bool = False) -> dict:
    eps = warmup_selection(load_episodes("train", repo))
    system = prompts.system()
    rows = []
    for i, ep in enumerate(sorted(eps, key=lambda e: md5key(e["scene_id"], e["episode_index"]))):
        name = ACTIONS[i % len(ACTIONS)]
        img = out / "warmup" / "images" / f"w{i:02d}_{ep['episode_index']:05d}.png"
        sha = render_to(renderers, ep["scene_id"], ep["start_c2w"], img, overwrite)
        text = prompts.user(WARMUP_QUESTION.format(name=name), 0, ep["start_c2w"][:3, 3])
        rows.append(sft_record(f"e3vs_warmup_{i:02d}", system, text, img, name, {
            "kind": "warmup", "i": i, "episode_index": ep["episode_index"], "scene_id": ep["scene_id"],
            "action": name, "image_sha256": sha}))
    sha = _write_jsonl(out / "warmup20.jsonl", rows)
    man = {"kind": "warmup", "n": len(rows), "jsonl": str(out / "warmup20.jsonl"), "jsonl_sha256": sha,
           "rule": "PREREG_E3VS_PILOT 9.2", "episodes": [r["meta"]["episode_index"] for r in rows],
           "actions": [r["meta"]["action"] for r in rows], "prompt": prompts.meta(), "renderers": renderers.meta,
           "code": _code(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json(out / "warmup_manifest.json", man)
    return man


def read_pool(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def build_order(pool_path: Path, seed: int, out: Path) -> dict:
    pool = read_pool(pool_path)
    ded = dedup_pool(pool)
    order = stratified_order(ded, seed)
    shares = collections.Counter(r["question_type"] for r in ded)
    res = {"seed": seed, "pool": str(pool_path), "pool_sha256": sha256_file(pool_path), "n_pool_records": len(pool),
           "n_reached": sum(r.get("status") == "reached" and not r.get("extra") for r in pool), "n_dedup": len(ded),
           "type_counts": dict(sorted(shares.items())), "rule": "PREREG_E3VS_PILOT 9.3",
           "order": [{"episode_index": r["episode_index"], "scene_id": r["scene_id"],
                      "question_type": r["question_type"], "opt_len": r["opt_len"]} for r in order],
           "code": _code(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json(out / f"order_seed{seed}.json", res)
    return res


def tier_records(order: dict, pool: list[dict], n, seed: int) -> list[dict]:
    """The episodes of tier n in tier order: the first n of the order; n == ALL_TIER: every reached, non-extra
    pool record, the order's episodes first, then the other starts in stratified_order(rest, seed)."""
    by_ep = {r["episode_index"]: r for r in pool if r.get("status") == "reached" and not r.get("extra")}
    if n == ALL_TIER:
        head = [by_ep[o["episode_index"]] for o in order["order"]]
        seen = {r["episode_index"] for r in head}
        return head + stratified_order([r for r in by_ep.values() if r["episode_index"] not in seen], seed)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"tier must be a positive integer or {ALL_TIER!r}, got {n!r}")
    if n > len(order["order"]):
        raise ValueError(f"tier {n} > {len(order['order'])} episodes in the order")
    return [by_ep[o["episode_index"]] for o in order["order"][:n]]


def parse_tiers(spec: str) -> list:
    """'100' -> [100]; '300,ALL' -> [300, 'ALL']."""
    return [ALL_TIER if x == ALL_TIER else int(x) for x in re.split(r"[,\s]+", spec) if x]


def build_tier(out: Path, seed: int, n, prompts: Prompts, renderers: SceneRenderers,
               pool_path: Path, overwrite: bool = False) -> dict:
    """seed<S>/demos_<n>.jsonl + manifest; n is a positive integer or ALL_TIER."""
    order = json.loads((out / f"order_seed{seed}.json").read_text())
    if order["pool_sha256"] != sha256_file(pool_path):
        raise AssertionError(f"{pool_path} changed since order_seed{seed}.json was built")
    chosen = tier_records(order, read_pool(pool_path), n, seed)
    in_order = {o["episode_index"] for o in order["order"]}
    system = prompts.system()
    rows = []
    for rec in sorted(chosen, key=lambda r: (r["scene_id"], r["episode_index"])):  # one PLY load per scene
        for d in decision_steps(rec):
            img = out / "images" / "train" / f"{rec['episode_index']:05d}" / f"t{d['step']:02d}.png"
            sha = render_to(renderers, rec["scene_id"], d["pose"], img, overwrite)
            text = prompts.user(rec["question"], d["step"], d["pose"][:3, 3], d["last_action"], d["collision"])
            rows.append(sft_record(f"e3vs_train_{rec['episode_index']:05d}_t{d['step']:02d}", system, text, img,
                                   d["action"], {"kind": "demo", "episode_index": rec["episode_index"],
                                                 "scene_id": rec["scene_id"], "question_type": rec["question_type"],
                                                 "step": d["step"], "action": d["action"], "image_sha256": sha}))
    rank = {r["episode_index"]: i for i, r in enumerate(chosen)}  # tier order (a numeric tier: the order's)
    rows.sort(key=lambda r: (rank[r["meta"]["episode_index"]], r["meta"]["step"]))
    path = out / f"seed{seed}" / f"demos_{n}.jsonl"
    sha = _write_jsonl(path, rows)
    man = {"kind": "tier", "seed": seed, "tier": n, "n_episodes": len(chosen), "n_records": len(rows),
           "n_from_order": sum(r["episode_index"] in in_order for r in chosen),
           "jsonl": str(path), "jsonl_sha256": sha, "order": str(out / f"order_seed{seed}.json"),
           "pool": str(pool_path), "pool_sha256": order["pool_sha256"],
           "type_counts": dict(sorted(collections.Counter(r["question_type"] for r in chosen).items())),
           "actions": dict(sorted(collections.Counter(r["meta"]["action"] for r in rows).items())),
           "prompt": prompts.meta(), "renderers": renderers.meta, "code": _code(),
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_json(out / f"seed{seed}" / f"demos_{n}_manifest.json", man)
    return man


def default_pool() -> Path:
    return paths.E3VS_BFS / "mesh" / "train" / "demo_pool.jsonl"  # the prep job's --tag mesh (mesh_collision.TAG)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("warmup", "order", "tier", "all"))
    ap.add_argument("--out", type=Path, default=None, help="default OUT/e3vs/sft")
    ap.add_argument("--pool", type=Path, default=None, help="default OUT/e3vs/bfs/mesh/train/demo_pool.jsonl")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tiers", default=",".join(map(str, DEFAULT_TIERS)), help="comma list, e.g. 100 or 300,ALL")
    ap.add_argument("--renderer", default="gsplat", choices=("gsplat", "fake"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true", help="re-render images that exist")
    a = ap.parse_args(argv)
    repo = Path(a.repo) if a.repo else repo_dir()
    if a.out is None and repo.resolve() == LOGIN_TMP_REPO.resolve():
        raise SystemExit("the repo is the read-only login_tmp copy (provenance pending the user's OK): pass --out "
                         "under the scratch dir instead of writing OUT/e3vs/sft")
    out = Path(a.out) if a.out else paths.E3VS_SFT
    pool = Path(a.pool) if a.pool else default_pool()
    check_repo(repo)
    prompts = Prompts(repo)
    renderers = SceneRenderers(a.renderer, a.device)
    tiers = parse_tiers(a.tiers)
    if a.cmd in ("warmup", "all"):
        m = build_warmup(out, prompts, renderers, repo, a.overwrite)
        print(f"[make_sft] warmup: {m['n']} records -> {m['jsonl']}", flush=True)
    if a.cmd in ("order", "all"):
        m = build_order(pool, a.seed, out)
        print(f"[make_sft] order seed {a.seed}: {m['n_dedup']} episodes (pool {m['n_reached']} reached) "
              f"types {m['type_counts']}", flush=True)
    if a.cmd in ("tier", "all"):
        for n in tiers:
            m = build_tier(out, a.seed, n, prompts, renderers, pool, a.overwrite)
            print(f"[make_sft] tier {n}: {m['n_episodes']} episodes, {m['n_records']} records -> {m['jsonl']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
