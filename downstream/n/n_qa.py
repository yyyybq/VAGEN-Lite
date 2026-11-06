#!/usr/bin/env python3
"""N series, "can it answer, can it do" (the user's task book of 2026-11-05, item 7): two-choice questions on the
start image of N1's merged 600 instances, asked zero-shot to the eight models, then paired with their N1 episodes.

    python downstream/n/n_qa.py truth   --instances I.jsonl --scene SID --out DIR        # simulator Python, per scene
    python downstream/n/n_qa.py paths   --instances I.jsonl --scene SID --out DIR        # simulator Python, per scene
    python downstream/n/n_qa.py collect --truth DIR --out TRUTH.jsonl
    python downstream/n/n_qa.py make    --instances I.jsonl --gen GEN [GEN ...] [--truth TRUTH.jsonl]
                                        [--paths PATHS.jsonl] [--object-image start|shortest|best] [--kinds K ...]
                                        --out DIR
    python downstream/n/n_qa.py samples --questions Q.jsonl --n 20 --out PAGE.html
    python downstream/n/n_qa.py analyze --questions Q.jsonl --answers M=A.jsonl [...] --episodes EVAL [EVAL ...]
                                        --instances I.jsonl --out DIR

truth: every instance of the scene, from its start pose, each of the six actions for one step in N1's environment
  (n_env.Scene, N1's robot body; the robot hidden as in the episodes): blocked or not (n_env.Scene.try_action) and the
  shaping score after it (n_env.box_judge, the score the episodes record as max_shaping_score; a blocked action leaves
  the pose and the score where they were).  DIR/<SID>.jsonl.
paths (11-05, after the first truth: at 423 of the 600 starts no single step changes the shaping score): the search
  the instances were generated with (n_smoke.reachable's: poses reachable from the start in at most MAX_DEPTH
  actions, MAX_STATES poses, a move blocked when its new pose collides), keeping every edge; then per instance, with
  the box judge as the goal, the fewest actions to finish after each first action (a blocked first action wastes one
  step).  DIR/<SID>.jsonl.
make: kinds of question, in X0's two-choice layout (a prompt naming the camera, two lettered options, "Answer
  with the letter of the correct option only."), the right letter drawn at random (seeded):
    left_right   "is the <A> to the left or to the right of the <B>?": both objects of the instance visible at the
                 start (segmentation share >= n_env.MIN_VISIBLE_FRACTION and a visible-pixel centre in the stored
                 marks), their centres at least LR_PIXELS apart across the 256-px image (X0's rule);
    near_far     "which is closer to the camera?": both visible as above; the horizontal distance from the camera to
                 each object's box footprint (the v2 judges' distance) differs by at least NF_METERS and by a factor
                 of NF_RATIO (X0's thresholds; X0 itself used the median depth of the visible pixels, which N1's
                 stored renders do not keep);
    best_action  "which single action brings the robot closer to finishing the task?" (the instance's own task
                 text): the action whose one step raises the shaping score the most against one drawn at random from
                 those at least DELTA below it; only where the best one raises the start score by at least DELTA;
    first_action "which action should the robot take first to finish the task in the fewest actions?" (paths): an
                 action that starts a shortest way (the recorded shortest path's first one when it is such) against
                 one that makes the way at least 2 actions longer or reaches no success within the search.
  The names are the instance's object names (n_env.natural_name); an instance whose two names are the same is left
  out of the two object kinds.  --object-image asks the object kinds on another view of the instance (shortest or
  best: its success images, where both objects are visible by construction) instead of the start; the action kinds
  are always asked on the start image.  DIR/questions.jsonl, DIR/summary.json.
samples: N questions spread over the kinds and task types, as a review page: the image (with the two objects marked
  for the object kinds), the prompt, the right answer and the numbers behind it.
analyze: two readings of a reply: strict (x0_qa.parse_letter: the option letter alone) and lenient (also a reply that
  is one option's text, or names exactly one option's text).  Per model, kind and reading: accuracy; the N1 episodes'
  success rate and mean max_shaping_score on the instances answered right against those answered wrong, also within
  the shortest path's length (2-3 / 4-5 / 6+ actions) and near-threshold or not; across models, on the instances where
  A answers right and B wrong, how often A succeeds and B fails against how often B succeeds and A fails (exact
  binomial test, as McNemar's).  DIR/report.json, DIR/report.md.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "x"))
import n_env as N  # noqa: E402

LR_PIXELS = 30.0  # X0's left/right margin on a 256-px image
NF_METERS, NF_RATIO = 0.2, 1.2  # X0's near/far margins
DELTA = 0.02  # best_action: the least rise over the start, and the least gap to the other option
KINDS = ("left_right", "near_far", "best_action", "first_action")
ACTION_TEXT = {"move_forward": "move forward", "move_backward": "move backward", "move_left": "move left",
               "move_right": "move right", "turn_left": "turn left", "turn_right": "turn right"}
CAMERA_TEXT = "This image is from the robot's camera."
ANSWER_TEXT = "Answer with the letter of the correct option only."
LENGTH_BANDS = (("2-3", 2, 3), ("4-5", 4, 5), ("6+", 6, 99))


def read_jsonl(path) -> list:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def write_jsonl(path, rows) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ------------------------------------------------------------------------------------------------- truth
def shaping(judge: dict) -> float:
    """The episodes' shaping score of a box judge (n_episode.shaping)."""
    m = judge["metric"]
    return float(m.get("shaping_score", m.get("score")) or 0.0)


def one_step(scene, inst: dict) -> dict:
    """From the instance's start pose, each action for one step: blocked or not and the shaping score after it."""
    start = tuple(inst["start_pose"])
    base = shaping(N.box_judge(inst["item"], start))
    out = {"instance_id": inst["instance_id"], "start_shaping": base, "actions": {}}
    for a in N.ACTIONS:
        scene.set_pose(start)
        new, hit = scene.try_action(start, a)
        out["actions"][a] = {"blocked": hit is not None, "by": scene.owner(hit) if hit is not None else None,
                             "shaping": base if hit is not None else shaping(N.box_judge(inst["item"], new))}
    scene.set_pose(start)
    return out


def truth_cmd(args) -> int:
    insts = [r for r in read_jsonl(args.instances) if r["scene_id"] == args.scene]
    if not insts:
        print(f"[n_qa] no instance of {args.scene}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for hide in sorted({bool(i.get("hide_robot", False)) for i in insts}):  # n_episode's default and grouping
        scene = N.Scene(insts[0]["task"], int(insts[0]["seed"]), hide_robot=hide)
        try:
            for inst in insts:
                if bool(inst.get("hide_robot", False)) == hide:
                    rows[inst["instance_id"]] = one_step(scene, inst)
        finally:
            scene.close()
    rows = [rows[i["instance_id"]] for i in insts]
    write_jsonl(args.out / f"{args.scene}.jsonl", rows)
    print(f"[n_qa] {args.scene}: {len(rows)} instances")
    return 0


def collect_cmd(args) -> int:
    rows = [r for p in sorted(Path(args.truth).glob("*.jsonl")) for r in read_jsonl(p)]
    write_jsonl(args.out, rows)
    print(f"[n_qa] {len(rows)} instances from {len(list(Path(args.truth).glob('*.jsonl')))} scenes")
    return 0


# ------------------------------------------------------------------------------------------------- paths
MAX_DEPTH, MAX_STATES = 12, 15000  # n_smoke.reachable's, which the instances were generated with


def search(scene, start, max_depth: int = MAX_DEPTH, max_states: int = MAX_STATES) -> tuple:
    """Breadth first over the poses reachable from start, as n_smoke.reachable (pose_key states, a move blocked
    when its new pose collides, at most max_depth actions and max_states poses), also keeping every edge:
    ({key: (pose, depth)}, {key: {action: key, or None when blocked}}, capped)."""
    k0 = N.pose_key(start)
    states, edges, blocked = {k0: (tuple(start), 0)}, {}, set()
    queue, capped = collections.deque([k0]), False
    while queue:
        k = queue.popleft()
        pose, depth = states[k]
        if depth >= max_depth:
            continue
        out = {}
        for a in N.ACTIONS:
            new = tuple(N.apply_action(pose, a))
            nk = N.pose_key(new)
            if nk in blocked:
                out[a] = None
            elif nk in states:
                out[a] = nk
            else:
                scene.set_pose(new)
                if scene.collision() is not None:
                    blocked.add(nk)
                    out[a] = None
                else:
                    states[nk] = (new, depth + 1)
                    out[a] = nk
                    queue.append(nk)
                    if len(states) >= max_states:
                        capped = True
                        break
        edges[k] = out
        if capped:
            break
    scene.set_pose(start)
    return states, edges, capped


def to_goal(states: dict, edges: dict, goal) -> dict:
    """{key: fewest actions from it to any goal key} over the kept edges (keys with no path left out)."""
    rev = collections.defaultdict(list)
    for k, out in edges.items():
        for nk in out.values():
            if nk is not None:
                rev[nk].append(k)
    dist = {k: 0 for k in goal}
    queue = collections.deque(goal)
    while queue:
        k = queue.popleft()
        for p in rev[k]:
            if p not in dist:
                dist[p] = dist[k] + 1
                queue.append(p)
    return dist


def exact_depth(states: dict, edges: dict, capped: bool, max_depth: int = MAX_DEPTH) -> int:
    """Costs up to this many actions are exact (the true fewest), larger ones and None only bounds: a first action's
    way of k + 1 actions needs the edges of poses at most k actions from the start.  Breadth first expands every pose
    of a depth before the next, so a capped search has all poses shallower than the one it stopped in expanded;
    an uncapped one all poses shallower than max_depth."""
    return max((states[k][1] for k in edges), default=0) if capped else max_depth


def first_actions(states: dict, edges: dict, capped: bool, inst: dict, success, max_depth: int = MAX_DEPTH) -> dict:
    """Per action from the start: the fewest actions to finish the instance when it is taken first (a blocked one
    leaves the robot where it is: 1 + the start's own distance); None if no pose passing success(pose) is
    reachable through it within the search.  exact_up_to: exact_depth (costs above it, and None, mean at least
    exact_up_to + 1)."""
    k0 = N.pose_key(inst["start_pose"])
    goal = [k for k, (pose, _) in states.items() if success(pose)]
    dist = to_goal(states, edges, goal)
    out = {}
    for a, nk in edges.get(k0, {}).items():
        if nk is None:
            out[a] = None if k0 not in dist else 1 + dist[k0]
        else:
            out[a] = None if nk not in dist else 1 + dist[nk]
    return {"instance_id": inst["instance_id"], "cost": out, "start_dist": dist.get(k0),
            "blocked": [a for a, nk in edges.get(k0, {}).items() if nk is None], "goal_poses": len(goal),
            "poses": len(states), "capped": capped, "exact_up_to": exact_depth(states, edges, capped, max_depth)}


def paths_cmd(args) -> int:
    """Per scene: the search from the start, then per instance first_actions with the box judge as the goal."""
    insts = [r for r in read_jsonl(args.instances) if r["scene_id"] == args.scene]
    if not insts:
        print(f"[n_qa] no instance of {args.scene}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    groups = collections.defaultdict(list)  # the robot setting and start as n_episode builds them
    for i in insts:
        groups[(bool(i.get("hide_robot", False)), N.pose_key(i["start_pose"]))].append(i)
    for (hide, _), group in sorted(groups.items()):
        scene = N.Scene(group[0]["task"], int(group[0]["seed"]), hide_robot=hide)
        try:
            states, edges, capped = search(scene, tuple(group[0]["start_pose"]), args.max_depth, args.max_states)
        finally:
            scene.close()
        for inst in group:
            rows[inst["instance_id"]] = first_actions(states, edges, capped, inst,
                                                      lambda pose, item=inst["item"]: N.box_judge(item, pose)["success"],
                                                      args.max_depth)
    write_jsonl(args.out / f"{args.scene}.jsonl", [rows[i["instance_id"]] for i in insts])
    print(f"[n_qa] {args.scene}: {len(rows)} instances")
    return 0


# -------------------------------------------------------------------------------------------------- make
def footprint_distance(pose, obj: dict) -> float:
    """Horizontal distance from the camera (over the pose's x, y) to the object's box footprint; 0 inside it."""
    x, y = pose[0], pose[1]
    lo, hi = obj["bbox_min"], obj["bbox_max"]
    dx = max(lo[0] - x, 0.0, x - hi[0])
    dy = max(lo[1] - y, 0.0, y - hi[1])
    return math.hypot(dx, dy)


def view_of(inst: dict, tag: str) -> tuple:
    """(pose, judge numbers, marks) of an instance's start or of its shortest / best success pose."""
    if tag == "start":
        return inst["start_pose"], inst.get("start_judge") or {}, (inst.get("marks") or {}).get("start") or {}
    return inst[tag]["pose"], inst[tag].get("judge") or {}, (inst.get("marks") or {}).get(tag) or {}


def visible_pair(inst: dict, tag: str = "start"):
    """The two objects if both are visible in the view (share and stored centre) and named apart, else None."""
    objs = inst.get("objects") or []
    if len(objs) != 2:
        return None
    _, judge, marks = view_of(inst, tag)
    vis = judge.get("visible") or {}
    if "a" not in marks or "b" not in marks:
        return None
    if any(vis.get(o["key"], 0.0) < N.MIN_VISIBLE_FRACTION for o in objs):
        return None
    names = [N.natural_name(o["name"]) for o in objs]
    if names[0] == names[1]:
        return None
    return objs, names, (marks["a"], marks["b"])


def two_choice(stem: str, options: list, right: int, rng: random.Random) -> tuple:
    """(prompt, options in their drawn order, the right letter)."""
    order = [0, 1] if rng.random() < 0.5 else [1, 0]
    shown = [options[i] for i in order]
    letter = "AB"[order.index(right)]
    prompt = f"{stem}\nA. {shown[0]}\nB. {shown[1]}\n{ANSWER_TEXT}"
    return prompt, shown, letter


def object_questions(inst: dict, common: dict, tag: str, rng: random.Random, skipped) -> list:
    """left_right and near_far on the two objects in one view (tag: start, shortest or best)."""
    pair = visible_pair(inst, tag)
    if pair is None:
        skipped["object kinds: not two visible, differently named objects"] += 1
        return []
    out, iid = [], inst["instance_id"]
    objs, names, (ma, mb) = pair
    dx = ma[0] - mb[0]
    if abs(dx) >= LR_PIXELS:
        subj = rng.randrange(2)
        other = 1 - subj
        left = (dx < 0) if subj == 0 else (dx > 0)  # the subject's centre left of the other's
        stem = f"{CAMERA_TEXT} In this image, is the {names[subj]} to the left or to the right of the {names[other]}?"
        prompt, shown, letter = two_choice(stem, ["left", "right"], 0 if left else 1, rng)
        out.append(dict(common, id=f"{iid}::left_right", kind="left_right", prompt=prompt, options=shown,
                        answer=letter, numbers={"dx_px": round(dx, 1), "subject": names[subj], "other": names[other]}))
    else:
        skipped["left_right: centres closer than 30 px"] += 1
    pose = view_of(inst, tag)[0]
    da, db = (footprint_distance(pose, o) for o in objs)
    near = 0 if da < db else 1
    dn, dfar = min(da, db), max(da, db)
    if dfar - dn >= NF_METERS and dfar >= NF_RATIO * max(dn, 1e-9):
        stem = f"{CAMERA_TEXT} Which of these objects is closer to the camera?"
        prompt, shown, letter = two_choice(stem, [f"the {names[0]}", f"the {names[1]}"], near, rng)
        out.append(dict(common, id=f"{iid}::near_far", kind="near_far", prompt=prompt, options=shown,
                        answer=letter, numbers={"distance_m": {names[0]: round(da, 3), names[1]: round(db, 3)}}))
    else:
        skipped["near_far: distances too close"] += 1
    return out


def first_action_question(inst: dict, common: dict, p: dict, rng: random.Random, skipped):
    """first_action: an action that starts a shortest way to a box success (search's costs) against one that makes
    the way at least 2 actions longer, or is blocked at the start (the user's task book of 2026-11-06: "干扰项取让最短
    走法至少多 2 步或会被挡的动作").  The search may be capped (p["exact_up_to"], exact_depth): a cost above it, or
    None, only says the action needs at least exact_up_to + 1, so it is a distractor only when that bound is d + 2 or
    more, and the question is skipped when the shortest way itself is longer than exact_up_to + 1 (then it may be
    shorter).  A free distractor (not blocked) is taken when there is one, else a blocked one (blocked: 1 + d,
    exact)."""
    cost = p["cost"]
    blocked = set(p.get("blocked") or [])
    exact = p.get("exact_up_to")
    if exact is None:  # rows written before exact_up_to: exact only when the search was not capped
        exact = MAX_DEPTH if not p.get("capped") else 0
    known = {a: c for a, c in cost.items() if c is not None}
    if not known:
        skipped["first_action: no success reachable"] += 1
        return None
    d = min(known.values())
    optimal = [a for a in N.ACTIONS if cost.get(a) == d and a not in blocked]
    if not optimal:
        skipped["first_action: no success reachable"] += 1
        return None
    if d > exact + 1:
        skipped["first_action: shortest way beyond the search's exact depth"] += 1
        return None

    def at_least(c):  # the fewest actions the action truly needs, as far as the search can tell
        return c if c is not None and c <= exact else exact + 1

    free = [a for a in N.ACTIONS if a in cost and a not in optimal and a not in blocked and at_least(cost[a]) >= d + 2]
    stuck = [a for a in N.ACTIONS if a in blocked]
    if not free and not stuck:
        skipped["first_action: no action clearly worse"] += 1
        return None
    recorded = ((inst.get("shortest") or {}).get("path") or [None])[0]
    best = recorded if recorded in optimal else optimal[0]
    other = rng.choice(free or stuck)
    stem = (f"{CAMERA_TEXT} The robot's task: {inst['description']}\n"
            "Which action should the robot take first to finish the task in the fewest actions?")
    prompt, shown, letter = two_choice(stem, [ACTION_TEXT[best], ACTION_TEXT[other]], 0, rng)
    return dict(common, id=f"{inst['instance_id']}::first_action", kind="first_action", prompt=prompt, options=shown,
                answer=letter, numbers={"cost": cost, "fewest": d, "optimal": optimal, "best": best, "other": other,
                                        "other_kind": "blocked" if other in blocked else "longer",
                                        "other_at_least": d + 1 if other in blocked else at_least(cost[other]),
                                        "exact_up_to": exact, "blocked": p.get("blocked"), "recorded_first": recorded,
                                        "recorded_actions": (inst.get("shortest") or {}).get("actions")})


def make_questions(insts: list, images: dict, truth: dict, seed: int = 20261105, delta: float = DELTA,
                   paths=None, object_image: str = "start", object_images=None, kinds=KINDS) -> tuple:
    """images: start image per instance; object_images: the object_image view's (default the start's).  paths:
    paths_cmd's rows per instance (for first_action)."""
    rng = random.Random(seed)
    qs, skipped = [], collections.Counter()
    object_images = images if object_images is None else object_images
    for inst in insts:
        iid = inst["instance_id"]
        start = {"instance_id": iid, "task_type": inst["task_type"], "image_path": images.get(iid), "image_tag": "start"}
        view = dict(start, image_path=object_images.get(iid), image_tag=object_image)
        if {"left_right", "near_far"} & set(kinds):
            if view["image_path"]:
                qs += [q for q in object_questions(inst, view, object_image, rng, skipped) if q["kind"] in kinds]
            else:
                skipped[f"object kinds: no {object_image} image"] += 1
        if not start["image_path"]:
            skipped["action kinds: no start image"] += 1
            continue
        if "best_action" in kinds:
            t = truth.get(iid)
            if t is None:
                skipped["best_action: no truth"] += 1
            else:
                scores = {a: v["shaping"] for a, v in t["actions"].items()}
                best = max(scores, key=lambda a: (scores[a], -N.ACTIONS.index(a)))
                lower = [a for a in N.ACTIONS if scores[a] <= scores[best] - delta]
                if scores[best] < t["start_shaping"] + delta or not lower:
                    skipped["best_action: no action rises enough or none clearly worse"] += 1
                else:
                    other = rng.choice(lower)
                    stem = (f"{CAMERA_TEXT} The robot's task: {inst['description']}\n"
                            "Which single action brings the robot closer to finishing the task?")
                    prompt, shown, letter = two_choice(stem, [ACTION_TEXT[best], ACTION_TEXT[other]], 0, rng)
                    qs.append(dict(start, id=f"{iid}::best_action", kind="best_action", prompt=prompt, options=shown,
                                   answer=letter, numbers={"start": round(t["start_shaping"], 4),
                                                           "after": {a: round(s, 4) for a, s in scores.items()},
                                                           "blocked": [a for a, v in t["actions"].items() if v["blocked"]],
                                                           "best": best, "other": other}))
        if "first_action" in kinds and paths is not None:
            p = paths.get(iid)
            if p is None:
                skipped["first_action: no search"] += 1
            else:
                q = first_action_question(inst, start, p, rng, skipped)
                if q:
                    qs.append(q)
    return qs, dict(skipped)


def image_paths(insts: list, gens, tag: str = "start") -> dict:
    """instance id -> absolute path of its image of one view (start, shortest, best), from whichever generation
    folder holds it."""
    out = {}
    for inst in insts:
        rel = (inst.get("images") or {}).get(tag) or f"images/{inst['instance_id']}_{tag}.png"
        for g in gens:
            p = Path(g) / rel
            if p.is_file():
                out[inst["instance_id"]] = str(p)
                break
    return out


def make_cmd(args) -> int:
    insts = read_jsonl(args.instances)
    truth = {r["instance_id"]: r for r in read_jsonl(args.truth)} if args.truth else {}
    paths = {r["instance_id"]: r for r in read_jsonl(args.paths)} if args.paths else None
    kinds = tuple(args.kinds or KINDS)
    qs, skipped = make_questions(insts, image_paths(insts, args.gen), truth, args.seed, args.delta, paths=paths,
                                 object_image=args.object_image,
                                 object_images=image_paths(insts, args.gen, args.object_image), kinds=kinds)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "questions.jsonl", qs)
    summary = {"instances": len(insts), "questions": len(qs),
               "per_kind": dict(collections.Counter(q["kind"] for q in qs)),
               "right_letter_A": {k: sum(q["answer"] == "A" for q in qs if q["kind"] == k) for k in kinds},
               "skipped": skipped, "seed": args.seed, "delta": args.delta, "kinds": list(kinds),
               "object_image": args.object_image}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


# ----------------------------------------------------------------------------------------------- samples
def pick_samples(qs: list, n: int, seed: int = 0) -> list:
    """n questions taken in turn from the kinds, each kind's own taken in turn from its task types (shuffled)."""
    rng = random.Random(seed)
    pools = {k: collections.defaultdict(list) for k in KINDS}
    for q in qs:
        pools[q["kind"]][q["task_type"]].append(q)
    queues = {}
    for k in KINDS:
        for v in pools[k].values():
            rng.shuffle(v)
        types, queue = sorted(pools[k]), []
        while any(pools[k][t] for t in types):
            queue += [pools[k][t].pop() for t in types if pools[k][t]]
        queues[k] = queue
    out = []
    while len(out) < n and any(queues.values()):
        for k in KINDS:
            if queues[k] and len(out) < n:
                out.append(queues[k].pop(0))
    return out


def pick_by_kind(qs: list, counts: dict, seed: int = 0) -> list:
    """counts {kind: n}: n questions of each kind (taken in turn from its task types), kinds in KINDS order."""
    out = []
    for k in KINDS:
        if counts.get(k):
            out += pick_samples([q for q in qs if q["kind"] == k], int(counts[k]), seed)
    return out


def marked_image(q: dict, inst: dict) -> str:
    import base64
    import io

    from PIL import Image, ImageDraw

    img = Image.open(q["image_path"]).convert("RGB").resize((512, 512))
    d = ImageDraw.Draw(img)
    if q["kind"] in ("left_right", "near_far"):
        marks = inst["marks"][q.get("image_tag", "start")]
        for letter, obj in zip("ab", inst["objects"]):
            if letter in marks:
                x, y = marks[letter][0] * 2, marks[letter][1] * 2
                d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(220, 40, 40), width=4)
                d.text((x + 11, y - 8), N.natural_name(obj["name"]), fill=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_note(q: dict) -> str:
    """One line in Chinese under a first_action sample: the shortest way and why the distractor is worse."""
    if q["kind"] != "first_action":
        return ""
    n = q["numbers"]
    why = ("起点处被挡（原地不动，至少 " if n.get("other_kind") == "blocked" else "走它至少要 ") + f"{n.get('other_at_least')} 步）"
    return (f"<p>搜索：最短 {n['fewest']} 步（搜索的步数在 {n.get('exact_up_to')} 步以内都准确）；"
            f"最短走法的第一步可以是 {html.escape('、'.join(n['optimal']))}；"
            f"干扰项 {html.escape(n['other'])}：{why}。</p>")


def samples_cmd(args) -> int:
    qs = read_jsonl(args.questions)
    insts = {r["instance_id"]: r for r in read_jsonl(args.instances)}
    picked = (pick_by_kind(qs, dict(kv.split("=") for kv in args.counts), args.seed) if args.counts
              else pick_samples(qs, args.n, args.seed))
    cards = []
    view_zh = {"start": "起始画面", "shortest": "最短走法终点的画面", "best": "最好成功位姿的画面"}
    for k, q in enumerate(picked, 1):
        cards.append(
            f"<section><h2>{k}. {html.escape(q['kind'])} · {html.escape(q['task_type'])} · "
            f"{html.escape(q['instance_id'])}</h2><p>画面：{view_zh.get(q.get('image_tag', 'start'), q.get('image_tag'))}</p>"
            f"<img src='{marked_image(q, insts[q['instance_id']])}' width='512'>"
            f"<pre>{html.escape(q['prompt'])}</pre><p><b>正确答案：{q['answer']}</b>（{html.escape(q['options']['AB'.index(q['answer'])])}）</p>"
            f"{sample_note(q)}<pre>{html.escape(json.dumps(q['numbers'], ensure_ascii=False, indent=1))}</pre></section>")
    page = ("<!doctype html><meta charset='utf-8'><title>N 系列问答样例</title>"
            "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}section{border-bottom:1px solid #ccc;"
            "padding:12px 0}pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}</style>"
            f"<h1>N 系列问答样例（{len(picked)} 道）</h1><p>红圈：两个物体在该题所用画面上的可见像素中心（模型看到的图里没有圈）。"
            "第一步题的图是起始画面，不画圈。</p>"
            + "".join(cards))
    Path(args.out).write_text(page, encoding="utf-8")
    write_jsonl(Path(args.out).with_suffix(".jsonl"), picked)
    print(f"[n_qa] {len(picked)} samples -> {args.out}")
    return 0


# ----------------------------------------------------------------------------------------------- analyze
def parse_strict(reply: str):
    import x0_qa

    return x0_qa.parse_letter(reply)


def parse_lenient(reply: str, options: list):
    letter = parse_strict(reply)
    if letter:
        return letter
    t = re.sub(r"<[^>]*>", " ", str(reply or "")).strip().lower().rstrip(".").strip()
    for i, o in enumerate(options):
        if t == str(o).lower():
            return "AB"[i]
    hits = [i for i, o in enumerate(options) if str(o).lower() in t]
    return "AB"[hits[0]] if len(hits) == 1 else None


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


def band_of(actions: int) -> str:
    for name, a, b in LENGTH_BANDS:
        if a <= actions <= b:
            return name
    return "other"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def episodes_of(dirs, model: str) -> dict:
    out = {}
    for d in dirs:
        for r in read_jsonl(Path(d) / model / "episodes.jsonl"):
            if not r.get("error"):
                out[r["instance_id"]] = r
    return out


def FIRST_ACTION_GROUPS(num: dict) -> tuple:
    """first_action's extra strata (11-06): 445 of the first 600 right answers were move_backward and most
    distractors blocked, so answered-right against answered-wrong is also split by the right action and by the kind of
    distractor (num: instance id -> the question's numbers)."""
    return (("answer move_backward", lambda i: num[i].get("best") == "move_backward"),
            ("answer other", lambda i: num[i].get("best") != "move_backward"),
            ("distractor blocked", lambda i: num[i].get("other_kind") == "blocked"),
            ("distractor longer", lambda i: num[i].get("other_kind") == "longer"))


def analyze(qs: list, answers: dict, episodes: dict, insts: dict) -> dict:
    """answers: {model: {question id: reply}}; episodes: {model: {instance id: row}}."""
    rep = {"models": {}, "pairs": {}}
    right = {}
    for m, ans in answers.items():
        rep["models"][m] = {}
        for reading in ("strict", "lenient"):
            for kind in KINDS:
                kq = [q for q in qs if q["kind"] == kind and q["id"] in ans]
                got = {q["id"]: (parse_strict(ans[q["id"]]) if reading == "strict"
                                 else parse_lenient(ans[q["id"]], q["options"])) for q in kq}
                ok = {q["instance_id"]: got[q["id"]] == q["answer"] for q in kq}
                right[(m, reading, kind)] = ok
                ep = episodes.get(m, {})
                num = {q["instance_id"]: q.get("numbers") or {} for q in kq}
                row = {"n": len(kq), "correct": sum(ok.values()), "unparsed": sum(v is None for v in got.values()),
                       "accuracy": sum(ok.values()) / len(kq) if kq else float("nan"), "groups": {}}
                for name, sel in (("all", lambda i: True),
                                  *[(f"length {b}", (lambda b: lambda i: band_of(insts[i]["shortest"]["actions"]) == b)(b))
                                    for b, _, _ in LENGTH_BANDS],
                                  ("near threshold", lambda i: bool(insts[i].get("near_threshold"))),
                                  ("clear", lambda i: not insts[i].get("near_threshold")),
                                  *(FIRST_ACTION_GROUPS(num) if kind == "first_action" else ())):
                    cell = {}
                    for side, want in (("right", True), ("wrong", False)):
                        ids = [i for i, v in ok.items() if v is want and sel(i) and i in ep]
                        cell[side] = {"n": len(ids), "success": sum(bool(ep[i].get("success")) for i in ids),
                                      "max_shaping": mean(float(ep[i].get("max_shaping_score") or 0.0) for i in ids)}
                    row["groups"][name] = cell
                rep["models"][m][f"{reading}/{kind}"] = row
    models = list(answers)
    for reading in ("strict", "lenient"):
        for kind in KINDS:
            for a in models:
                for b in models:
                    if a == b:
                        continue
                    ra, rb = right.get((a, reading, kind), {}), right.get((b, reading, kind), {})
                    ea, eb = episodes.get(a, {}), episodes.get(b, {})
                    ids = [i for i in ra if ra.get(i) and rb.get(i) is False and i in ea and i in eb]
                    a_only = sum(bool(ea[i].get("success")) and not eb[i].get("success") for i in ids)
                    b_only = sum(bool(eb[i].get("success")) and not ea[i].get("success") for i in ids)
                    rep["pairs"][f"{reading}/{kind}/{a}>{b}"] = {"n": len(ids), "a_succeeds_b_fails": a_only,
                                                                 "b_succeeds_a_fails": b_only,
                                                                 "p": binom_two_sided(a_only, a_only + b_only)}
    return rep


def markdown(rep: dict) -> str:
    lines = ["# N 系列\"会答是否会做\"（自动生成）", "",
             "每格：答对实例上 N1 回合的成功数 / 实例数（平均最高塑形分） 对 答错实例上的同样数字。", ""]
    for key in sorted({k for r in rep["models"].values() for k in r}):
        lines += [f"## {key}", "", "| 模型 | 正确率 | 读不出 | 全部：答对 | 全部：答错 | 门槛附近：答对 | 门槛附近：答错 |",
                  "| --- | ---: | ---: | --- | --- | --- | --- |"]
        for m, r in rep["models"].items():
            row = r.get(key)
            if not row:
                continue

            def cell(c):
                return f"{c['success']}/{c['n']}（{c['max_shaping']:.3f}）" if c["n"] else "—"
            g = row["groups"]
            lines.append(f"| {m} | {row['correct']}/{row['n']}（{100 * row['accuracy']:.1f}%） | {row['unparsed']} | "
                         f"{cell(g['all']['right'])} | {cell(g['all']['wrong'])} | {cell(g['near threshold']['right'])} | "
                         f"{cell(g['near threshold']['wrong'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def analyze_cmd(args) -> int:
    qs = read_jsonl(args.questions)
    insts = {r["instance_id"]: r for r in read_jsonl(args.instances)}
    answers = {}
    for spec in args.answers:
        m, path = spec.split("=", 1)
        answers[m] = {r["id"]: r.get("reply") for r in read_jsonl(path)}
    episodes = {m: episodes_of(args.episodes, m) for m in answers}
    rep = analyze(qs, answers, episodes, insts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (args.out / "report.md").write_text(markdown(rep), encoding="utf-8")
    print(json.dumps({m: {k: round(v["accuracy"], 3) for k, v in r.items()} for m, r in rep["models"].items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--instances", type=Path, required=True)
    t.add_argument("--scene", required=True)
    t.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("paths")
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    p.add_argument("--max-states", type=int, default=MAX_STATES)
    c = sub.add_parser("collect")
    c.add_argument("--truth", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("make")
    m.add_argument("--instances", type=Path, required=True)
    m.add_argument("--gen", type=Path, nargs="+", required=True)
    m.add_argument("--truth", type=Path, default=None, help="truth's rows (best_action)")
    m.add_argument("--paths", type=Path, default=None, help="paths' rows (first_action)")
    m.add_argument("--object-image", choices=("start", "shortest", "best"), default="start",
                   help="the view the two object kinds are asked on")
    m.add_argument("--kinds", nargs="*", choices=KINDS, default=None)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=20261105)
    m.add_argument("--delta", type=float, default=DELTA)
    s = sub.add_parser("samples")
    s.add_argument("--questions", type=Path, required=True)
    s.add_argument("--instances", type=Path, required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--counts", nargs="*", default=None, help="KIND=N per kind (instead of --n spread over the kinds)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--questions", type=Path, required=True)
    a.add_argument("--answers", nargs="+", required=True, help="MODEL=answers.jsonl (q0_ask.py's)")
    a.add_argument("--episodes", type=Path, nargs="+", required=True, help="N1's evaluation folders")
    a.add_argument("--instances", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"truth": truth_cmd, "paths": paths_cmd, "collect": collect_cmd, "make": make_cmd, "samples": samples_cmd,
            "analyze": analyze_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
