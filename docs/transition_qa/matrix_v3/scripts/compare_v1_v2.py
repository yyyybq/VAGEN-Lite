"""Group G data v1 (dcaa0207) against v2 (e4e245c0, the pre-launch review fixes): every change must come from one of
the two fixes -- a target with box corners behind the camera (the near-plane box) or a scene whose rows carry a
structure Y sign the env applies (the walls) -- and nothing but assistant labels / Pg boxes may differ."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
import numpy as np  # noqa: E402

from vagen.envs.active_spatial import pointing as PT  # noqa: E402

G = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/g_prep_20261106")
SFT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft")
FMT = "targets_norm1000"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


rows = {r["task_id"]: r for r in jl(SFT / "rows.jsonl")}
poses = {f["frame_id"]: f["c2w"] for f in jl(SFT / "frames.jsonl")}
by_scene = collections.defaultdict(list)
for r in rows.values():
    by_scene[r["scene_id"]].append(r)
signed = {s for s, rs in by_scene.items() if PT.scene_structure_sign(rs) is not None}


def behind(item, fid):
    """Some target of the task with box corners behind the camera at this frame."""
    c2w = np.asarray(poses[fid], float)
    return any((PT._camera_corners(o, c2w)[1][:, 2] <= 1e-6).any() for o in PT.slot_objects(item)
               if "bbox_min" in o and "bbox_max" in o)


def frame_of(path):
    return str(path).rsplit("/", 1)[-1][:-4]


def compare_traj(name, v1, v2, msg_key, img_key, tid_key):
    a, b = jl(v1), jl(v2)
    assert [r[tid_key] for r in a] == [r[tid_key] for r in b], name
    c = collections.Counter()
    for r1, r2 in zip(a, b):
        m1, m2 = r1[msg_key], r2[msg_key]
        assert len(m1) == len(m2)
        item = rows[r1[tid_key]]
        imgs = r1[img_key]
        k = 0
        for x, y in zip(m1, m2):
            if x["role"] != "assistant":
                if x["content"] != y["content"]:
                    c["prompt_changed"] += 1  # must stay 0
                continue
            c["turns"] += 1
            if x["content"] != y["content"]:
                c["changed"] += 1
                why = []
                if behind(item, frame_of(imgs[k])):
                    why.append("behind")
                if item["scene_id"] in signed:
                    why.append("signed_scene")
                c["changed_" + ("+".join(why) or "UNEXPLAINED")] += 1
            k += 1
    print(name, dict(c))
    return c


ok = True
for split in ("train", "val"):
    c = compare_traj(f"P2t_{split}", G / "p2t" / FMT / f"P2t_{split}.jsonl", G / "p2t_v2" / FMT / f"P2t_{split}.jsonl",
                     "messages", "images", "source_id")
    ok &= not c["prompt_changed"] and not c["changed_UNEXPLAINED"]
c = compare_traj("sft_trajectories", G / "p2t" / FMT / "sft_trajectories.jsonl", G / "p2t_v2" / FMT / "sft_trajectories.jsonl",
                 "conversations", "conversation_image_paths", "source_task_id")
ok &= not c["prompt_changed"] and not c["changed_UNEXPLAINED"]

# Pg candidates: the same (task, state, slot) set, boxes changed only behind the camera
def cands(d):
    out = {}
    for f in sorted((d / "raw").glob("*.jsonl")):
        for r in jl(f):
            out[(r["task_id"], r["state_id"], r["slot"])] = r
    return out


c1, c2 = cands(G / "pg"), cands(G / "pg_v2")
pc = collections.Counter(only_v1=len(set(c1) - set(c2)), only_v2=len(set(c2) - set(c1)))
for k in set(c1) & set(c2):
    if c1[k]["box"] != c2[k]["box"]:
        pc["box_changed"] += 1
        item = rows[k[0]]
        fid = frame_of(c1[k]["image"])
        pc["box_changed_" + ("behind" if behind(item, fid) else "signed_scene" if item["scene_id"] in signed else "UNEXPLAINED")] += 1
for k in set(c1) ^ set(c2):
    pc["diff_in_signed_scene"] += rows[k[0]]["scene_id"] in signed
print("Pg candidates", dict(pc))
ok &= not pc["box_changed_UNEXPLAINED"] and pc["diff_in_signed_scene"] == pc["only_v1"] + pc["only_v2"]
if (G / "pg_v2" / FMT / "pg_train.jsonl").exists():
    for f in ("pg_train.jsonl", "pg_test.jsonl", "pg_probe.jsonl"):
        a = [r["sample_id"] for r in jl(G / "pg" / FMT / f)]
        b = [r["sample_id"] for r in jl(G / "pg_v2" / FMT / f)]
        print(f, "same questions" if a == b else f"questions differ: {len(set(a) ^ set(b))} of {len(a)}")
print("ALL CHANGES EXPLAINED" if ok else "UNEXPLAINED CHANGES")
