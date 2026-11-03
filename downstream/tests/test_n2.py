"""N2 (few-shot; the user's decisions 2-3 of 2026-11-02 and 1 of 2026-11-03): the evaluation instances, the
demonstration datasets (downstream/n/n2_data.py), the LoRA trainer's pieces that need no model
(downstream/n/n2_train.py: reply spans, schedule, adapter save and in-memory merge) and the per-model job
(downstream/n/run_n2.sh)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT / "train"))
import n2_data as D  # noqa: E402
import n2_train as T  # noqa: E402
import n_env as N  # noqa: E402


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# -------------------------------------------------------------------------------------------- n2_data
def test_instances_merge_both_sets_refuse_duplicates_and_take_the_smoke_subset(tmp_path):
    a = [{"instance_id": f"A_s1000_{t}", "task_type": t} for t in N.FAMILIES for _ in range(1)]
    b = [{"instance_id": f"A_s3000_{t}_{k}", "task_type": t} for t in N.FAMILIES for k in range(3)]
    write_jsonl(tmp_path / "a.jsonl", a)
    write_jsonl(tmp_path / "b.jsonl", b)
    s = D.instances([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], tmp_path / "out", smoke_per_family=2)
    assert s["instances"] == 24 and s["smoke"] == 12
    smoke = [json.loads(x) for x in (tmp_path / "out/instances_smoke.jsonl").read_text().splitlines()]
    assert [x["instance_id"] for x in smoke[:2]] == ["A_s1000_projective_relations", "A_s3000_projective_relations_0"]
    write_jsonl(tmp_path / "c.jsonl", a[:1])
    with pytest.raises(SystemExit):
        D.instances([tmp_path / "a.jsonl", tmp_path / "c.jsonl"], tmp_path / "out2")


def test_action_runs_chunking_caps_each_run_at_three():
    acts = ["turn_left"] * 4 + ["move_forward", "move_forward", "turn_right"]
    assert D.chunk_action_runs(acts, 3) == [["turn_left"] * 3, ["turn_left"], ["move_forward"] * 2, ["turn_right"]]
    assert D.chunk_action_runs(acts, 5) == [["turn_left"] * 4, ["move_forward"] * 2, ["turn_right"]]


def demo(t, k, n_actions=5, ok=True, scene=None):
    acts = ["turn_left", "turn_left", "move_forward", "move_forward", "move_left"][:n_actions]
    frames = [{"pose": [0.1 * i, 0.0, 0.0], "action": a, "image": f"path_frames/{t}_{k}_p{i:02d}.png",
               "success": ok and i == n_actions} for i, a in enumerate([None] + acts)]
    return {"instance_id": f"{t}_{k}", "task_type": t, "scene_id": scene or f"S{k}", "task": f"Task{k % 3}",
            "item": {"task_description": f"do {t}"}, "path_frames": frames}


def test_order_is_nested_even_over_types_and_ignores_the_input_order():
    demos = [demo(t, k) for t in N.FAMILIES for k in range(1 if t == "occlusion_alignment" else 4)]
    order = D.demo_order(demos, seed=5)
    assert [d["task_type"] for d in order[:6]] == list(N.FAMILIES)  # one of each type first
    assert sum(d["task_type"] == "occlusion_alignment" for d in order) == 1  # runs out, the others go on
    assert [d["instance_id"] for d in D.demo_order(list(reversed(demos)), seed=5)] == [d["instance_id"] for d in order]
    assert [d["instance_id"] for d in D.demo_order(demos, seed=6)] != [d["instance_id"] for d in order]


@pytest.fixture
def stub_prompts(monkeypatch):
    import n_episode as E

    monkeypatch.setattr(E, "upstream_config", lambda path: ({"path": str(path)}, {}))
    monkeypatch.setattr(E, "system_text", lambda cfg, t: f"SYS {t} {cfg['path']}")
    monkeypatch.setattr(E, "user_text", lambda cfg, first, pose, desc, fb="":
                        f"{'FIRST' if first else fb}|<image>|{pose[0]:.1f}|{desc}")
    return E


def test_build_writes_nested_conversations_per_generation(tmp_path, stub_prompts):
    gen = tmp_path / "gen"
    gen.mkdir()
    demos = [demo(t, k) for t in N.FAMILIES for k in range(3)]
    demos += [demo("equidistance", 9, ok=False)]  # its path never succeeds: not used
    write_jsonl(gen / "instances_final.jsonl", demos)
    s = D.build(gen, tmp_path / "out", sizes=(6, 12), max_per_turn=3, configs={"qwen25": "c25", "qwen3": "c3"})
    assert s["found"] == 19 and s["usable"] == 18 and s["sizes"]["6"]["per_type"] == {t: 1 for t in N.FAMILIES}
    assert s["sizes"]["12"]["actions_per_turn"] == {1: 12, 2: 24}  # tl tl | mf mf | ml
    six = [json.loads(x) for x in (tmp_path / "out/qwen25/demos_6.jsonl").read_text().splitlines()]
    twelve = [json.loads(x) for x in (tmp_path / "out/qwen25/demos_12.jsonl").read_text().splitlines()]
    assert [r["id"] for r in twelve[:6]] == [r["id"] for r in six]  # nested
    r = six[0]
    assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert r["messages"][0]["content"].endswith(" c25")
    assert [m["content"] for m in r["messages"][2::2]] == ["<action>turn_left|turn_left|</action>",
                                                            "<action>move_forward|move_forward|</action>",
                                                            "<action>move_left|</action>"]
    assert r["messages"][1]["content"].startswith("FIRST|<image>|0.0|")
    assert r["messages"][3]["content"].startswith("Action executed.|<image>|0.2|")  # the frame at the turn start
    assert [Path(p).name for p in r["images"]] == [f"{r['id']}_p00.png", f"{r['id']}_p02.png", f"{r['id']}_p04.png"]
    q3 = json.loads((tmp_path / "out/qwen3/demos_6.jsonl").read_text().splitlines()[0])
    assert q3["id"] == r["id"] and q3["messages"][0]["content"].endswith(" c3")
    order = json.loads((tmp_path / "out/order.json").read_text())
    assert len(order) == 18 and order[0]["turns"] == [["turn_left"] * 2, ["move_forward"] * 2, ["move_left"]]


def test_build_refuses_evaluation_scenes_and_too_few_demonstrations(tmp_path, stub_prompts):
    gen = tmp_path / "gen"
    gen.mkdir()
    write_jsonl(gen / "instances_final.jsonl", [demo(t, k) for t in N.FAMILIES for k in range(2)])
    write_jsonl(tmp_path / "eval.jsonl", [{"scene_id": "S1"}])
    with pytest.raises(SystemExit, match="evaluation scenes"):
        D.build(gen, tmp_path / "o1", sizes=(6,), exclude=[tmp_path / "eval.jsonl"], configs={"qwen25": "c"})
    with pytest.raises(SystemExit, match="only 12 usable"):
        D.build(gen, tmp_path / "o2", sizes=(13,), configs={"qwen25": "c"})


# ------------------------------------------------------------------------------------------- n2_train
def test_reply_spans_cover_each_reply_and_its_end_token():
    ids = [0, 1, 2, 3, 5, 6, 9, 7, 1, 2, 3, 8, 9, 4]
    assert T.reply_spans(ids, [1, 2, 3], 9) == [(4, 7), (11, 13)]
    T.check_spans({"id": "x", "messages": [{"role": "assistant"}] * 2}, [(4, 7), (11, 13)])
    with pytest.raises(ValueError):
        T.check_spans({"id": "x", "messages": [{"role": "assistant"}] * 3}, [(4, 7), (11, 13)])


def test_chat_messages_turn_image_placeholders_into_parts():
    msgs = T.chat_messages({"messages": [{"role": "system", "content": "sys"},
                                         {"role": "user", "content": "[Obs]:\n<image>\npose"}]})
    assert msgs[0]["content"] == [{"type": "text", "text": "sys"}]
    assert msgs[1]["content"] == [{"type": "text", "text": "[Obs]:\n"}, {"type": "image"},
                                  {"type": "text", "text": "\npose"}]


def test_schedule_warms_up_then_decays_to_zero():
    assert T.lr_at(0, 100) == pytest.approx(1e-4 / 3) and T.lr_at(2, 100) == pytest.approx(1e-4)
    assert T.lr_at(3, 100) == pytest.approx(1e-4) and T.lr_at(99, 100) < 1e-6
    assert T.lr_at(0, 5) == pytest.approx(1e-4)  # at least one warm-up step


def test_adapter_saved_then_merged_gives_the_wrapped_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    import lora

    def tiny():
        torch.manual_seed(0)
        m = nn.Module()
        m.model = nn.Module()
        m.model.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        layer.self_attn.q_proj, layer.mlp.down_proj = nn.Linear(6, 6), nn.Linear(6, 6)
        m.model.language_model.layers = nn.ModuleList([layer])
        return m

    wrapped, plain = tiny(), tiny()
    info = lora.wrap(wrapped, rank=2, alpha=4, dropout=0.0, seed=0)
    assert info["wrapped"] == 2
    for mod in wrapped.modules():
        if isinstance(mod, lora.LoRALinear):
            mod.lora_B.data.normal_()
    state = T.adapter_state(wrapped)
    assert sorted(state) == ["model.language_model.layers.0.mlp.down_proj", "model.language_model.layers.0.self_attn.q_proj"]
    torch.save(state, tmp_path / "adapter.pt")
    (tmp_path / "meta.json").write_text(json.dumps({"rank": 2, "alpha": 4}))
    assert T.merge_adapter(plain, tmp_path / "adapter.pt") == 2
    x = torch.randn(3, 6)
    layer_w, layer_p = wrapped.model.language_model.layers[0], plain.model.language_model.layers[0]
    with torch.no_grad():
        assert torch.allclose(layer_w.self_attn.q_proj(x), layer_p.self_attn.q_proj(x), atol=1e-5)
        assert torch.allclose(layer_w.mlp.down_proj(x), layer_p.mlp.down_proj(x), atol=1e-5)


# ------------------------------------------------------------------------------------------- run_n2.sh
TRAIN_STUB = r"""import json, os, sys
from pathlib import Path
a = sys.argv
out = Path(a[a.index("--out") + 1])
data = a[a.index("--data") + 1]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"train": data, "model": a[a.index("--model") + 1]}) + "\n")
if "fail" in Path(data).name:
    sys.exit(3)
out.mkdir(parents=True, exist_ok=True)
(out / "adapter.pt").write_text("x")
"""

EPISODES_STUB = r"""#!/usr/bin/env bash
mkdir -p "$1"
printf '{"eval": "%s", "instances": "%s", "model": "%s", "workers": "%s", "limit": "%s", "adapter": "%s", "val_env": "%s"}\n' \
  "$1" "$2" "$3" "$4" "$5" "${N_ADAPTER:-}" "${N_VAL_ENV:-}" >> "$STUB_LOG"
echo '{"success": true}' >> "$1/episodes.jsonl"
"""


def run_job(tmp_path, model, tiers, extra_env=None):
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True, exist_ok=True)
    (wt / "downstream/n/run_n2.sh").write_text((ROOT / "n/run_n2.sh").read_text())
    (wt / "downstream/n/n2_train.py").write_text(TRAIN_STUB)
    (wt / "downstream/n/run_n_episodes.sh").write_text(EPISODES_STUB)
    env = dict(os.environ, N_VLM_PY=sys.executable, STUB_LOG=str(tmp_path / "log.jsonl"), **(extra_env or {}))
    env.pop("N_VAL_ENV", None)
    return subprocess.run(["bash", str(wt / "downstream/n/run_n2.sh"), str(tmp_path / "run"), model, "/data",
                           "/inst.jsonl", tiers, "6", "0"], capture_output=True, text=True, env=env, timeout=60)


def test_job_trains_then_evaluates_each_tier_and_skips_a_trained_one(tmp_path):
    (tmp_path / "run/adapters/100").mkdir(parents=True)
    (tmp_path / "run/adapters/100/adapter.pt").write_text("x")  # trained before
    res = run_job(tmp_path, "B", "100 300")
    assert res.returncode == 0, res.stdout + res.stderr
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [x.get("train") for x in log if "train" in x] == ["/data/qwen25/demos_300.jsonl"]
    evals = [x for x in log if "eval" in x]
    assert [Path(x["eval"]).name for x in evals] == ["100", "300"] and all(x["val_env"] == "" for x in evals)
    assert evals[1]["adapter"] == str(tmp_path / "run/adapters/300/adapter.pt") and evals[1]["workers"] == "6"
    assert "evaluate 300 rc=0 1 successes of 1" in (tmp_path / "run/n2.log").read_text()


def test_job_qwen3_variant_and_a_failed_training_skips_its_evaluation(tmp_path):
    res = run_job(tmp_path, "Q3S", "fail 100", {"N_VAL_ENV_QWEN3": "/q3.yaml"})
    assert res.returncode == 1
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [x["train"] for x in log if "train" in x] == ["/data/qwen3/demos_fail.jsonl", "/data/qwen3/demos_100.jsonl"]
    evals = [x for x in log if "eval" in x]
    assert [Path(x["eval"]).name for x in evals] == ["100"] and evals[0]["val_env"] == "/q3.yaml"
