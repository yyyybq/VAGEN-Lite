"""T11 for the phase-2 arms (docs/q38/DESIGN_ASYNC.md 5.1 T11, 6 W10): the configs rl-config writes for them.

- Q5 PT (辛 = the reward line's rw_tcswf3; reference YQ5 = reward_runs/runs/rw_p3n_s2tcswf3c/rl_rw_s2tcswf3c, the seed
  variant 20261021 that DESIGN_ASYNC names): from YQ5's own command line (its logs/rl_config_rw_s2tcswf3c.log), the
  8B path with the ported --turn-credit section writes YQ5 itself (paths aside), and the q38 stack with the 27B C0 start
  writes a config whose every key-by-key difference to YQ5 is registered: test_config.py's register (the q38 stack's
  renames, R1-R13), the qwen3_5 layout and HF export, the start model, or Q5_REGISTER below; the contract passes it
  at the PT readings' endpoints (160, 320);
- the existing p1 configs: the four 27B main-line RL configs the live chains wrote with c2c6ed47
  (runs/q38-main-rl/s2026101{9,21}/rl_q38{s0,v4}s{1,2}), regenerated from their logged command lines, are byte-identical
  (the run and code-tree folders mapped to tokens);
- the main line's per-turn PPO (8B qw3tgd: v10_p1tgae_Dcond + V4 keys) on the q38 stack: profile turn_ppo, with phase
  3's value network (the 27B critic layout, qw3tgd's critic values); the contract passes it at 310 (qw3tgd's endpoint).

Reads the run folders read-only; writes to a temporary folder (Q38_TEST_TMP, or /mnt/umm/users/yinbaiqiao/q38/tmp_tests).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from data_gen.interiorgs_v10 import training_prep as tp

REPO = Path(__file__).resolve().parents[3]
YQ5 = Path("/mnt/umm/users/yinbaiqiao/reward_runs/runs/rw_p3n_s2tcswf3c/rl_rw_s2tcswf3c")
YQ5_LOG = YQ5.parent / "logs" / "rl_config_rw_s2tcswf3c.log"
YQ5_TEMPLATE = YQ5.parent / "ppo_template_e3.yaml"
MODEL27 = "/mnt/umm/users/yinbaiqiao/q38/runs/q38-c0-sft/lf_q38c0/model/checkpoint-770"
MAIN = Path("/mnt/umm/users/yinbaiqiao/q38/runs/q38-main-rl")
MAIN_RUNS = [("s20261019", "q38s0s1"), ("s20261019", "q38v4s1"), ("s20261021", "q38s0s2"), ("s20261021", "q38v4s2")]
LIVE_TREE = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38"
TMP_ROOT = Path("/mnt/umm/users/yinbaiqiao/q38/tmp_tests")
INPUTS = [YQ5 / "ppo.yaml", YQ5_LOG, YQ5_TEMPLATE, Path(MODEL27) / "config.json"]
if not all(p.exists() for p in INPUTS):
    pytest.skip(f"needs {[str(p) for p in INPUTS if not p.exists()]}", allow_module_level=True)

from vagen.q38.tests import test_config as T11  # noqa: E402  (its own module-level skip needs the 8B inputs)

# phase 2's own registrations (the differences of the PT arm that T11's 8B main-line runs do not have)
Q5_DROPPED = [("algorithm.turn_credit.*", "stack:rename",
               "-> vagen.turn_credit.* (V1's AlgoConfig refuses an unknown algorithm key; vagen/q38/trainer.py)")]
Q5_NEW = [("vagen.turn_credit.*", "stack:rename", "<- algorithm.turn_credit.*, the same values"),
          ("vagen.profile", "stack:new", "q5_pt: the update shape the contract checks")]
MODEL_KEYS = ("actor_rollout_ref.model.path", "critic.model.path", "critic.model.tokenizer_path",
              "trainer.v1.sampler.sampler_kwargs.model_path")


def _argv(log: Path):
    """The rl-config arguments of a chain's logged command line.  The chain logs a JSON option (--turn-credit {...})
    without shell quotes, so it is cut out before the split and put back as one word."""
    line = log.read_text().splitlines()[0][2:]
    objects = []
    while "{" in line:
        start = line.index("{")
        end = line.index("}", start) + 1  # the settings objects have no nested braces
        objects.append(line[start:end])
        json.loads(objects[-1])
        line = f"{line[:start]}__JSON{len(objects) - 1}__{line[end:]}"
    words = [objects[int(w[6:-2])] if w.startswith("__JSON") else w for w in shlex.split(line)]
    return words[words.index("rl-config"):]


def _set_opt(argv, opt, value):
    argv = list(argv)
    argv[argv.index(opt) + 1] = value
    return argv


@pytest.fixture(scope="module")
def work():
    base = os.environ.get("Q38_TEST_TMP") or (str(TMP_ROOT) if TMP_ROOT.is_dir() else None)
    path = Path(tempfile.mkdtemp(prefix="test_config_q5_", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def pt(work):
    """{"8b": (out, run), "q38": (out, run)}: YQ5's command line on both stacks (the e3 template copied here)."""
    template = work / "ppo_template_e3.yaml"
    template.write_text(YQ5_TEMPLATE.read_text())
    argv = _set_opt(_argv(YQ5_LOG), "--ppo-template", str(template))
    out = {}
    for name, extra in (("8b", []), ("q38", ["--model", MODEL27, "--model-type", "qwen3_5", "--name", "q38_t_tcswf3c"])):
        rl, run = work / f"rl_{name}", work / f"run_{name}"
        a = _set_opt(_set_opt(argv, "--out", str(rl)), "--run-root", str(run))
        for i in range(0, len(extra), 2):
            a = _set_opt(a, extra[i], extra[i + 1])
        assert tp.main(a) == 0
        out[name] = (rl, run)
    out["template"] = template
    return out


def test_pt_8b_path_writes_yq5(pt):
    rl, run = pt["8b"]
    subs_mine = [(str(run.resolve()), "<RUN>"), (str(rl.resolve()), "<OUT>"), (str(pt["template"]), "<E3>")]
    subs_yq5 = [(str(YQ5.parent / "rl_run_rw_s2tcswf3c"), "<RUN>"), (str(YQ5), "<OUT>"), (str(YQ5_TEMPLATE), "<E3>")]
    assert T11._norm((rl / "ppo.yaml").read_text(), subs_mine) == T11._norm((YQ5 / "ppo.yaml").read_text(), subs_yq5)
    for name in ("train_env.yaml", "val_env.yaml", "train.jsonl", "val.jsonl"):
        assert T11._norm((rl / name).read_text(), subs_mine) == T11._norm((YQ5 / name).read_text(), subs_yq5), name
    mine = json.loads((rl / "rl_config_report.json").read_text())
    yq5 = json.loads((YQ5 / "rl_config_report.json").read_text())
    differ = sorted(k for k in set(mine) | set(yq5) if mine.get(k, "<absent>") != yq5.get(k, "<absent>"))
    assert differ == ["ppo_template"], differ  # the template copy's path; the key order differs (registered)


def _classify(key, kind, layout_keys, raw):
    target = raw.get(key)
    target = target[2:-1].split(",")[0].replace("oc.select:", "") if isinstance(target, str) and \
        target.startswith("${") else None
    if key in MODEL_KEYS or target in MODEL_KEYS:
        return "start-model"
    if key in T11_RUN_NAMES:
        return "run-name"
    if key in layout_keys or target in layout_keys or (kind == "dropped" and any(k.startswith(key + ".")
                                                                                 for k in layout_keys)):
        return "q38-layout"
    table = {"dropped": T11.DROPPED + Q5_DROPPED, "changed": T11.CHANGED,
             "new": T11.NEW + T11.HF_EXPORT_NEW + Q5_NEW}[kind]
    m = T11._match(key, table)
    return m[1] if m else None


T11_RUN_NAMES = ("trainer.experiment_name",)


def test_pt_q38_differences_to_yq5_registered(pt):
    rl, run = pt["q38"]
    q = yaml.safe_load((rl / "ppo.yaml").read_text())
    y = yaml.safe_load((YQ5 / "ppo.yaml").read_text())
    fq = T11._resolved_flat(q, [(str(run.resolve()), "<RUN>"), (str(rl.resolve()), "<OUT>"), (str(pt["template"]), "<E3>")])
    fy = T11._resolved_flat(y, [(str(YQ5.parent / "rl_run_rw_s2tcswf3c"), "<RUN>"), (str(YQ5), "<OUT>"),
                                (str(YQ5_TEMPLATE), "<E3>")])
    default = T11._flat(yaml.safe_load(tp.Q38_VERL_DEFAULT.read_text()))
    raw = T11._flat(q)
    layout_keys = set(T11._flat(yaml.safe_load(tp.Q38_LAYOUTS["qwen3_5"].read_text())))
    rows = [("dropped", k) for k in sorted(set(fy) - set(fq))]
    rows += [("changed", k) for k in sorted(set(fy) & set(fq)) if fy[k] != fq[k]]
    rows += [("new", k) for k in sorted(set(fq) - set(fy)) if not T11._at_default(k, raw, default)]
    classes = {(kind, k): _classify(k, kind, layout_keys, raw) for kind, k in rows}
    unexplained = [f"{kind} {k}: YQ5 {fy.get(k, '<absent>')!r} -> q38 {fq.get(k, '<absent>')!r}"
                   for (kind, k), c in classes.items() if c is None]
    assert not unexplained, "\n".join(unexplained)
    assert {c for c in classes.values()} >= {"stack:rename", "q38-layout", "start-model", "R1", "R2"}
    # the PT recipe itself is YQ5's
    assert q["vagen"]["turn_credit"] == y["algorithm"]["turn_credit"]
    for k in ("data.train_files", "data.val_files", "data.seed", "filter.name", "filter.enable",
              "filter.filter_kwargs.target_groups", "actor_rollout_ref.actor.ppo_epochs",
              "actor_rollout_ref.actor.ppo_mini_batch_size", "data.train_batch_size", "actor_rollout_ref.rollout.n",
              "actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.kl_loss_coef",
              "actor_rollout_ref.actor.entropy_coeff", "algorithm.adv_estimator", "trainer.save_freq",
              "trainer.critic_warmup", "actor_rollout_ref.actor.clip_ratio_c", "actor_rollout_ref.actor.kl_loss_type",
              "actor_rollout_ref.actor.optim.clip_grad"):
        want = fy["actor_rollout_ref.actor.grad_clip"] if k.endswith("optim.clip_grad") else fy[k]
        assert fq[k] == want, k
    assert (q["vagen"]["ga_8b"], q["vagen"]["profile"]) == (24, "q5_pt")
    assert q["trainer"]["v1"]["sampler"]["sampler_kwargs"]["adv_estimator"] == "${algorithm.adv_estimator}"
    for split in ("train", "val"):  # the env yamls are YQ5's (the env template, S0, the table and its neighbours)
        a = yaml.safe_load((rl / f"{split}_env.yaml").read_text().replace(str(rl.resolve()), "<OUT>"))
        b = yaml.safe_load((YQ5 / f"{split}_env.yaml").read_text().replace(str(YQ5), "<OUT>"))
        assert a == b, split
    report = json.loads((rl / "rl_config_report.json").read_text())
    assert report["q38"]["profile"] == "q5_pt" and report["adv_estimator"] == "turn_credit"


@pytest.mark.parametrize("endpoint", [160, 320])
def test_pt_q38_contract(pt, endpoint):
    from omegaconf import OmegaConf

    from vagen.q38 import contract

    report = contract.evaluate(OmegaConf.load(pt["q38"][0] / "ppo.yaml"), endpoint)
    assert report["passed"], contract.summary(report)
    assert report["profile"]["shape"] == "q5_pt"


@pytest.mark.skipif(not MAIN.is_dir(), reason=f"needs {MAIN}")
@pytest.mark.parametrize("seed_dir,run", MAIN_RUNS)
def test_existing_main_line_configs_unchanged(work, seed_dir, run):
    live = MAIN / seed_dir / f"rl_{run}"
    argv = _argv(MAIN / seed_dir / "logs" / f"rl_config_{run}.log")
    live_run = argv[argv.index("--run-root") + 1]
    out, new_run = work / f"main_{run}", work / f"main_run_{run}"
    assert tp.main(_set_opt(_set_opt(argv, "--out", str(out)), "--run-root", str(new_run))) == 0
    for name in ("ppo.yaml", "rl_config_report.json", "train_env.yaml", "val_env.yaml", "train.jsonl", "val.jsonl"):
        mine = (out / name).read_text().replace(str(new_run.resolve()), "<RUN>").replace(
            str(out.resolve()), "<OUT>").replace(str(REPO), "<TREE>")
        theirs = (live / name).read_text().replace(live_run, "<RUN>").replace(str(live), "<OUT>").replace(
            LIVE_TREE, "<TREE>")
        assert mine == theirs, (run, name)


def test_turn_ppo_on_the_q38_stack(work):
    """qw3tgd's options (v10_p1tgae_Dcond + the V4 keys) with the 27B start: profile turn_ppo, the package's per-turn
    gamma / lam, the value network on (explicitly), warm-up 11; phase 3: the 27B critic layout (FSDP2, optimizer state
    on the card for the step), the 8B critic values, and the contract passes at qw3tgd's endpoint 310."""
    from omegaconf import OmegaConf

    from vagen.q38 import contract

    argv = _argv(MAIN / "s20261019" / "logs" / "rl_config_q38v4s1.log")
    argv = _set_opt(argv, "--standard", "v10_p1tgae_Dcond")
    out, run = work / "rl_tgv4", work / "run_tgv4"
    assert tp.main(_set_opt(_set_opt(argv, "--out", str(out)), "--run-root", str(run))) == 0
    cfg = yaml.safe_load((out / "ppo.yaml").read_text())
    assert (cfg["algorithm"]["adv_estimator"], cfg["algorithm"]["gamma"], cfg["algorithm"]["lam"]) == ("turn_gae", 1.0, 0.95)
    assert cfg["critic"]["enable"] is True and cfg["trainer"]["critic_warmup"] == 11
    assert cfg["vagen"]["profile"] == "turn_ppo" and cfg["vagen"]["ga_8b"] == 12
    assert cfg["filter"]["name"] == "keep_by_return_std"
    assert "turn_credit" not in cfg["vagen"] and "turn_credit" not in cfg["algorithm"]
    c = cfg["critic"]
    assert (c["strategy"], c["fsdp"]["strategy"], c["fsdp"]["optimizer_offload_step"]) == ("fsdp2", "fsdp2", True)
    assert cfg["actor_rollout_ref"]["actor"]["strategy"] == "fsdp2"
    assert (c["optim"]["lr"], c["optim"]["lr_scheduler_type"], c["optim"]["clip_grad"], c["cliprange_value"]) == \
        (2e-5, "constant", 1.0, 0.5)
    assert (c["ppo_micro_batch_size_per_gpu"], c["fsdp"]["param_offload"], c["fsdp"]["optimizer_offload"]) == (1, True, True)
    assert c["model"]["path"] == MODEL27 and c["model"]["tokenizer_path"] == MODEL27
    report = json.loads((out / "rl_config_report.json").read_text())
    assert sorted(report["q38"]["critic_layout"]["keys"]) == sorted(tp.Q38_CRITIC_LAYOUT_KEYS)
    for endpoint in (310, 160):
        report = contract.evaluate(OmegaConf.create(cfg), endpoint)
        assert report["passed"], contract.summary(report)


def test_turn_ppo_refuses_a_changed_8b_critic(work, monkeypatch):
    """A run whose 8B config changed a critic key the q38 config takes from the recipe is refused, not dropped."""
    real = tp.q38_ppo_config

    def changed(ppo8, *a, **k):
        ppo8 = json.loads(json.dumps(ppo8))
        ppo8["critic"]["optim"]["lr"] = 5e-5
        return real(ppo8, *a, **k)

    monkeypatch.setattr(tp, "q38_ppo_config", changed)
    argv = _argv(MAIN / "s20261019" / "logs" / "rl_config_q38v4s1.log")
    argv = _set_opt(argv, "--standard", "v10_p1tgae_Dcond")
    out, run = work / "rl_tgv4_bad", work / "run_tgv4_bad"
    with pytest.raises(SystemExit, match="critic.optim.lr"):
        tp.main(_set_opt(_set_opt(argv, "--out", str(out)), "--run-root", str(run)))
