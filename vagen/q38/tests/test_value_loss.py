"""Phase 3 (docs/q38/DESIGN_ASYNC.md 5.5 T-loss, 6 W12; PEAR_ASSESSMENT B5): the value network of turn_gae.

- T-loss (value): vagen/q38/losses.py vagen_value_loss, run as verl V1's engine runs a critic loss (one micro-batch of one
  episode at a time, dp_size / global_batch_size from the trainer, the per-rank Metric lists aggregated by
  Metric.aggregate_dp, the ranks' gradients averaged), against the 8B's own update: F061 (VAGEN-Lite/verl, verl 0.6.1)
  DataParallelPPOCritic.update_critic (dp_critic.py:191-301) in the vagen-lite environment (``-I -B``, read-only import),
  rank after rank, on a value "model" that is an embedding table (one parameter a position, so every prediction's
  gradient is read off the SGD step).  Same episodes, same predictions: the gradient of every prediction, the
  objective and the 8B's logged critic/vf_loss agree to <= 1e-6; the loss lands on the turn anchors only; padding rows
  give 0 and count in the denominator; verl's stock value_loss on the same rows regresses the non-anchor positions to
  the sentinel (PEAR B5);
- the mask (F061 dp_critic.py:231-234 with the 8B's compute_value_mask, extracted from vagen/ray_trainer.py), the
  guards, a micro-batch of padding rows only, the pooled in-update readings (critic_update_readings) against a direct
  computation, critic_batch_readings against the 8B's compute_valid_return_metrics (extracted from
  vagen/ray_trainer.py) and by staleness;
- the trainer: _setup puts vagen_value_loss in place of verl's, _compute_values / _update_critic pass the one-episode
  micro-batch settings to the critic's engine and take them back, the update's readings and the 8B-unit vf_loss, the
  turn_gae branch's readings (EV by staleness 0 / 1), the sentinel-dominated readings dropped.

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_value_loss.py -q -p no:cacheprovider
"""

from __future__ import annotations

import ast
import copy
import functools
import json
import math
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vagen.q38.tests.sampler_fixtures import work_dir
from vagen.q38.tests.test_trainer import F061_ROOT, PY_8B, TrainerTQ, make_trainer
from vagen.q38.tests.test_turn_arms import STEP, arm_config, run_adv, turn_ppo_critic

REPO = Path(__file__).resolve().parents[3]
RT_SOURCE = REPO / "vagen" / "ray_trainer.py"  # the 8B trainer (read for its functions, not imported)
IGNORE = -100.0
CLIP = 0.5  # critic.cliprange_value of qw3tgd (ppo.yaml:385)
LOSS_CFG = SimpleNamespace(cliprange_value=CLIP, loss_agg_mode="seq-mean-token-mean", loss_scale_factor=None)


# ======================================================================================================================
# helpers
# ======================================================================================================================


def rt_functions(*names):
    """Functions of the 8B trainer (vagen/ray_trainer.py), compiled from its own source (the module itself imports
    verl 0.6.1, which this environment does not have)."""
    tree = ast.parse(RT_SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), [n.name for n in nodes]
    ns = {"torch": torch, "np": np, "DataProto": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RT_SOURCE), "exec"), ns)
    return [ns[n] for n in names]


def critic_rows(n_real, n_pad, seed, clip_some=True):
    """Episodes as the critic sees them: answers (mask 1) and observations (mask 0); each turn's return on its first
    answer token (the anchor), -100 elsewhere (vagen/turn_gae.py); the old values; the current predictions (some far
    enough from the old values that the value clip acts).  Padding rows: response length 1, mask 0, return -100
    (padding_utils.py; the q38 turn branch's returns)."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n_real):
        mask, anchors = [], []
        for t in range(int(torch.randint(1, 6, (1,), generator=g))):
            if t:
                mask += [0] * int(torch.randint(1, 4, (1,), generator=g))
            anchors.append(len(mask))
            mask += [1] * int(torch.randint(1, 6, (1,), generator=g))
        r = len(mask)
        returns = [IGNORE] * r
        for a in anchors:
            returns[a] = float(torch.randn(1, generator=g)) * 2.0
        values = (torch.randn(r, generator=g) * 1.5).tolist()
        noise = torch.randn(r, generator=g) * (1.0 if clip_some and i % 3 == 0 else 0.3)
        vpred = (torch.tensor(values) + noise).tolist()
        rows.append({"mask": mask, "returns": returns, "values": values, "vpred": vpred, "anchors": anchors,
                     "prompt_len": int(torch.randint(3, 9, (1,), generator=g)), "real": True})
    for _ in range(n_pad):
        rows.append({"mask": [0], "returns": [IGNORE], "values": [0.0], "vpred": [0.0], "anchors": [], "prompt_len": 127,
                     "real": False})
    return rows


def micro_data(mb, dp, gbs, batch_num_tokens):
    """The TensorDict a critic micro-batch reaches the loss with (jagged response-form fields, the engine's
    non-tensors)."""
    from verl.utils import tensordict_utils as tu

    def nested(name, dtype=torch.float32):
        return torch.nested.as_nested_tensor([torch.tensor(r[name], dtype=dtype) for r in mb], layout=torch.jagged)

    return tu.get_tensordict(
        {"prompts": torch.nested.as_nested_tensor([torch.zeros(r["prompt_len"], dtype=torch.int64) for r in mb],
                                                  layout=torch.jagged),
         "responses": torch.nested.as_nested_tensor([torch.zeros(len(r["mask"]), dtype=torch.int64) for r in mb],
                                                    layout=torch.jagged),
         "values": nested("values"), "returns": nested("returns"), "response_mask": nested("mask", torch.int64)},
        non_tensor_dict={"dp_size": dp, "batch_num_tokens": batch_num_tokens, "global_batch_size": gbs},
    )


def leaves_of(rows):
    """One leaf a row over (prompt + response): the predictions sit at [P-1, P+R-1) (no_padding_2_padding)."""
    out = []
    for r in rows:
        p, n = r["prompt_len"], len(r["mask"])
        full = torch.full((p + n,), 7.0)
        full[p - 1: p + n - 1] = torch.tensor(r["vpred"], dtype=torch.float32)
        out.append(full.requires_grad_(True))
    return out


def v1_critic(rows, loss_fn, dp=8, mini_n=96, micro=1, epochs=1):
    """The rows through ``loss_fn`` as verl V1 runs a critic update (engine_workers.py:243-340 / transformer_impl.py
    :788-841): per mini-batch, rank and micro-batch the loss and its backward (the ranks' gradients averaged, FSDP);
    the per-rank Metric lists aggregated with Metric.aggregate_dp, plain values flattened; one entry a mini-batch
    iteration; then TB:1894-1896 (rename_dict "critic/", reduce_metrics).  Returns (grads of the predictions a row,
    objective, the trainer's reduced metrics, the worker's per-iteration lists)."""
    from verl.utils.metric import Metric, reduce_metrics
    from verl.utils.py_functional import rename_dict

    assert len(rows) % mini_n == 0 and mini_n % dp == 0
    leaves = leaves_of(rows)
    objective = 0.0
    lists: dict = {}
    for _ in range(epochs):
        for m0 in range(0, len(rows), mini_n):
            idx = list(range(m0, m0 + mini_n))
            tokens = int(sum(sum(rows[i]["mask"]) for i in idx))
            per_rank = mini_n // dp
            rank_metrics = []
            for rank in range(dp):
                mine = idx[rank * per_rank:(rank + 1) * per_rank]
                acc: dict = {}
                for j in range(0, len(mine), micro):
                    mb = [rows[i] for i in mine[j:j + micro]]
                    data = micro_data(mb, dp, mini_n, tokens)
                    out = {"values": torch.nested.as_nested_tensor([leaves[i] for i in mine[j:j + micro]],
                                                                   layout=torch.jagged)}
                    loss, metrics = loss_fn(model_output=out, data=data)
                    (loss / dp).backward()
                    objective += float(loss.detach()) / dp
                    for k, v in metrics.items():
                        if isinstance(v, Metric):
                            acc.setdefault(k, Metric(aggregation=v.aggregation)).append(v)
                        else:
                            acc.setdefault(k, []).append(float(v))
                rank_metrics.append(acc)
            for k in rank_metrics[0]:
                vals = [rm[k] for rm in rank_metrics]
                agg = Metric.aggregate_dp(vals) if isinstance(vals[0], Metric) else float(np.mean(sum(vals, [])))
                lists.setdefault(k, []).append(float(agg))
    worker = {k: list(v) for k, v in lists.items()}
    reduced = reduce_metrics(rename_dict(copy.deepcopy(lists), "critic/"))
    grads = []
    for r, leaf in zip(rows, leaves):
        p, n = r["prompt_len"], len(r["mask"])
        g = leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
        assert float(g[:p - 1].abs().sum()) == 0.0 and float(g[p + n - 1:].abs().sum()) == 0.0
        grads.append(g[p - 1: p + n - 1].tolist())
    return grads, objective, {k: float(v) for k, v in reduced.items()}, worker


def q38_loss(gbs=96, micro=1):
    from vagen.q38.losses import vagen_value_loss

    return functools.partial(vagen_value_loss, config=LOSS_CFG, global_batch_size=gbs, max_rows_per_micro=micro)


def anchors_of(rows):
    return [(i, a) for i, r in enumerate(rows) for a in range(len(r["mask"]))
            if r["mask"][a] and abs(r["returns"][a] - IGNORE) >= 1e-2]


def hand_objective(rows, mini_n=96):
    """The 8B objective in closed form: sum over the episodes of 0.5 x the anchor mean of max((v-R)^2, (clip-R)^2),
    over mini x n (padding rows have no anchor)."""
    total = 0.0
    for r in rows:
        terms = []
        for a in range(len(r["mask"])):
            if r["mask"][a] and abs(r["returns"][a] - IGNORE) >= 1e-2:
                v, old, ret = r["vpred"][a], r["values"][a], r["returns"][a]
                vc = min(max(v, old - CLIP), old + CLIP)
                terms.append(max((v - ret) ** 2, (vc - ret) ** 2))
        if terms:
            total += 0.5 * sum(terms) / len(terms)
    return total / mini_n


# ======================================================================================================================
# the 8B's own critic update (F061 dp_critic.py) in the vagen-lite environment
# ======================================================================================================================

SCRIPT_CRITIC_8B = r'''
import json, sys
sys.path.insert(0, "@F061@")
from types import SimpleNamespace
import torch
import verl
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.critic.dp_critic import DataParallelPPOCritic

assert verl.__file__.startswith("@F061@"), verl.__file__
spec = json.load(open(sys.argv[1]))
rows, dp, per_gpu = spec["rows"], spec["dp"], spec["mini_per_gpu"]
# qw3tgd's critic (ppo.yaml:377-386, 423), after fsdp_workers' normalisation (mini x n / ranks per GPU); a clip too large
# to act, so the SGD step of lr 1 is minus the gradient
cfg = OmegaConf.create({"ppo_mini_batch_size": per_gpu, "ppo_micro_batch_size_per_gpu": 1, "ppo_epochs": 1,
                        "use_dynamic_bsz": False, "cliprange_value": spec["clip"], "loss_agg_mode": "token-mean",
                        "grad_clip": 1e30, "model": {"use_remove_padding": False}, "ulysses_sequence_parallel_size": 1,
                        "ppo_max_token_len_per_gpu": 32768, "shuffle": False})


class Table(torch.nn.Module):
    """A value 'model' whose output at a position is a parameter of its own (an embedding of the position's id)."""

    def __init__(self, n):
        super().__init__()
        self.table = torch.nn.Embedding(n, 1)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kw):
        return SimpleNamespace(logits=self.table(input_ids))


chunk = len(rows) // dp
out = {"grads": [], "metrics": []}
for rank in range(dp):
    data = rows[rank * chunk:(rank + 1) * chunk]
    P = max(r["prompt_len"] for r in data)
    R = max(len(r["mask"]) for r in data)
    n_ids = 1 + sum(len(r["mask"]) for r in data)
    model = Table(n_ids)
    ids = torch.zeros((len(data), P + R), dtype=torch.long)
    attn = torch.zeros((len(data), P + R), dtype=torch.long)
    base, spans = 1, []
    with torch.no_grad():
        model.table.weight.zero_()
        for j, r in enumerate(data):
            n = len(r["mask"])
            for k in range(n):  # the value read for response token k sits at position P - 1 + k (dp_critic.py:131)
                ids[j, P - 1 + k] = base + k
                model.table.weight[base + k, 0] = r["vpred"][k]
            attn[j, P - r["prompt_len"]: P + n] = 1
            spans.append((base, n))
            base += n
    def pad(name, fill, dtype):
        return torch.tensor([r[name] + [fill] * (R - len(r[name])) for r in data], dtype=dtype)
    batch = DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": attn, "position_ids": torch.clamp(torch.cumsum(attn, -1) - 1, min=0),
        "responses": torch.zeros((len(data), R), dtype=torch.long), "response_mask": pad("mask", 0, torch.long),
        "values": pad("values", 0.0, torch.float32), "returns": pad("returns", -100.0, torch.float32)})
    critic = DataParallelPPOCritic(cfg, model, torch.optim.SGD(model.parameters(), lr=1.0))
    w0 = model.table.weight.detach().clone()
    metrics = critic.update_critic(batch)
    g = (w0 - model.table.weight.detach())[:, 0]
    out["grads"] += [g[b:b + n].tolist() for b, n in spans]
    out["metrics"].append({k: [float(x) for x in v] for k, v in metrics.items()})
print(json.dumps(out))
'''


def run_8b_critic(rows, tmp, dp=8, mini_per_gpu=12):
    if not os.path.exists(PY_8B):
        pytest.skip(f"the vagen-lite python is not at {PY_8B}")
    spec = {"rows": rows, "dp": dp, "mini_per_gpu": mini_per_gpu, "clip": CLIP}
    path = os.path.join(tmp, f"critic-{uuid.uuid4().hex[:6]}.json")
    with open(path, "w") as fh:
        json.dump(spec, fh)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                 "RAYON_NUM_THREADS")})
    env.update(TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([PY_8B, "-I", "-B", "-c", SCRIPT_CRITIC_8B.replace("@F061@", F061_ROOT), path],
                          capture_output=True, text=True, timeout=3000, env=env)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def case_8b():
    rows = critic_rows(96, 0, seed=31)
    with work_dir("p3_critic8b") as tmp:
        ref = run_8b_critic(rows, tmp)
    return rows, ref


def _err(a, b):
    return abs(a - b) / max(1.0, abs(b))


def test_t_loss_value_equals_the_8b_update(case_8b):
    """Every prediction's gradient, the objective and the logged vf_loss: vagen_value_loss (V1's way: seq-mean-token-
    mean over mini x n = 96, dp 8, one episode a micro-batch) against F061 dp_critic.update_critic (token-mean, one
    episode a micro-batch, 1 / gradient_accumulation 12, rank by rank, the ranks averaged)."""
    from vagen.q38.trainer import kl_to_8b_unit

    rows, ref = case_8b
    grads, objective, reduced, _ = v1_critic(rows, q38_loss())
    assert len(ref["grads"]) == len(rows)
    worst = 0.0
    for i, (mine, theirs) in enumerate(zip(grads, ref["grads"], strict=True)):
        assert len(mine) == len(theirs) == len(rows[i]["mask"])
        for a, b in zip(mine, theirs):
            worst = max(worst, abs(a - b / 8))  # FSDP averages the 8 ranks' gradients
    assert worst <= 1e-6, worst
    # the gradient sits on the anchors only, on both sides
    anchor_set = set(anchors_of(rows))
    for i, g in enumerate(ref["grads"]):
        for k, x in enumerate(g):
            if (i, k) not in anchor_set:
                assert x == 0.0 and grads[i][k] == 0.0, (i, k)
    assert sum(1 for i, g in enumerate(grads) for k, x in enumerate(g) if x != 0.0) > 0.7 * len(anchor_set)
    # objective: the 8B's = mean over ranks of the micro-batch losses (vf_loss / ga each); the closed form
    obj_8b = sum(sum(m["critic/vf_loss"]) for m in ref["metrics"]) / 8
    assert _err(objective, obj_8b) <= 1e-6 and _err(objective, hand_objective(rows)) <= 1e-6, (objective, obj_8b)
    # the 8B's logged critic/vf_loss (the mean over every micro-batch entry of every rank) = vf_loss_8b
    logged_8b = float(np.mean(sum((m["critic/vf_loss"] for m in ref["metrics"]), [])))
    assert _err(kl_to_8b_unit(reduced["critic/vf_loss"], 96, 12, 96), logged_8b) <= 1e-6
    assert _err(reduced["critic/vf_loss"], 12 * logged_8b) <= 1e-6  # V1's is the true loss, 12 x the 8B's number
    print(f"T-loss value vs F061 dp_critic: 96 episodes, {len(anchor_set)} anchors, max |grad q38 - grad 8B / 8| "
          f"{worst:.3g}; objective q38 {objective:.9g} 8B {obj_8b:.9g} closed form {hand_objective(rows):.9g}; "
          f"critic/vf_loss q38 {reduced['critic/vf_loss']:.9g}, 8B logged {logged_8b:.9g}")


def test_t_loss_readings_against_the_8b_update(case_8b):
    """The pooled in-update readings: anchors, extremes and the pooled EV / MAE / RMSE equal a direct computation over
    all anchors; the 8B's per-rank readings (dp_critic.py:277-299, averaged over the 8 ranks in its log) match where
    they are the same quantity (counts / 8, global extremes)."""
    from vagen.q38.losses import critic_update_readings

    rows, ref = case_8b
    _, _, reduced, _ = v1_critic(rows, q38_loss())
    readings = critic_update_readings(reduced)
    assert not any(k.startswith("critic/raw/") for k in reduced)
    pts = anchors_of(rows)
    p = np.array([rows[i]["vpred"][a] for i, a in pts], dtype=np.float64)
    t = np.array([rows[i]["returns"][a] for i, a in pts], dtype=np.float64)
    e = p - t
    assert readings["critic/update/valid_count"] == len(pts) == sum(np.mean(m["critic/valid_count"]) for m in ref["metrics"])
    assert readings["critic/update/explained_var_valid"] == pytest.approx(1 - e.var() / (t.var() + 1e-5), abs=1e-6)
    assert readings["critic/update/mae_valid"] == pytest.approx(np.abs(e).mean(), abs=1e-6)
    assert readings["critic/update/rmse_valid"] == pytest.approx(math.sqrt((e * e).mean()), abs=1e-6)
    assert readings["critic/vpred_mean"] == pytest.approx(p.mean(), abs=1e-6)
    for name, fn, key8 in (("pred_valid_min", min, "critic/pred_valid_min"), ("pred_valid_max", max, "critic/pred_valid_max"),
                           ("target_valid_min", min, "critic/target_valid_min"),
                           ("target_valid_max", max, "critic/target_valid_max")):
        assert reduced["critic/update/" + name] == pytest.approx(fn(m[key8][0] for m in ref["metrics"]), abs=1e-6)
    clipped = sum(1 for i, a in pts if (min(max(rows[i]["vpred"][a], rows[i]["values"][a] - CLIP),
                                            rows[i]["values"][a] + CLIP) - rows[i]["returns"][a]) ** 2
                  > (rows[i]["vpred"][a] - rows[i]["returns"][a]) ** 2)
    assert clipped > 0 and readings["critic/update/clipfrac"] == pytest.approx(clipped / len(pts))
    assert readings["critic/update/rows_with_anchor"] == 96 and readings["critic/update/rows"] == 96
    # the sentinel positions: every real row's non-anchor answer tokens, and the mean prediction there
    sent = [(i, k) for i, r in enumerate(rows) for k in range(len(r["mask"])) if r["mask"][k] and (i, k) not in set(pts)]
    assert readings["critic/update/sentinel"] == len(sent)
    assert readings["critic/update/sentinel_pred_mean"] == pytest.approx(np.mean([rows[i]["vpred"][k] for i, k in sent]))


def test_t_loss_padding_rows_count_in_the_denominator():
    """72 real rows + 24 padding rows (balanced into the 96-row mini-batch): the padding rows get no gradient, the
    objective is the real rows' sum / 96, as the actor's loss (DESIGN 3.1)."""
    rows = critic_rows(72, 24, seed=32)
    order = torch.randperm(96, generator=torch.Generator().manual_seed(4)).tolist()
    rows = [rows[i] for i in order]
    grads, objective, reduced, _ = v1_critic(rows, q38_loss())
    assert _err(objective, hand_objective(rows)) <= 1e-6
    for r, g in zip(rows, grads, strict=True):
        if not r["real"]:
            assert g == [0.0]
    from vagen.q38.losses import critic_update_readings

    readings = critic_update_readings(reduced)
    assert readings["critic/update/rows_with_anchor"] == 72 and readings["critic/update/rows"] == 96


def test_stock_value_loss_regresses_to_the_sentinel():
    """PEAR B5: verl's own value_loss (VS:158-216) on the same rows masks with response_mask only, so the non-anchor
    answer tokens are pulled to -100 (their gradient pushes the prediction down), and its loss is dominated by them."""
    from verl.workers.utils.losses import value_loss

    rows = critic_rows(96, 0, seed=33, clip_some=False)
    stock = functools.partial(value_loss, config=LOSS_CFG)
    g_stock, obj_stock, _, _ = v1_critic(rows, stock)
    g_mine, obj_mine, _, _ = v1_critic(rows, q38_loss())
    anchor_set = set(anchors_of(rows))
    pulled = [(i, k) for i, r in enumerate(rows) for k in range(len(r["mask"]))
              if r["mask"][k] and (i, k) not in anchor_set]
    # d/dv 0.5 (v + 100)^2 > 0: descent lowers v (0 only where the value clip saturates, v < old - 0.5)
    assert pulled and all(g_stock[i][k] >= 0 for i, k in pulled)
    assert sum(1 for i, k in pulled if g_stock[i][k] > 0) >= 0.9 * len(pulled)
    assert all(g_mine[i][k] == 0.0 for i, k in pulled)
    assert obj_stock > 100 * obj_mine


def test_valid_mask_is_the_8bs():
    """vagen.q38.losses.value_valid_mask = F061 dp_critic.py:231-234 composed with the 8B's compute_value_mask
    (vagen/ray_trainer.py:1126-1138, its own source)."""
    from vagen.q38.losses import value_valid_mask

    compute_value_mask, = rt_functions("_default_eps", "compute_value_mask")[1:]
    returns = torch.tensor([[-100.0, -100.005, -99.98, 0.0, 5.0, -100.0 + 1e-7, 3.5]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1]])
    for dtype in (torch.float32, torch.bfloat16):
        r = returns.to(dtype)
        value_mask = compute_value_mask(SimpleNamespace(batch={"returns": r, "response_mask": mask}))
        sentinel = (r.detach().float() + 100.0).abs() < 1e-2
        f061 = mask.bool() & value_mask.bool() & ~sentinel
        assert torch.equal(value_valid_mask(r, mask), f061), dtype
    assert value_valid_mask(returns, mask).tolist() == [[False, False, True, True, False, False, True]]


def test_micro_batch_of_padding_rows_only():
    """No anchor in the micro-batch: loss 0 (no error, unlike F061 dp_critic.py:237-238), still in the graph, zero
    gradient; the raw counts 0, the extremes +-inf (ignored by the min / max over the batch)."""
    from vagen.q38.losses import RAW_PREFIX, UPDATE_PREFIX, vagen_value_loss

    rows = critic_rows(0, 1, seed=1)
    leaves = leaves_of(rows)
    data = micro_data(rows, 8, 96, 0)
    out = {"values": torch.nested.as_nested_tensor(leaves, layout=torch.jagged)}
    loss, metrics = vagen_value_loss(LOSS_CFG, out, data, global_batch_size=96, max_rows_per_micro=1)
    assert float(loss) == 0.0 and loss.requires_grad
    loss.backward()
    assert float(leaves[0].grad.abs().sum()) == 0.0
    assert metrics[RAW_PREFIX + "anchors"].values == [0.0] and metrics[RAW_PREFIX + "rows"].values == [8.0]
    assert metrics[UPDATE_PREFIX + "pred_valid_min"].values == [math.inf]
    assert metrics[UPDATE_PREFIX + "pred_valid_max"].values == [-math.inf]


def test_guards():
    from vagen.q38.losses import vagen_value_loss

    rows = critic_rows(2, 0, seed=2)
    out = {"values": torch.nested.as_nested_tensor(leaves_of(rows), layout=torch.jagged)}
    with pytest.raises(RuntimeError, match="micro-batch of 2 sequences"):
        vagen_value_loss(LOSS_CFG, out, micro_data(rows, 8, 96, 5), global_batch_size=96, max_rows_per_micro=1)
    one = rows[:1]
    out1 = {"values": torch.nested.as_nested_tensor(leaves_of(one), layout=torch.jagged)}
    with pytest.raises(ValueError, match="global_batch_size"):
        vagen_value_loss(LOSS_CFG, out1, micro_data(one, 8, 192, 5), global_batch_size=96, max_rows_per_micro=1)
    with pytest.raises(ValueError, match="loss_scale_factor"):
        vagen_value_loss(SimpleNamespace(cliprange_value=0.5, loss_agg_mode="seq-mean-token-mean", loss_scale_factor=4.0),
                         out1, micro_data(one, 8, 96, 5), global_batch_size=96)
    with pytest.raises(ValueError, match="loss_agg_mode"):
        vagen_value_loss(SimpleNamespace(cliprange_value=0.5, loss_agg_mode="token-mean", loss_scale_factor=None),
                         out1, micro_data(one, 8, 96, 5), global_batch_size=96)


def test_readings_pool_the_passes():
    """Three passes over one batch (S-PPO3): the trainer's reduced raw sums are the mean over the passes, so the
    pooled readings are the readings over all passes' predictions together (as dp_critic pooled them)."""
    from vagen.q38.losses import critic_update_readings

    rows = critic_rows(96, 0, seed=34)
    _, _, reduced, worker = v1_critic(rows, q38_loss(), epochs=3)
    assert all(len(v) == 3 for v in worker.values())
    r3 = critic_update_readings(dict(reduced))
    _, _, reduced1, _ = v1_critic(rows, q38_loss(), epochs=1)
    r1 = critic_update_readings(dict(reduced1))
    for k in r1:  # the same predictions every pass here (no optimizer step): the same readings
        assert r3[k] == pytest.approx(r1[k], rel=1e-9, abs=1e-12), k


# ======================================================================================================================
# the driver's readings of a turn_gae batch
# ======================================================================================================================


def _padded(rows, name, fill):
    n = max(len(r["mask"]) for r in rows)
    return torch.tensor([r[name] + [fill] * (n - len(r[name])) for r in rows], dtype=torch.float32)


def test_batch_readings_are_the_8bs_and_split_by_staleness():
    from vagen.q38.losses import critic_batch_readings

    (compute_valid_return_metrics,) = rt_functions("_default_eps", "compute_value_mask",
                                                   "compute_valid_return_metrics")[2:]
    rows = critic_rows(40, 0, seed=35)
    values, returns = _padded(rows, "vpred", 0.0), _padded(rows, "returns", IGNORE)
    mask = _padded(rows, "mask", 0).to(torch.int64)
    adv = torch.randn(values.shape, generator=torch.Generator().manual_seed(9)) * mask
    lags = [0] * 25 + [1] * 13 + [None, 2]
    mine = critic_batch_readings(values, returns, mask, adv, lags)
    theirs = compute_valid_return_metrics(SimpleNamespace(batch={"returns": returns, "response_mask": mask.bool(),
                                                                 "advantages": adv, "values": values}))
    assert set(theirs) <= set(mine), sorted(set(theirs) - set(mine))
    for k, v in theirs.items():
        assert mine[k] == pytest.approx(v, rel=1e-6, abs=1e-7), k
    for b, sl in ((0, slice(0, 25)), (1, slice(25, 38))):
        sub = critic_batch_readings(values[sl], returns[sl], mask[sl], adv[sl], [b] * (sl.stop - sl.start))
        ref = compute_valid_return_metrics(SimpleNamespace(batch={"returns": returns[sl], "response_mask": mask[sl].bool(),
                                                                  "values": values[sl]}))
        assert mine[f"vagen/critic/rows_s{b}"] == sl.stop - sl.start
        assert mine[f"vagen/critic/ev_s{b}"] == pytest.approx(ref["critic/values/explained_var_valid"], rel=1e-6)
        assert mine[f"vagen/critic/ev_s{b}"] == pytest.approx(sub["critic/values/explained_var_valid"], rel=1e-6)
        per_row = []
        for r in rows[sl]:
            terms = [(r["vpred"][a] - r["returns"][a]) ** 2 for a in range(len(r["mask"]))
                     if r["mask"][a] and r["returns"][a] != IGNORE]
            per_row.append(0.5 * sum(terms) / len(terms))
        assert mine[f"vagen/critic/vf_loss_s{b}"] == pytest.approx(float(np.mean(per_row)), rel=1e-6)
    assert mine["vagen/critic/rows_s_other"] == 2


# ======================================================================================================================
# the trainer (fake TransferQueue and worker groups, test_trainer.py's fakes)
# ======================================================================================================================


def turn_ppo_config(tmp_path, **over):
    cfg = turn_ppo_critic(arm_config("turn_gae", tmp_path, **{"critic.enable": True, "trainer.critic_warmup": 11}))
    for k, v in over.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


def _parent_with(cls, name):
    from vagen.q38.trainer import VagenTrainerMixin

    mro = cls.__mro__
    return next(c for c in mro[mro.index(VagenTrainerMixin) + 1:] if name in c.__dict__)


def test_setup_swaps_the_value_loss(monkeypatch, tmp_path):
    from verl.workers.utils.losses import value_loss

    from vagen.q38.losses import ValueLossConfig, vagen_value_loss

    calls = []

    class WG:
        def set_loss_fn(self, fn):
            calls.append(fn)

    def parent_setup(self):  # what TB:344-350 does for the critic
        self.critic_wg = WG()
        self.critic_wg.set_loss_fn(functools.partial(value_loss, config=None))

    t = make_trainer(cfg=turn_ppo_config(tmp_path))
    monkeypatch.setattr(_parent_with(type(t), "_setup"), "_setup", parent_setup)
    t.use_critic = True
    t._setup()
    assert len(calls) == 2 and calls[0].func is value_loss and calls[1].func is vagen_value_loss
    kw = calls[1].keywords
    assert kw["global_batch_size"] == 96 and kw["max_rows_per_micro"] == 1 and kw["ignore_value"] == -100.0
    assert kw["config"] == ValueLossConfig(cliprange_value=0.5, loss_agg_mode="seq-mean-token-mean")
    import pickle

    pickle.loads(pickle.dumps(calls[1]))  # set_loss_fn sends it to every worker
    calls.clear()
    t.use_critic = False
    t._setup()
    assert [c.func for c in calls] == [value_loss]
    for over, msg in (({"critic.ppo_micro_batch_size_per_gpu": 2}, "micro_batch"), ({"critic.use_dynamic_bsz": True}, "dynamic"),
                      ({"critic.loss_agg_mode": "token-mean"}, "loss_agg_mode")):
        t = make_trainer(cfg=turn_ppo_config(tmp_path, **over))
        t.use_critic = True
        with pytest.raises(ValueError, match=msg):
            t._value_loss_fn()
    t = make_trainer(cfg=turn_ppo_config(tmp_path, **{"algorithm.adv_estimator": "grpo"}))
    with pytest.raises(ValueError, match="turn_gae"):
        t._value_loss_fn()


class FakeCriticWG:
    """compute_values writes values over (prompt + response) as the critic worker does; train_mini_batch returns what
    the worker's train_mini_batch returns, computed by v1_critic with the loss the trainer set."""

    def __init__(self, fake, rows_by_key):
        self.fake, self.rows = fake, rows_by_key
        self.seen = []
        self.loss_fn = None

    def set_loss_fn(self, fn):
        self.loss_fn = fn

    def infer_batch(self, batch):
        self.seen.append(("infer", dict(batch.extra_info)))
        for k in batch.keys:
            f = self.fake.fields[k]
            p, r = len(f["prompts"]), len(f["responses"])
            full = torch.full((p + r,), 9.0)
            full[p - 1: p + r - 1] = torch.tensor(self.rows[k]["values"], dtype=torch.float32)
            f["values"] = full
        return SimpleNamespace(futures=[])

    def train_mini_batch(self, batch):
        self.seen.append(("train", dict(batch.extra_info)))
        rows = []
        for k in batch.keys:
            f = self.fake.fields[k]
            rows.append({"mask": f["response_mask"].tolist(), "returns": f["returns"].tolist(),
                         "values": f["values"].tolist(), "vpred": self.rows[k]["vpred"], "prompt_len": len(f["prompts"]),
                         "real": True})
        _, _, _, worker = v1_critic(rows, self.loss_fn, mini_n=batch.extra_info["global_batch_size"])
        metrics = dict(worker)
        metrics.update({"mfu": [0.1], "loss": [0.0], "grad_norm": [1.0], "lr": [2e-5]})
        return SimpleNamespace(get=lambda: {"metrics": metrics})


def critic_step(monkeypatch, tmp_path, n_pad=8):
    """One turn_gae step through the mixin with a critic: old log-probs, values, advantages, critic update."""
    from verl.trainer.ppo.v1 import trainer_base

    monkeypatch.setattr(trainer_base.ray, "get", lambda x: x)
    fake, batch, eps, metrics = run_adv(monkeypatch, "turn_gae", tmp_path, n_pad=n_pad, cfg=turn_ppo_config(tmp_path))
    return fake, batch, eps, metrics


def test_compute_values_and_update_critic(monkeypatch, tmp_path):
    import transfer_queue

    from verl.trainer.ppo.v1 import trainer_base

    from vagen.q38.tests.test_trainer import FakeReplayBuffer, FakeWorkerGroup
    from vagen.q38.tests.test_turn_arms import GROUPS, ONES, turn_batch

    monkeypatch.setattr(trainer_base.ray, "get", lambda x: x)
    fake = TrainerTQ()
    monkeypatch.setattr(transfer_queue, "kv_batch_get", fake.kv_batch_get)
    monkeypatch.setattr(transfer_queue, "kv_batch_put", fake.kv_batch_put)
    batch, eps = turn_batch(fake, GROUPS, n_pad=72)  # 24 real rows + 72 padding rows = one 96-row critic mini-batch
    g = torch.Generator().manual_seed(17)
    rows_by_key = {}
    for k in batch.keys:
        n = len(fake.fields[k]["responses"])
        v = (torch.randn(n, generator=g) * 1.5).tolist()
        rows_by_key[k] = {"values": v, "vpred": (torch.tensor(v) + torch.randn(n, generator=g) * 0.4).tolist()}
        del fake.fields[k]["values"]
    t = make_trainer(cfg=turn_ppo_config(tmp_path), wg=FakeWorkerGroup(fake), global_steps=STEP)
    t.use_critic = True
    cwg = FakeCriticWG(fake, rows_by_key)
    t.critic_wg = cwg
    t._set_value_loss()
    t._compute_old_log_prob(batch, {})
    metrics = {}
    t._compute_values(batch, metrics)
    assert cwg.seen[-1] == ("infer", {**cwg.seen[-1][1], "use_dynamic_bsz": False, "micro_batch_size_per_gpu": 1})
    assert "use_dynamic_bsz" not in batch.extra_info and "micro_batch_size_per_gpu" not in batch.extra_info
    for k in batch.keys:  # the parent's write-back: the response form
        assert fake.fields[k]["values"].tolist() == pytest.approx(rows_by_key[k]["values"])
    t.replay_buffer = FakeReplayBuffer(ONES)
    t._compute_advantage(batch, metrics)
    t._update_critic(batch, metrics)
    kind, info = cwg.seen[-1]
    assert kind == "train" and info["use_dynamic_bsz"] is False and info["micro_batch_size_per_gpu"] == 1
    assert info["global_batch_size"] == 96 and info["mini_batch_size"] == 96 and info["epochs"] == 1
    assert "use_dynamic_bsz" not in batch.extra_info and "micro_batch_size_per_gpu" not in batch.extra_info
    assert not any(k.startswith("critic/raw/") for k in metrics)
    # the update's readings against a direct computation over the real rows' anchors
    real = [k for k, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding")]
    pts = [(k, a) for k in real for a, (m, r) in enumerate(zip(fake.fields[k]["response_mask"].tolist(),
                                                              fake.fields[k]["returns"].tolist()))
           if m and r != -100.0]
    p = np.array([rows_by_key[k]["vpred"][a] for k, a in pts])
    tt = np.array([float(fake.fields[k]["returns"][a]) for k, a in pts])
    e = p - tt
    assert metrics["critic/update/valid_count"] == len(pts)
    assert metrics["critic/update/explained_var_valid"] == pytest.approx(1 - e.var() / (tt.var() + 1e-5), abs=1e-6)
    per_row = {}
    for k, a in pts:
        v, old, ret = rows_by_key[k]["vpred"][a], float(fake.fields[k]["values"][a]), float(fake.fields[k]["returns"][a])
        vc = min(max(v, old - CLIP), old + CLIP)
        per_row.setdefault(k, []).append(max((v - ret) ** 2, (vc - ret) ** 2))
    true_loss = sum(0.5 * sum(x) / len(x) for x in per_row.values()) / 96
    assert metrics["critic/vf_loss"] == pytest.approx(true_loss, rel=1e-6)
    assert metrics["critic/vf_loss_8b"] == pytest.approx(true_loss * 96 / (12 * 24), rel=1e-6)
    assert metrics["vagen/critic/update_rows"] == 96 and metrics["vagen/critic/update_real_rows"] == 24
    assert metrics["perf/mfu/critic"] == 0.1


def test_turn_branch_readings(monkeypatch, tmp_path):
    """The turn_gae branch reports the critic's fit on the batch's real rows: the 8B's compute_valid_return_metrics
    names (its own code on the same rows) and EV / vf_loss by staleness 0 / 1 (GROUPS: ga, gc staleness 0, gb 1)."""
    from vagen.q38.losses import critic_batch_readings

    (compute_valid_return_metrics,) = rt_functions("_default_eps", "compute_value_mask",
                                                   "compute_valid_return_metrics")[2:]
    fake, batch, eps, metrics = critic_step(monkeypatch, tmp_path)
    real = [k for k, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding")]
    n = max(len(fake.fields[k]["response_mask"]) for k in real)

    def padded(name, fill):
        return torch.stack([torch.cat([fake.fields[k][name].float(),
                                       torch.full((n - len(fake.fields[k][name]),), fill)]) for k in real])

    values, returns, mask, adv = padded("values", 0.0), padded("returns", -100.0), padded("response_mask", 0), \
        padded("advantages", 0.0)
    lags = [0 if k.startswith(("ga", "gc")) else 1 for k in real]
    want = critic_batch_readings(values, returns, mask.long(), adv, lags)
    for k, v in want.items():
        assert metrics[k] == pytest.approx(v, rel=1e-6, abs=1e-7), k
    theirs = compute_valid_return_metrics(SimpleNamespace(batch={"returns": returns, "response_mask": mask.bool(),
                                                                 "advantages": adv, "values": values}))
    for k, v in theirs.items():
        assert metrics[k] == pytest.approx(v, rel=1e-6, abs=1e-7), k
    assert metrics["vagen/critic/rows_s0"] == 16 and metrics["vagen/critic/rows_s1"] == 8
    assert all(math.isfinite(metrics[f"vagen/critic/{x}_s{b}"]) for x in ("ev", "vf_loss") for b in (0, 1))
    assert metrics["critic/advantages/valid_mean"] == pytest.approx(0.0, abs=1e-5)  # whitened over the real rows


def test_turn_branch_refuses_a_real_row_without_anchor(monkeypatch, tmp_path):
    import transfer_queue

    from verl.trainer.ppo.v1 import trainer_base

    from vagen.q38.tests.test_trainer import FakeReplayBuffer, FakeWorkerGroup
    from vagen.q38.tests.test_turn_arms import GROUPS, ONES, turn_batch

    monkeypatch.setattr(trainer_base.ray, "get", lambda x: x)
    fake = TrainerTQ()
    monkeypatch.setattr(transfer_queue, "kv_batch_get", fake.kv_batch_get)
    monkeypatch.setattr(transfer_queue, "kv_batch_put", fake.kv_batch_put)
    batch, _ = turn_batch(fake, GROUPS, n_pad=0)
    k = batch.keys[3]  # an episode with no generated turn: no answer token, no turn record
    f = fake.fields[k]
    r = len(f["responses"])
    f["response_mask"] = torch.zeros(r, dtype=torch.int64)
    f["rm_scores"] = torch.zeros(r)
    f["extra_fields"] = dict(f["extra_fields"], turn_rewards={"spans": [], "rewards": []})
    t = make_trainer(cfg=turn_ppo_config(tmp_path), wg=FakeWorkerGroup(fake), global_steps=STEP)
    t.use_critic = True
    t._compute_old_log_prob(batch, {})
    t.replay_buffer = FakeReplayBuffer(ONES)
    with pytest.raises(RuntimeError, match="no value target"):
        t._compute_advantage(batch, {})


def test_compute_metrics_drops_the_sentinel_readings(monkeypatch, tmp_path):
    from vagen.q38.losses import SENTINEL_DOMINATED

    def parent(self, batch, metrics, timing_raw, global_steps, epoch):
        metrics.update({k: np.float64(-90.0) for k in SENTINEL_DOMINATED})
        metrics["critic/score/mean"] = np.float64(1.5)

    for estimator, dropped in (("turn_gae", True), ("grpo", False)):
        cfg = turn_ppo_config(tmp_path) if estimator == "turn_gae" else arm_config("grpo", tmp_path)
        t = make_trainer(cfg=cfg)
        monkeypatch.setattr(_parent_with(type(t), "_compute_metrics"), "_compute_metrics", parent)
        metrics = {}
        t._compute_metrics(None, metrics, {}, 1, 0)
        assert metrics["critic/score/mean"] == 1.5 and type(metrics["critic/score/mean"]) is float
        assert all((k in metrics) != dropped for k in SENTINEL_DOMINATED), estimator
