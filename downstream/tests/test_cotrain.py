"""C1 co-training (2026-10-30): which text sample each process takes, the labels, the patched training step (a tiny
model on the CPU), the --cotrain config and the short run's check."""
import contextlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "x"))
import cotrain_check as CC  # noqa: E402
import cotrain_text as CT  # noqa: E402
import x4_ppl  # noqa: E402


def test_even_ranks_read_trajectories_odd_ranks_qa():
    assert [CT.source_of(r, 8) for r in range(8)] == [("traj", 0, 4), ("qa", 0, 4), ("traj", 1, 4), ("qa", 1, 4),
                                                      ("traj", 2, 4), ("qa", 2, 4), ("traj", 3, 4), ("qa", 3, 4)]


def test_each_record_once_per_pass_and_the_same_after_a_resume():
    sizes = {"traj": 10, "qa": 6}
    seen = [CT.pick(step, rank, 8, sizes, seed=1) for step in range(5) for rank in range(0, 8, 2)]  # 20 traj picks
    first_pass = [i for _, i in seen[:10]]
    assert sorted(first_pass) == list(range(10)) and sorted(i for _, i in seen[10:]) == list(range(10))
    assert first_pass != [i for _, i in seen[10:]]  # each pass in its own order
    assert CT.pick(3, 5, 8, sizes, seed=1) == CT.pick(3, 5, 8, sizes, seed=1)  # a function of the step alone
    assert CT.pick(3, 5, 8, sizes, seed=1)[0] == "qa"
    assert CT.pick(3, 4, 8, sizes, seed=1, offset=1) == CT.pick(4, 4, 8, sizes, seed=1)  # "the next position"


def test_only_answer_tokens_carry_labels():
    S, E, A, N = x4_ppl.IM_START, x4_ppl.IM_END, x4_ppl.ASSISTANT, x4_ppl.NEWLINE
    ids = [S, 9, N, 5, 6, E, N, S, A, N, 11, 12, E, N, S, 9, N, 7, E, S, A, N, 13, E]
    labels = CT.labels_for(ids)
    kept = [i for i, x in enumerate(labels) if x != CT.IGNORE]
    assert kept == [10, 11, 12, 22, 23] and [labels[i] for i in kept] == [11, 12, E, 13, E]


def test_video_records_are_left_out(tmp_path):
    f = tmp_path / "qa.jsonl"
    f.write_text(json.dumps({"messages": [{"role": "user", "content": "<image>q"}], "images": ["/a.png"]}) + "\n"
                 + json.dumps({"messages": [], "images": [], "videos": [["/f1.png"]]}) + "\n")
    assert len(CT.load_records(f)) == 1 and CT.load_records(f)[0]["images"] == ["/a.png"]


class FakeProcessor:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
        return json.dumps(chat)

    def __call__(self, text, images=None, return_tensors="pt"):
        import torch

        S, E, A, N = x4_ppl.IM_START, x4_ppl.IM_END, x4_ppl.ASSISTANT, x4_ppl.NEWLINE
        n = len(text[0])
        ids = [S, 9, N] + [5] * (n % 7) + [E, S, A, N, 11, E] if "answer" in text[0] else [S, 9, N, 5, E]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


def test_text_batches_skip_samples_without_answers(tmp_path):
    pytest.importorskip("torch")
    for name, content in (("traj", "answer"), ("qa", "none")):
        rec = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": content}], "images": []}
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(rec) + "\n")
    tb = CT.TextBatches(FakeProcessor(), rank=0, world=2, seed=1,
                        sources={"traj": tmp_path / "traj.jsonl", "qa": tmp_path / "qa.jsonl"})
    name, i, inputs = tb.at(0)
    assert name == "traj" and int((inputs["labels"] != CT.IGNORE).sum()) == 2
    tb_qa = CT.TextBatches(FakeProcessor(), rank=1, world=2, seed=1,
                           sources={"traj": tmp_path / "traj.jsonl", "qa": tmp_path / "qa.jsonl"})
    with pytest.raises(RuntimeError):
        tb_qa.at(0)  # the QA stand-in has no answer tokens anywhere


# ------------------------------------------------------------------------------------------- the training step
def test_train_step_adds_lambda_times_text_loss_once():
    torch = pytest.importorskip("torch")
    import cotrain_entry as CE

    w_action = torch.nn.Parameter(torch.tensor(2.0))
    head = torch.nn.Linear(1, 1, bias=False)
    last = torch.nn.Linear(1, 1, bias=False)

    class VLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = head
            self.model = type("M", (), {"language_model": type("L", (), {"layers": [last]})()})()

        def forward(self, input_ids=None, labels=None, use_cache=None, **kw):
            x = torch.ones(1, 1)
            return type("O", (), {"loss": (head(last(x)) - 3.0).pow(2).sum()})()

    vlm = VLM()
    vla = type("V", (), {"qwen_vl_interface": type("Q", (), {"model": vlm})()})()

    class Acc:
        device, sync_gradients, losses = "cpu", True, []

        def accumulate(self, model):
            return contextlib.nullcontext()

        def backward(self, loss):
            self.losses.append(float(loss.detach()))
            loss.backward()

        def clip_grad_norm_(self, params, v):
            pass

        def gather(self, t):
            return t

    class Model:
        def forward(self, batch):
            return {"action_loss": (w_action * 1.0) ** 2}

        def parameters(self):
            return iter([w_action])

    T = type("T", (), {"_unwrap_model": staticmethod(lambda acc, m: vla)})
    trainer = type("Trainer", (), {})()
    trainer.accelerator, trainer.model, trainer.completed_steps = Acc(), Model(), 0
    trainer.optimizer = type("O", (), {"zero_grad": lambda s: None, "step": lambda s: None})()
    trainer.lr_scheduler = type("S", (), {"step": lambda s: None})()
    trainer.config = type("C", (), {"trainer": type("Tr", (), {"gradient_clipping": 1.0})()})()
    batch = type("B", (), {"at": lambda s, step: ("qa", 4, {"input_ids": torch.tensor([[1]]),
                                                             "labels": torch.tensor([[1]])})})()
    step = CE.make_train_step(T, {"lambda": 0.1, "diag_every": 1}, lambda tr: batch)
    m = step(trainer, batch_vla=None)
    text = float((head(last(torch.ones(1, 1))) - 3.0).pow(2).sum())
    assert trainer.accelerator.losses == [pytest.approx(4.0 + 0.1 * text)]
    assert m["action_dit_loss"] == pytest.approx(4.0) and m["text_loss_qa"] == pytest.approx(text)
    assert m["grad_lm_head_text"] > 0 and m["grad_lm_head_action"] == 0.0 and m["text_index"] == 4
    assert m["grad_last_layer_action"] == 0.0 and m["grad_last_layer_text"] > 0
    assert not getattr(vlm, "_c1_checkpointing", False)  # off unless asked

    calls = []
    vlm.gradient_checkpointing_enable = lambda **kw: calls.append(kw)
    vlm.config = type("Cfg", (), {"use_cache": True})()
    step = CE.make_train_step(T, {"lambda": 0.1, "gradient_checkpointing": True}, lambda tr: batch)
    step(trainer, batch_vla=None)
    step(trainer, batch_vla=None)
    assert calls == [{"gradient_checkpointing_kwargs": {"use_reentrant": False}}]  # once, before the first step
    assert vlm.config.use_cache is False


# ---------------------------------------------------------------------------------------- config and check
def test_cotrain_option_parsing():
    import make_experiment as ME

    co = ME.parse_cotrain("lambda=0.1,diag_every=10,skip_final_save=1")
    assert co["lambda"] == 0.1 and co["diag_every"] == 10 and co["skip_final_save"] is True
    assert ME.parse_cotrain("lambda=0.1,gradient_checkpointing=1")["gradient_checkpointing"] is True
    assert co["sources"] == CT.SOURCES and "1:1" in co["mix"]
    assert ME.parse_cotrain("") == {}
    with pytest.raises(ValueError):
        ME.parse_cotrain("diag_every=10")
    with pytest.raises(ValueError):
        ME.parse_cotrain("lambda=0.1,lr=3")


def test_short_run_check(tmp_path):
    exp, ref = tmp_path / "exp", tmp_path / "ref"
    (exp / "run").mkdir(parents=True)
    (ref / "run").mkdir(parents=True)
    rows = []
    for s in range(1, 201):
        r = {"step": s, "action_dit_loss": 1.0 - s / 400, "text_loss": 2.0 - s / 300, "text_loss_traj": 2.0 - s / 300,
             "text_loss_qa": 1.5 - s / 500, "timing/model": 0.9}
        if s % 10 == 0:
            r.update(grad_lm_head_text=0.5, grad_lm_head_action=0.0, grad_last_layer_text=0.2, grad_last_layer_action=0.4)
        rows.append(r)
    (exp / "run/metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ref / "run/metrics.jsonl").write_text("".join(json.dumps({"step": s, "timing/model": 0.7}) + "\n" for s in range(1, 100)))
    (exp / "gpu.csv").write_text("2026/10/30 10:00:00.000, 0, 61000 MiB, 90 %\n2026/10/30 10:00:10.000, 1, 63500 MiB, 80 %\n")
    res = CC.check(exp, ref)
    assert res["passed"] == {"1_losses_fall": True, "2_lm_head_gradient": True}
    assert res["4_memory_peak_mib"] == 63500 and res["5_last_layer_grad_ratio"]["mean"] == pytest.approx(0.5)
    assert res["3_seconds_per_step"]["slower_by"] == pytest.approx(0.9 / 0.7 - 1)
    assert res["2_lm_head_gradient"]["diagnosed_steps"] == 20
