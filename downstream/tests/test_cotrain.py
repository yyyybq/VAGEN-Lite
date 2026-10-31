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
    assert name == "traj" and int((inputs["labels"] != CT.IGNORE).sum()) == 2 and tb.too_long == 0
    short = CT.TextBatches(FakeProcessor(), rank=0, world=2, seed=1, max_tokens=3,
                           sources={"traj": tmp_path / "traj.jsonl", "qa": tmp_path / "qa.jsonl"})
    with pytest.raises(RuntimeError):
        short.at(0)  # every sample is longer than three tokens
    assert short.too_long == 100  # each one passed over is counted
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

        def forward(self, input_ids=None, labels=None, use_cache=None, logits_to_keep=None, **kw):
            """Logits over a two-word vocabulary at every position (word 1 favoured by head(last(1))), only at
            logits_to_keep's positions when given, as transformers' models do."""
            self.kept = logits_to_keep
            z = head(last(torch.ones(1, 1)))
            rows = torch.cat([z * 0.0, z], dim=-1).expand(input_ids.shape[1], 2)
            logits = rows if logits_to_keep is None else rows[logits_to_keep]
            return type("O", (), {"logits": logits.unsqueeze(0), "loss": None})()

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
    # three tokens, the answer being the last two (the label of position t + 1 is predicted at position t)
    batch = type("B", (), {"at": lambda s, step: ("qa", 4, {"input_ids": torch.tensor([[7, 8, 9]]),
                                                             "labels": torch.tensor([[-100, 1, 0]])})})()
    step = CE.make_train_step(T, {"lambda": 0.1, "diag_every": 1}, lambda tr: batch)
    m = step(trainer, batch_vla=None)
    z = float(head(last(torch.ones(1, 1))))
    row = torch.tensor([[0.0, z]])
    text = float((torch.nn.functional.cross_entropy(row, torch.tensor([1]))
                  + torch.nn.functional.cross_entropy(row, torch.tensor([0]))) / 2)
    assert vlm.kept.tolist() == [0, 1]  # logits only where the next token is an answer token
    assert trainer.accelerator.losses == [pytest.approx(4.0 + 0.1 * text)]
    assert m["action_dit_loss"] == pytest.approx(4.0) and m["text_loss_qa"] == pytest.approx(text)
    assert m["grad_lm_head_text"] > 0 and m["grad_lm_head_action"] == 0.0 and m["text_index"] == 4
    assert m["grad_last_layer_action"] == 0.0 and m["grad_last_layer_text"] > 0
    assert m["text_len_max"] == 3 and "mem_peak_gib_max" in m and m["text_too_long_total"] == 0
    assert not getattr(vlm, "_c1_checkpointing", False)  # off unless asked

    calls = []
    vlm.gradient_checkpointing_enable = lambda **kw: calls.append(kw)
    vlm.config = type("Cfg", (), {"use_cache": True})()
    step = CE.make_train_step(T, {"lambda": 0.1, "gradient_checkpointing": True}, lambda tr: batch)
    first = step(trainer, batch_vla=None)
    second = step(trainer, batch_vla=None)
    assert calls == [{"gradient_checkpointing_kwargs": {"use_reentrant": False}}]  # once, before the first step
    assert vlm.config.use_cache is False
    assert "checkpointing_modules" in first and "checkpointing_modules" not in second

    # the user's option B (2026-10-31): only the text forward checkpoints
    blk = torch.nn.Module()
    blk.gradient_checkpointing = False
    seen = []
    vlm2 = VLM()
    vlm2.blk = blk
    vlm2.config = type("Cfg", (), {"use_cache": True})()

    def enable(**kw):
        blk.gradient_checkpointing = True

    vlm2.gradient_checkpointing_enable = enable
    text_forward = vlm2.forward

    def watched(*a, **kw):
        seen.append(("text", blk.gradient_checkpointing))
        return text_forward(*a, **kw)

    vlm2.forward = watched

    class Model2(Model):
        def forward(self, batch):
            seen.append(("action", blk.gradient_checkpointing))
            return super().forward(batch)

    vla2 = type("V", (), {"qwen_vl_interface": type("Q", (), {"model": vlm2})()})()
    T2 = type("T", (), {"_unwrap_model": staticmethod(lambda acc, m: vla2)})
    trainer.model = Model2()
    step = CE.make_train_step(T2, {"lambda": 0.1, "gradient_checkpointing": True, "checkpoint_scope": "text"},
                              lambda tr: batch)
    assert step(trainer, batch_vla=None)["checkpointing_modules"] == 1
    step(trainer, batch_vla=None)
    assert seen == [("action", False), ("text", True)] * 2


def test_answer_loss_equals_the_models_own_shifted_loss():
    """2026-10-31: answer_loss makes logits only where the next token carries a label; the number is the standard
    shifted mean cross-entropy over the labelled tokens."""
    torch = pytest.importorskip("torch")
    import cotrain_entry as CE

    torch.manual_seed(0)
    table = torch.randn(5, 4)  # position -> logits over a four-word vocabulary

    class VLM:
        def __call__(self, input_ids=None, labels=None, use_cache=None, logits_to_keep=None, **kw):
            logits = table[: input_ids.shape[1]]
            if labels is not None:  # the model's own loss: shift, ignore -100, mean
                loss = torch.nn.functional.cross_entropy(logits[:-1], labels[0, 1:], ignore_index=-100)
                return type("O", (), {"loss": loss, "logits": logits.unsqueeze(0)})()
            return type("O", (), {"logits": logits[logits_to_keep].unsqueeze(0), "loss": None})()

    inputs = {"input_ids": torch.zeros(1, 5, dtype=torch.long), "labels": torch.tensor([[-100, -100, 2, 3, 1]])}
    own = VLM()(input_ids=inputs["input_ids"], labels=inputs["labels"]).loss
    assert float(CE.answer_loss(VLM(), inputs)) == pytest.approx(float(own))
    assert "labels" in inputs  # the caller's dict is left as it was


# ---------------------------------------------------------------------------------------- config and check
def test_cotrain_option_parsing():
    import make_experiment as ME

    co = ME.parse_cotrain("lambda=0.1,diag_every=10,skip_final_save=1")
    assert co["lambda"] == 0.1 and co["diag_every"] == 10 and co["skip_final_save"] is True
    assert ME.parse_cotrain("lambda=0.1,gradient_checkpointing=1")["gradient_checkpointing"] is True
    assert ME.parse_cotrain("lambda=0.1,gradient_checkpointing=1,checkpoint_scope=text")["checkpoint_scope"] == "text"
    with pytest.raises(ValueError):
        ME.parse_cotrain("lambda=0.1,checkpoint_scope=vision")
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
