"""Phase 3, the 27B value network's model class on CPU (no weights; DESIGN_ASYNC 5.5 S8 item a before the GPU).

verl's FSDP engine builds the critic (model_type value_model) with AutoModelForTokenClassification from the CausalLM
checkpoint (verl/verl/workers/engine/fsdp/transformer_impl.py:302-317, verl/verl/utils/model.py:641-682).  For a
qwen3_5 checkpoint transformers 5.10.4 maps that to Qwen3_5ForTokenClassification = GenericForTokenClassification over
Qwen3_5Model (vision tower + language model) with a 1-unit ``score`` head.  On a tiny random config of the 4B's
(Qwen3.5-4B hub snapshot 851bf6e8: the token ids, rope and vision patching; 2 decoder layers, one gated-delta-net and
one full attention, a 1-block vision tower), with the gated-delta-net's torch fallbacks (FLA's Triton kernels need a
GPU):

- the class, its weights from a Qwen3_5ForConditionalGeneration checkpoint (only ``score`` new, only ``lm_head``
  unused), and the 1-unit head;
- the multimodal inputs our agent loop produces, built as the TransferQueue writer builds them (test_agent_loop.py's
  tq_fields: AgentLoopWorker._compute_multi_modal_inputs / _compute_position_ids on the 4B processor: pixel_values,
  image_grid_thw, images_seqlens, position_ids (4, L)), through the engine's own input path (FSDPEngineWithLMHead.
  prepare_model_inputs, use_remove_padding: one packed row, position_ids (4, 1, N)) and verl's qwen3_5 monkey patch,
  then FSDPEngineWithValueHead.prepare_model_outputs and vagen_value_loss with a backward: finite values, equal to the
  unpatched padded forward; gradients reach the head and the language model;
- the reason for one episode a critic micro-batch: the engine passes cu_seqlens only to a forward that names it
  (transformer_impl.py:610); this class's does not, and two episodes packed into one row without cu_seqlens give the
  second one other values than alone (the gated-delta-net layer carries its state across the boundary);
- verl's load_valuehead_model asks for flash_attention_2 whatever the config says (utils/model.py:649): the critic's
  attention differs from the actor's eager (an open question of phase 3).

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_critic_model_cpu.py -q -s -p no:cacheprovider
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vagen.q38.tests.test_agent_loop import MODEL_PATHS, load_model, noise_image

MODEL = "qwen35_4b"


@pytest.fixture(scope="module")
def cpu_gdn():
    """The gated-delta-net layers on their torch fallbacks (module symbols None before the model is built), restored
    after the module."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    mp = pytest.MonkeyPatch()
    for name in ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        mp.setattr(M, name, None)
    yield M
    mp.undo()


def tiny_config(layer_types=("linear_attention", "full_attention")):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_PATHS[MODEL])
    assert cfg.model_type == "qwen3_5"
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_hidden_layers, t.layer_types, t.full_attention_interval = len(layer_types), list(layer_types), 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 64
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
    rope = dict(t.rope_parameters)
    rope["mrope_section"] = [3, 3, 2]  # head_dim 64 x partial_rotary_factor 0.25 / 2
    t.rope_parameters = rope
    t.mtp_num_hidden_layers = 0
    v.depth, v.hidden_size, v.num_heads, v.intermediate_size, v.out_hidden_size = 1, 32, 2, 64, 64
    # what the engine sets on a value model's config (transformer_impl.py:308-311)
    cfg.num_labels, cfg.classifier_dropout, cfg.hidden_dropout, cfg.summary_dropout_prob = 1, 0.0, "0", 0.0
    return cfg


def build(cfg, seed=0):
    from transformers import AutoModelForTokenClassification

    torch.manual_seed(seed)
    model = AutoModelForTokenClassification.from_config(cfg, attn_implementation="eager", torch_dtype=torch.float32)
    return model.eval()


def episode_fields(proc, n_images, seed):
    """One episode's TransferQueue fields as the agent loop's writer computes them (test_agent_loop.py tq_fields):
    the prompt with ``n_images`` frames (one a turn), an answer, input_ids, position_ids (4, L), multi_modal_inputs."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    images = [noise_image(seed + i) for i in range(n_images)]
    messages = []
    for i in range(n_images):
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Turn {i}: go."}]})
        if i < n_images - 1:
            messages.append({"role": "assistant", "content": "<action>move_forward|</action>"})
    prompt = proc.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
    enc = proc.processor(text=[prompt], images=images, return_tensors="pt")
    prompt_ids = enc["input_ids"][0]
    response_ids = torch.tensor(proc.tokenizer.encode("<action>turn_left|move_forward|</action>",
                                                      add_special_tokens=False) + [proc.tokenizer.eos_token_id])
    input_ids = torch.cat([prompt_ids, response_ids])
    worker = SimpleNamespace(processor=proc.processor, tokenizer=proc.tokenizer, config=OmegaConf.create({"data": {}}),
                             mm_processor_kwargs={})
    worker._get_mm_processor_kwargs = lambda audio_data=None: {}
    output = SimpleNamespace(multi_modal_data={"images": images}, mm_processor_kwargs=None)
    mm = AgentLoopWorker._compute_multi_modal_inputs(worker, output, input_ids)
    pos = AgentLoopWorker._compute_position_ids(worker, input_ids.unsqueeze(0), torch.ones_like(input_ids).unsqueeze(0),
                                                mm).squeeze(0)
    return {"prompts": prompt_ids, "responses": response_ids, "input_ids": input_ids, "position_ids": pos,
            "multi_modal_inputs": mm, "n_images": n_images}


@pytest.fixture(scope="module")
def episodes():
    proc = load_model(MODEL)
    return [episode_fields(proc, 1, 100), episode_fields(proc, 2, 200)]


def engine_inputs(eps):
    """FSDPEngineWithLMHead.prepare_model_inputs (transformer_impl.py:1241-1428) on a micro-batch of ``eps``, with
    use_remove_padding (the critic config's): one packed row, attention_mask None, position_ids (4, 1, N)."""
    from tensordict import NonTensorStack

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    def nested(name):
        return torch.nested.as_nested_tensor([e[name] for e in eps], layout=torch.jagged)

    td = tu.get_tensordict({"input_ids": nested("input_ids"), "prompts": nested("prompts"),
                            "responses": nested("responses"),
                            "position_ids": torch.nested.as_nested_tensor([e["position_ids"] for e in eps],
                                                                          layout=torch.jagged)},
                           non_tensor_dict={"use_remove_padding": True, "temperature": 0.8})
    td["multi_modal_inputs"] = NonTensorStack(*[e["multi_modal_inputs"] for e in eps])
    stub = SimpleNamespace(pad_to_length=False, use_ulysses_sp=False, ulysses_sequence_parallel_size=1,
                           pass_packed_cu_seqlens=False, _get_packed_pad_size=lambda n: 0)
    inputs, args = FSDPEngineWithLMHead.prepare_model_inputs(stub, td)
    return td, inputs, args, stub


def value_head_outputs(model, td, inputs, args, stub):
    """The module's forward as forward_step calls it (transformer_impl.py:1673-1680) and the value head's output
    (FSDPEngineWithValueHead.prepare_model_outputs, :1707-1751)."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead

    raw = model(**inputs, use_cache=False)
    stub = SimpleNamespace(**vars(stub), module=model)
    stub._gather_and_unpad_packed = lambda t, pad: FSDPEngineWithLMHead._gather_and_unpad_packed(stub, t, pad)
    return FSDPEngineWithValueHead.prepare_model_outputs(stub, raw, args, td, None)["values"]


def test_class_and_weights_from_the_causal_lm(cpu_gdn, tmp_path):
    from transformers import AutoModelForTokenClassification, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForTokenClassification

    cfg = tiny_config()
    torch.manual_seed(1)
    lm = Qwen3_5ForConditionalGeneration._from_config(cfg, attn_implementation="eager", torch_dtype=torch.float32)
    lm.save_pretrained(tmp_path / "lm")
    critic, info = AutoModelForTokenClassification.from_pretrained(
        tmp_path / "lm", config=cfg, attn_implementation="eager", torch_dtype=torch.float32, output_loading_info=True)
    assert type(critic) is Qwen3_5ForTokenClassification
    assert sorted(info["missing_keys"]) == ["score.bias", "score.weight"], info["missing_keys"]
    assert all(k.startswith("lm_head") for k in info["unexpected_keys"]), info["unexpected_keys"]
    assert critic.score.out_features == 1 and critic.score.in_features == cfg.text_config.hidden_size
    lm_sd, cr_sd = lm.state_dict(), critic.state_dict()
    shared = [k for k in cr_sd if not k.startswith("score.")]
    assert shared and all(torch.equal(cr_sd[k], lm_sd[k]) for k in shared)  # every tower / LM weight carried over
    assert any(".visual." in k for k in shared) and any(".language_model." in k for k in shared)
    print(f"[critic cpu] {type(critic).__name__}: {len(shared)} tensors from the CausalLM, new {info['missing_keys']}")


def test_forward_with_the_agent_loop_inputs(cpu_gdn, episodes, monkeypatch):
    """Both episodes through the engine's input path and verl's qwen3_5 patch, one at a time (one episode a
    micro-batch): finite values equal to the unpatched padded forward; vagen_value_loss on them; a backward."""
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding

    from vagen.q38.losses import vagen_value_loss

    M = cpu_gdn
    model = build(tiny_config())
    for e in episodes:
        assert e["position_ids"].shape == (4, len(e["input_ids"]))
        assert int((e["input_ids"] == model.config.image_token_id).sum()) == 64 * e["n_images"]  # 256 px frames
        assert set(e["multi_modal_inputs"]) >= {"pixel_values", "image_grid_thw"}
    # the unpatched model, padded single rows (attention_mask of ones), as a reference
    ref = []
    with torch.no_grad():
        for e in episodes:
            mm = {k: v for k, v in e["multi_modal_inputs"].items() if k in ("pixel_values", "image_grid_thw")}
            out = model(input_ids=e["input_ids"][None], attention_mask=torch.ones_like(e["input_ids"])[None],
                        position_ids=e["position_ids"][:, None, :], use_cache=False, **mm)
            ref.append(out.logits[0, :, 0])
    # verl's monkey patch, as the engine applies it at build time (transformer_impl.py:333-339); undone after the test
    from transformers.integrations import flash_attention

    for cls in (M.Qwen3_5Model, M.Qwen3_5DecoderLayer, M.Qwen3_5GatedDeltaNet, M.Qwen3_5ForConditionalGeneration):
        monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(M.Qwen3_5VisionModel, "fast_pos_embed_interpolate", M.Qwen3_5VisionModel.fast_pos_embed_interpolate)
    monkeypatch.setattr(flash_attention, "_flash_attention_forward", flash_attention._flash_attention_forward)
    try:
        apply_monkey_patch(model=model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False)
    except ModuleNotFoundError as e:  # the FLA autotune tweak needs Triton's kernels module
        pytest.skip(f"verl's qwen3_5 patch needs {e}")
    patched_forward = M.Qwen3_5Model.forward
    assert patched_forward.__name__ == "qwen3_5_base_forward"
    # the reason for one episode a micro-batch: no cu_seqlens parameter, so the engine passes none
    assert "cu_seqlens" not in inspect.signature(type(model).forward).parameters
    assert "cu_seqlens" in inspect.signature(M.Qwen3_5ForConditionalGeneration.forward).parameters  # the actor's
    model.train()  # the critic update's mode (dropout 0)
    for i, e in enumerate(episodes):
        td, inputs, args, stub = engine_inputs([e])
        assert inputs["position_ids"].shape == (4, 1, len(e["input_ids"])) and inputs["attention_mask"] is None
        assert "cu_seqlens" not in inputs
        values = value_head_outputs(model, td, inputs, args, stub)  # nested (1, prompt + response)
        flat = values.values()
        assert torch.isfinite(flat).all()
        assert torch.allclose(flat.detach(), ref[i], atol=1e-5, rtol=1e-4), (flat.detach() - ref[i]).abs().max()
        # vagen_value_loss on the value head's output: the response's anchor is its first token
        r = len(e["responses"])
        returns = torch.full((r,), -100.0)
        returns[0] = 1.25
        data = tu.get_tensordict(
            {"prompts": td["prompts"], "responses": td["responses"],
             "values": torch.nested.as_nested_tensor([torch.zeros(r)], layout=torch.jagged),
             "returns": torch.nested.as_nested_tensor([returns], layout=torch.jagged),
             "response_mask": torch.nested.as_nested_tensor([torch.ones(r, dtype=torch.int64)], layout=torch.jagged)},
            non_tensor_dict={"dp_size": 1, "batch_num_tokens": r, "global_batch_size": 1})
        cfg = SimpleNamespace(cliprange_value=1e6, loss_agg_mode="seq-mean-token-mean", loss_scale_factor=None)
        loss, metrics = vagen_value_loss(cfg, {"values": values}, data, global_batch_size=1, max_rows_per_micro=1)
        vpred = no_padding_2_padding(values, data)[0, 0]
        assert float(loss) == pytest.approx(0.5 * float((vpred - 1.25) ** 2), rel=1e-6)
        model.zero_grad()
        loss.backward()
        assert float(model.score.weight.grad.abs().sum()) > 0
        lm_grads = [p.grad for n, p in model.named_parameters() if ".language_model." in n and p.grad is not None]
        assert lm_grads and all(torch.isfinite(g).all() for g in lm_grads)
        assert metrics["critic/raw/anchors"].values == [1.0]
        print(f"[critic cpu] episode {i}: {len(e['input_ids'])} tokens, {e['n_images']} image(s), "
              f"multi_modal_inputs {sorted(e['multi_modal_inputs'])}, max |patched - unpatched| "
              f"{float((flat.detach() - ref[i]).abs().max()):.2e}, loss {float(loss):.4f}")


def test_packed_episodes_without_cu_seqlens_mix(cpu_gdn, episodes, monkeypatch):
    """Two episodes in one packed row, as the engine would pack a micro-batch of two for this class (no cu_seqlens):
    the second episode's values differ from its values alone, so the critic runs one episode a micro-batch
    (vagen/q38/trainer.py _critic_engine_keys, losses.vagen_value_loss max_rows_per_micro).  Measured as well with
    full attention only (no gated-delta-net layer)."""
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    M = cpu_gdn
    from transformers.integrations import flash_attention

    for cls in (M.Qwen3_5Model, M.Qwen3_5DecoderLayer, M.Qwen3_5GatedDeltaNet, M.Qwen3_5ForConditionalGeneration):
        monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(M.Qwen3_5VisionModel, "fast_pos_embed_interpolate", M.Qwen3_5VisionModel.fast_pos_embed_interpolate)
    monkeypatch.setattr(flash_attention, "_flash_attention_forward", flash_attention._flash_attention_forward)
    diffs = {}
    for name, types in (("hybrid", ("linear_attention", "full_attention")),
                        ("full_attention_only", ("full_attention", "full_attention"))):
        model = build(tiny_config(types), seed=3)
        try:
            apply_monkey_patch(model=model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False)
        except ModuleNotFoundError as e:
            pytest.skip(f"verl's qwen3_5 patch needs {e}")
        with torch.no_grad():
            alone = [value_head_outputs(model, *engine_inputs([e])).values() for e in episodes]
            td, inputs, args, stub = engine_inputs(episodes)
            assert "cu_seqlens" not in inputs and inputs["input_ids"].shape[1] == sum(len(e["input_ids"]) for e in episodes)
            both = value_head_outputs(model, td, inputs, args, stub)
        first, second = both.unbind()
        diffs[name] = (float((first - alone[0]).abs().max()), float((second - alone[1]).abs().max()))
    print(f"[critic cpu] packed without cu_seqlens, max |packed - alone| (first, second episode): {diffs}")
    assert diffs["hybrid"][0] < 1e-5  # the first episode sees nothing of the second
    assert diffs["hybrid"][1] > 1e-4  # the second one carries the first one's gated-delta state




def test_load_valuehead_model_asks_for_flash_attention_2():
    """verl's critic loader passes attn_implementation="flash_attention_2" to from_pretrained (utils/model.py:645-651),
    which wins over the config's override_config: the 27B critic would run FA2 where the actor runs eager (layout
    file, the user 11-17).  Pinned here: phase 3 open question (accept, or a verl patch)."""
    from verl.utils import model as vm

    src = inspect.getsource(vm.load_valuehead_model)
    assert src.count('attn_implementation="flash_attention_2"') == 2, src
