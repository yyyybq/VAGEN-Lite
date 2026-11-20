"""q38: the SFT of Qwen3.8-27B (model_type qwen3_5) with LLaMA-Factory main (SFT_OPTIONS.md option A, TEMPLATE_CHECK.md).

- lf_data: an 8B LLaMA-Factory folder (sharegpt jsonl + dataset_info.json + train.yaml) -> the 27B folder: every
  message stripped of leading / trailing whitespace (TEMPLATE_CHECK conclusion 2: Qwen3.8's chat template trims every
  message, LLaMA-Factory's qwen3_8 template strips only the system turn), the train.yaml with the qwen3_8 template and
  enable_thinking false, everything else of the 8B recipe kept;
- check_lf_qwen35: LLaMA-Factory's encoding of the converted data against processor.apply_chat_template(
  enable_thinking=False), token by token, labels, image tokens, pixels, M-RoPE positions, lengths, FSDP wrap units;
- lf_launcher: llamafactory.train.tuner.run_exp with this line's callback (per-step seconds, GPU and host memory,
  the per-epoch bf16 copy);
- to_bf16: an fp32 Hugging Face checkpoint (LLaMA-Factory's save, every trainable tensor fp32) -> a bf16 copy, checked
  tensor by tensor.
"""
