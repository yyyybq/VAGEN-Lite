#!/usr/bin/env python3
"""
probe_cambrian_qa.py — Cambrian-S backbone capability probe

测试 Cambrian-S RL checkpoint 在空间推理 QA 上的能力是否退化（对比基础模型）。
从 HuggingFace 格式 checkpoint 加载权重，使用 SigLIP 进行图像预处理。

关键实现说明（图像 token 展开）:
    CambrianProcessorWrapper.__call__() 将 <image> tokenize 为单个 special token (ID 151665)。
    CambrianForCausalLMAdapter.generate() 检查 input_ids 中是否有 IMAGE_TOKEN_INDEX=-200，
    只有存在时才触发 _embed_multimodal_batch()（视觉特征散射）。
    因此推理前必须将每个 <image> token 手动展开为 TOKENS_PER_IMAGE(756) 个 -200 token，
    否则 pixel_values 被静默忽略，退化为纯文字推理。

用法:
    # 对比 base model vs c8 各 step checkpoint
    python3 scripts/probe_cambrian_qa.py \
        --ckpts "base:/scratch/by2593/hf_cache/cambrian-s-7b,c8_s50:exps/vagen_active_spatial/c8_fwdfirst_rewscale/checkpoints/global_step_50/actor/huggingface,c8_s100:exps/vagen_active_spatial/c8_fwdfirst_rewscale/checkpoints/global_step_100/actor/huggingface,c8_s150:exps/vagen_active_spatial/c8_fwdfirst_rewscale/checkpoints/global_step_150/actor/huggingface,c8_s200:exps/vagen_active_spatial/c8_fwdfirst_rewscale/checkpoints/global_step_200/actor/huggingface" \
        --n_samples 200 --gpu 4

    # 仅测试单个 checkpoint（自动添加 base 对比）
    python3 scripts/probe_cambrian_qa.py \
        --ckpts "c8_s150:exps/vagen_active_spatial/c8_fwdfirst_rewscale/checkpoints/global_step_150/actor/huggingface" \
        --n_samples 200 --gpu 4 --include_base

    # 仅做 sanity check（跳过图片，快速验证环境）
    python3 scripts/probe_cambrian_qa.py \
        --ckpts "base:/scratch/by2593/hf_cache/cambrian-s-7b" \
        --n_samples 50 --gpu 4 --no_images --verbose
"""

import argparse
import gc
import json
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict

import torch
from PIL import Image

# ─────────────────────────── 配置 ───────────────────────────

MINDCUBE_JSONL      = "/scratch/by2593/MindCube/data/raw/MindCube_tinybench.jsonl"
MINDCUBE_IMAGE_BASE = "/scratch/by2593/MindCube/data"
BASE_MODEL_PATH     = "/scratch/by2593/hf_cache/cambrian-s-7b"

# SigLIP 图像尺寸（与 cambrian_processor.py 一致）
SIGLIP_MODEL       = "google/siglip2-so400m-patch14-384"
SIGLIP_SIZE        = 384
IMAGE_TOKEN        = "<image>"
IMAGE_TOKEN_INDEX  = -200   # FSDP adapter 中 <image> 展开后的 sentinel value
TOKENS_PER_IMAGE   = 756    # 27×28（SI tokens + newline tokens），与 cambrian_register.py 一致

# 文字 sanity-check（不依赖图片）
TEXT_SANITY = [
    {
        "id": "direction_text",
        "q":  "I face North. I turn 90° right. What direction am I facing? A. North  B. East  C. South  D. West",
        "gt": "B",
    },
    {
        "id": "logic_cat",
        "q":  "All cats are animals. Felix is a cat. Is Felix an animal? A. Yes  B. No",
        "gt": "A",
    },
    {
        "id": "simple_math",
        "q":  "What is 7 × 8? A. 54  B. 56  C. 58  D. 64",
        "gt": "B",
    },
    {
        "id": "color_sky",
        "q":  "What color is the sky on a clear day? A. Green  B. Red  C. Blue  D. Yellow",
        "gt": "C",
    },
    {
        "id": "spatial_left_right",
        "q":  "I am facing a wall. My right hand is pointing East. Which direction am I facing? A. North  B. South  C. East  D. West",
        "gt": "A",
    },
]


# ─────────────────────────── 模型加载 ───────────────────────────

def load_cambrian(ckpt_path: str, device: str):
    """从 HuggingFace 格式目录加载 Cambrian-S 模型和 tokenizer。"""
    import sys
    # 注册 Cambrian 模型类（使用 vagen 提供的 cambrian_register）
    vagen_root = os.path.join(os.path.dirname(__file__), "..")
    if vagen_root not in sys.path:
        sys.path.insert(0, vagen_root)
    try:
        import vagen.models.cambrian_register  # noqa: F401 — 触发 AutoModel 注册
    except ImportError:
        pass  # 若 vagen 不在 path 中，改用 trust_remote_code

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not ckpt_path.startswith("/"):
        ckpt_path = os.path.join(os.getcwd(), ckpt_path)

    print(f"  Loading Cambrian model from: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # tokenizer 从 RL checkpoint 直接加载（含 <image> 特殊 token）
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


def load_siglip_processor():
    """加载 SigLIP 图像预处理器（仅需一次）。"""
    from transformers import AutoImageProcessor
    print(f"  Loading SigLIP processor: {SIGLIP_MODEL}")
    return AutoImageProcessor.from_pretrained(SIGLIP_MODEL)


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────── 推理 ───────────────────────────

def _build_text_prompt(tokenizer, question: str) -> dict:
    """构建纯文字提示并 tokenize，返回包含 input_ids/attention_mask 的 dict。"""
    messages = [{"role": "user", "content": question}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer(text, return_tensors="pt")


def _expand_image_tokens(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """
    将 compact input_ids 中的每个 image_token_id（单个 <image> special token）
    展开为 TOKENS_PER_IMAGE(756) 个 IMAGE_TOKEN_INDEX(-200)。

    必须在调用 CambrianForCausalLMAdapter.generate() 之前执行，
    否则 generate() 中的 (input_ids == IMAGE_TOKEN_INDEX).any() 判断为 False，
    pixel_values 被静默忽略，退化为纯文字推理。

    输入:  (1, seq_len)
    输出:  (1, expanded_seq_len)，其中每个 image_token_id 被替换为 756 个 -200
    """
    flat = input_ids[0].tolist()
    expanded = []
    for tok in flat:
        if tok == image_token_id:
            expanded.extend([IMAGE_TOKEN_INDEX] * TOKENS_PER_IMAGE)
        else:
            expanded.append(tok)
    return torch.tensor([expanded], dtype=torch.long)


def run_text_inference(model, tokenizer, question: str, device: str) -> str:
    """纯文字推理（无图片）。"""
    inputs = _build_text_prompt(tokenizer, question).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(trimmed, skip_special_tokens=True)


def run_vqa_inference(
    model, tokenizer, siglip_proc,
    question: str, image_paths: list, device: str
) -> str:
    """
    带图片的 VQA 推理。

    流程:
      1. 加载 PIL 图像，用 SigLIP 生成 pixel_values
      2. 构建含 <image> 占位符的 prompt，tokenize → compact input_ids
      3. 展开 image tokens: 每个 <image>(单 token) → 756 × IMAGE_TOKEN_INDEX(-200)
      4. 调用 model.generate(expanded_input_ids, pixel_values=pixel_values)

    步骤 3 是关键：CambrianForCausalLMAdapter.generate() 检查
    (input_ids == -200).any() 来决定是否调用 _embed_multimodal_batch()。
    跳过展开会导致 pixel_values 被静默忽略。
    """
    if not image_paths or siglip_proc is None:
        return run_text_inference(model, tokenizer, question, device)

    # ── 加载 PIL 图像 ──
    pil_imgs = []
    for p in image_paths:
        full = p if os.path.isabs(p) else os.path.join(MINDCUBE_IMAGE_BASE, p)
        if os.path.exists(full):
            try:
                pil_imgs.append(Image.open(full).convert("RGB"))
            except Exception as e:
                print(f"    [WARN] Cannot open {full}: {e}", file=sys.stderr)

    if not pil_imgs:
        return run_text_inference(model, tokenizer, question, device)

    # ── SigLIP pixel_values: (N_images, 3, 384, 384) ──
    pv_out = siglip_proc(images=pil_imgs, return_tensors="pt")
    pixel_values = pv_out["pixel_values"].to(device=device, dtype=torch.bfloat16)

    # ── 构建含 <image> 占位符的提示 ──
    image_tokens = "\n".join([IMAGE_TOKEN] * len(pil_imgs))
    prompt_text = f"{image_tokens}\n{question}"
    compact_inputs = _build_text_prompt(tokenizer, prompt_text)
    compact_ids = compact_inputs["input_ids"]  # (1, seq_len)，<image> 为单 token

    # ── 展开 image tokens → 756 × IMAGE_TOKEN_INDEX(-200) ──
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    expanded_ids = _expand_image_tokens(compact_ids, image_token_id).to(device)
    expanded_mask = torch.ones(1, expanded_ids.shape[1], dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            out = model.generate(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                pixel_values=pixel_values,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            # 跳过输入部分（按展开后长度）
            trimmed = out[0][expanded_ids.shape[1]:]
            return tokenizer.decode(trimmed, skip_special_tokens=True)
        except Exception as e:
            print(f"    [WARN] VQA inference failed: {e}, falling back to text", file=sys.stderr)
            return run_text_inference(model, tokenizer, question, device)


def extract_answer_letter(response: str) -> str | None:
    """从回复中提取 A/B/C/D。"""
    m = re.search(r"(?:answer|answer is|The answer is)[:\s]+([A-D])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:^|[\n.]\s*)([A-D])[.\)]\s", response)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([A-D])\b", response)
    if letters:
        return letters[0].upper()
    return None


# ─────────────────────────── 评估 ───────────────────────────

def evaluate_checkpoint(
    ckpt_name: str,
    ckpt_path: str,
    items: list,
    siglip_proc,
    device: str,
    verbose: bool = False,
) -> dict:
    print(f"\n{'='*70}")
    print(f"  Evaluating: {ckpt_name}")
    print(f"  Path: {ckpt_path}")
    print(f"{'='*70}")

    model, tokenizer = load_cambrian(ckpt_path, device)

    # ── Text sanity check ──
    print("\n  [Text Sanity Check]")
    sanity_correct = 0
    for tc in TEXT_SANITY:
        resp = run_text_inference(model, tokenizer, tc["q"], device)
        pred = extract_answer_letter(resp)
        correct = pred == tc["gt"]
        sanity_correct += int(correct)
        mark = "✓" if correct else "✗"
        print(f"    {mark} {tc['id']}: pred={pred}, gt={tc['gt']}")
    print(f"  Sanity: {sanity_correct}/{len(TEXT_SANITY)}")

    # ── MindCube VQA ──
    print(f"\n  [MindCube VQA — {len(items)} items]")
    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    errors = []

    for i, item in enumerate(items):
        qtype = str(item["type"])
        gt    = item["gt_answer"].strip().strip("'\"").upper()
        imgs  = item.get("images", [])
        q     = item["question"]

        try:
            resp = run_vqa_inference(model, tokenizer, siglip_proc, q, imgs, device)
            pred = extract_answer_letter(resp)
            correct = (pred == gt)
        except Exception as e:
            errors.append((i, str(e)))
            pred, correct = None, False

        results_by_type[qtype]["total"] += 1
        results_by_type[qtype]["correct"] += int(correct)
        results_by_type["overall"]["total"] += 1
        results_by_type["overall"]["correct"] += int(correct)

        if verbose and not correct:
            print(f"    ✗ [{i}] type={qtype} pred={pred} gt={gt}")

        if (i + 1) % 50 == 0:
            ov = results_by_type["overall"]
            pct = ov["correct"] / ov["total"] * 100 if ov["total"] else 0
            print(f"    Progress {i+1}/{len(items)} — running acc: {pct:.1f}%")

    if errors:
        print(f"\n  [WARN] {len(errors)} inference errors (first 3): {errors[:3]}")

    unload(model)

    overall = results_by_type.pop("overall")
    overall_acc = overall["correct"] / overall["total"] if overall["total"] else 0
    output = {
        "name":        ckpt_name,
        "sanity":      f"{sanity_correct}/{len(TEXT_SANITY)}",
        "overall_acc": overall_acc,
        "overall_n":   overall["total"],
        "by_type":     {},
    }
    for t, v in sorted(results_by_type.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        output["by_type"][t] = {"acc": acc, "n": v["total"], "correct": v["correct"]}

    return output


# ─────────────────────────── 打印 ───────────────────────────

def print_results_table(all_results: list):
    all_types = sorted(set(t for r in all_results for t in r["by_type"]))
    print(f"\n{'═'*90}")
    print("  MindCube Spatial QA — Cambrian-S Backbone Probe")
    print(f"{'═'*90}")
    name_w = max(22, max(len(r["name"]) for r in all_results) + 2)
    print(f"  {'Checkpoint':<{name_w}}  {'Sanity':>8}  {'Overall':>8}  {'N':>5}", end="")
    for t in all_types:
        print(f"  {t[:9]:>9}", end="")
    print()
    print(f"  {'─'*name_w}  {'─'*8}  {'─'*8}  {'─'*5}", end="")
    for _ in all_types:
        print("  " + "─"*9, end="")
    print()
    for r in all_results:
        print(f"  {r['name']:<{name_w}}  {r['sanity']:>8}  {r['overall_acc']*100:>7.1f}%  {r['overall_n']:>5}", end="")
        for t in all_types:
            if t in r["by_type"]:
                print(f"  {r['by_type'][t]['acc']*100:>8.1f}%", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()

    if len(all_results) > 1:
        base = all_results[0]
        print(f"\n  [退化分析 vs {base['name']}]")
        for r in all_results[1:]:
            delta = (r["overall_acc"] - base["overall_acc"]) * 100
            sign  = "+" if delta >= 0 else ""
            flag  = "⚠ 退化" if delta < -3 else ("↗ 提升" if delta > 3 else "≈ 持平")
            print(f"    {r['name']:<{name_w-2}}  Δacc={sign}{delta:.1f}%  {flag}")
    print()


# ─────────────────────────── 主程序 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cambrian-S backbone probe on MindCube spatial QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 scripts/probe_cambrian_qa.py \\
            --ckpts "base:/scratch/by2593/hf_cache/cambrian-s-7b,c4_s80:exps/vagen_active_spatial/c4_fwdfirst/checkpoints/global_step_80/actor/huggingface" \\
            --n_samples 200 --gpu 0

          python3 scripts/probe_cambrian_qa.py \\
            --ckpts "c4_s80:exps/vagen_active_spatial/c4_fwdfirst/checkpoints/global_step_80/actor/huggingface" \\
            --n_samples 200 --gpu 0 --include_base
        """),
    )
    parser.add_argument("--ckpts", required=True,
        help='逗号分隔的 "name:path" 对，path 为本地绝对或相对路径')
    parser.add_argument("--include_base", action="store_true",
        help=f"自动在列表前加入 base model ({BASE_MODEL_PATH})")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--no_images", action="store_true",
        help="跳过图片加载，仅做纯文字推理（更快，适合 sanity-only 检查）")
    parser.add_argument("--mindcube",  default=MINDCUBE_JSONL)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={args.gpu})")

    # ── Load benchmark ──
    print(f"\nLoading MindCube tinybench from {args.mindcube}")
    with open(args.mindcube) as f:
        all_items = [json.loads(l) for l in f if l.strip()]

    import random
    random.seed(args.seed)
    n = min(args.n_samples, len(all_items))
    if n < len(all_items):
        by_type = defaultdict(list)
        for item in all_items:
            by_type[str(item["type"])].append(item)
        selected = []
        for t, t_items in by_type.items():
            k = max(1, round(n * len(t_items) / len(all_items)))
            selected.extend(random.sample(t_items, min(k, len(t_items))))
        random.shuffle(selected)
        selected = selected[:n]
    else:
        selected = all_items

    print(f"Selected {len(selected)} items. Type dist: {dict(Counter(str(x['type']) for x in selected))}")

    # ── Load SigLIP processor (shared across checkpoints) ──
    siglip_proc = None
    if not args.no_images:
        try:
            siglip_proc = load_siglip_processor()
        except Exception as e:
            print(f"[WARN] Could not load SigLIP: {e}. Falling back to text-only.", file=sys.stderr)

    # ── Parse checkpoints ──
    ckpt_list = []
    if args.include_base:
        ckpt_list.append(("base", BASE_MODEL_PATH))
    for spec in args.ckpts.split(","):
        spec = spec.strip()
        if ":" in spec:
            name, path = spec.split(":", 1)
            ckpt_list.append((name.strip(), path.strip()))
        else:
            print(f"[WARN] Skipping malformed entry: {spec!r}", file=sys.stderr)

    if not ckpt_list:
        print("[ERROR] No valid checkpoints.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCheckpoints: {[n for n,_ in ckpt_list]}")

    # ── Evaluate ──
    all_results = []
    for name, path in ckpt_list:
        result = evaluate_checkpoint(name, path, selected, siglip_proc, device, verbose=args.verbose)
        all_results.append(result)

    print_results_table(all_results)


if __name__ == "__main__":
    main()
