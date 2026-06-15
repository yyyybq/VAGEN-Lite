#!/usr/bin/env python3
"""
probe_spatial_qa.py — 离线 backbone capability probe

测试 RL checkpoint 在空间推理 QA 上的能力是否退化（对比基础模型）。
从 HuggingFace 格式的 checkpoint (actor/huggingface/) 加载权重，无需训练进程。

用法:
    # 对比 base model vs 多个 checkpoint
    python3 scripts/probe_spatial_qa.py \\
        --ckpts "base:Qwen/Qwen2.5-VL-3B-Instruct,v25_step240:exps/vagen_active_spatial/v25_groupadv_100scenes_klhi/checkpoints/global_step_240/actor/huggingface,v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \\
        --n_samples 200 --gpu 4

    # 仅测试单个 checkpoint（与 base 对比）
    python3 scripts/probe_spatial_qa.py \\
        --ckpts "v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \\
        --n_samples 300 --gpu 4 --include_base

输出:
    每个 checkpoint 的 MindCube tinybench 准确率（整体 + 按 type 细分）
    + 5 条文本纯文字 sanity-check（检测通用能力退化）
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict

import torch

# ─────────────────────────── 配置 ───────────────────────────

MINDCUBE_JSONL  = "/scratch/by2593/MindCube/data/raw/MindCube_tinybench.jsonl"
MINDCUBE_IMAGE_BASE = "/scratch/by2593/MindCube/data"
BASE_MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"

# 文字 sanity-check（不依赖图片，检测通用退化）
TEXT_SANITY = [
    {
        "id": "direction_text",
        "q":  "I face North. I turn 90° right. What direction am I facing? A. North  B. East  C. South  D. West",
        "gt": "B",
    },
    {
        "id": "logic_cat",
        "q":  "All cats are animals. Felix is a cat. Is Felix an animal? Answer: A. Yes  B. No",
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
        "gt": "A",  # facing North: right=East ✓
    },
]


# ─────────────────────────── 模型加载 ───────────────────────────

def load_model_and_processor(ckpt_path: str, device: str):
    """
    从 HuggingFace 格式路径加载模型和 processor。
    ckpt_path: 路径或 HuggingFace model ID (如 'Qwen/Qwen2.5-VL-3B-Instruct')
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig

    print(f"  Loading model from: {ckpt_path}")

    # Resolve relative paths
    if not ckpt_path.startswith("/") and not ckpt_path.startswith("Qwen/") and not ckpt_path.startswith("~"):
        ckpt_path = os.path.join(os.getcwd(), ckpt_path)

    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        ckpt_path if os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, "tokenizer.json"))
        else BASE_MODEL_ID,
        trust_remote_code=True,
    )

    print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, processor


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────── 推理 ───────────────────────────

def run_text_inference(model, processor, question: str, device: str) -> str:
    """纯文字推理（无图片）。"""
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


def run_vqa_inference(model, processor, question: str, image_paths: list, device: str) -> str:
    """带图片的 VQA 推理。"""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("qwen_vl_utils not found. Install via: pip install qwen-vl-utils")

    content = []
    for p in image_paths:
        full_path = p if os.path.isabs(p) else os.path.join(MINDCUBE_IMAGE_BASE, p)
        if os.path.exists(full_path):
            content.append({"type": "image", "image": f"file://{full_path}"})
        else:
            print(f"    [WARN] Image not found: {full_path}", file=sys.stderr)

    if not content:
        # Fall back to text-only if no images found
        return run_text_inference(model, processor, question, device)

    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_input, vid_input = process_vision_info(messages)
    inputs = processor(
        text=[text], images=img_input, videos=vid_input,
        padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


def extract_answer_letter(response: str) -> str | None:
    """从回复中提取 A/B/C/D。优先取第一个选项字母。"""
    # First: look for explicit "Answer: X" or "The answer is X"
    m = re.search(r"(?:answer|answer is|The answer is)[:\s]+([A-D])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Second: standalone letter at start or after period/newline
    m = re.search(r"(?:^|[\n.]\s*)([A-D])[.\)]\s", response)
    if m:
        return m.group(1).upper()
    # Third: any standalone capital letter A-D
    letters = re.findall(r"\b([A-D])\b", response)
    if letters:
        return letters[0].upper()
    return None


# ─────────────────────────── 评估 ───────────────────────────

def evaluate_checkpoint(
    ckpt_name: str,
    ckpt_path: str,
    items: list,
    device: str,
    verbose: bool = False,
) -> dict:
    """评估单个 checkpoint 在 MindCube 上的表现。"""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {ckpt_name}")
    print(f"  Path: {ckpt_path}")
    print(f"{'='*70}")

    model, processor = load_model_and_processor(ckpt_path, device)

    # ── Sanity check (text-only) ──
    print("\n  [Text Sanity Check]")
    sanity_correct = 0
    for tc in TEXT_SANITY:
        resp = run_text_inference(model, processor, tc["q"], device)
        pred = extract_answer_letter(resp)
        correct = pred == tc["gt"]
        sanity_correct += int(correct)
        mark = "✓" if correct else "✗"
        if verbose:
            print(f"    {mark} [{tc['id']}] pred={pred} gt={tc['gt']}  resp={resp[:80]!r}")
        else:
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
            resp = run_vqa_inference(model, processor, q, imgs, device)
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
            overall = results_by_type["overall"]
            pct = overall["correct"] / overall["total"] * 100 if overall["total"] else 0
            print(f"    Progress {i+1}/{len(items)} — running acc: {pct:.1f}%")

    if errors:
        print(f"\n  [WARN] {len(errors)} inference errors")

    unload_model(model)

    # ── Compile results ──
    output = {
        "name":        ckpt_name,
        "sanity":      f"{sanity_correct}/{len(TEXT_SANITY)}",
        "by_type":     {},
    }
    overall = results_by_type.pop("overall")
    overall_acc = overall["correct"] / overall["total"] if overall["total"] else 0
    output["overall_acc"] = overall_acc
    output["overall_n"]   = overall["total"]

    for t, v in sorted(results_by_type.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        output["by_type"][t] = {"acc": acc, "n": v["total"], "correct": v["correct"]}

    return output


# ─────────────────────────── 打印结果表 ───────────────────────────

def print_results_table(all_results: list):
    # Collect all types seen
    all_types = sorted(set(t for r in all_results for t in r["by_type"]))

    print(f"\n{'═'*90}")
    print("  MindCube Spatial QA — Backbone Probe Results")
    print(f"{'═'*90}")

    # Header
    name_w = max(20, max(len(r["name"]) for r in all_results) + 2)
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
                acc = r["by_type"][t]["acc"] * 100
                print(f"  {acc:>8.1f}%", end="")
            else:
                print(f"  {'—':>9}", end="")
        print()

    print(f"\n  Types: {', '.join(all_types)}")
    print(f"  (three_view=四视角, N_frame=N步移动后定位, general=综合空间推理)")

    # Degradation analysis vs first result (if >1 checkpoint)
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
        description="Offline backbone capability probe on MindCube spatial QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Test v26 step160 vs base model
          python3 scripts/probe_spatial_qa.py \\
            --ckpts "base:Qwen/Qwen2.5-VL-3B-Instruct,v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \\
            --n_samples 200 --gpu 4

          # Test multiple checkpoints (no base needed if you already have baseline)
          python3 scripts/probe_spatial_qa.py \\
            --ckpts "v25_step240:exps/vagen_active_spatial/v25_groupadv_100scenes_klhi/checkpoints/global_step_240/actor/huggingface,v26_step160:exps/vagen_active_spatial/v26_klhi_lr5e7/checkpoints/global_step_160/actor/huggingface" \\
            --n_samples 200 --gpu 4 --include_base
        """),
    )
    parser.add_argument(
        "--ckpts", required=True,
        help='逗号分隔的 "name:path" 对，path 可以是 HuggingFace ID 或本地目录。'
             '例: "base:Qwen/Qwen2.5-VL-3B-Instruct,v26:exps/.../global_step_160/actor/huggingface"',
    )
    parser.add_argument("--include_base", action="store_true",
                        help="自动在列表前加入 base model 作为基准对比")
    parser.add_argument("--n_samples",  type=int, default=200,
                        help="从 MindCube tinybench 中采样的样本数 (default=200, max=1050)")
    parser.add_argument("--gpu",        type=int, default=4,
                        help="使用的 GPU id (default=4)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="采样随机种子 (default=42)")
    parser.add_argument("--verbose",    action="store_true",
                        help="打印每个错误预测的详情")
    parser.add_argument("--show_examples", type=int, default=0, metavar="N",
                        help="从 --ckpts 中最后一个 checkpoint 打印 N 条完整回答示例 (default=0=不打印)")
    parser.add_argument("--mindcube",   default=MINDCUBE_JSONL,
                        help=f"MindCube tinybench JSONL 路径 (default: {MINDCUBE_JSONL})")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={args.gpu})")

    # ── Load benchmark data ──
    print(f"\nLoading MindCube tinybench from {args.mindcube}")
    with open(args.mindcube) as f:
        all_items = [json.loads(l) for l in f if l.strip()]

    # Stratified sample: maintain type distribution
    import random
    random.seed(args.seed)
    n = min(args.n_samples, len(all_items))
    if n < len(all_items):
        by_type = defaultdict(list)
        for item in all_items:
            by_type[str(item["type"])].append(item)
        selected = []
        # Sample proportionally
        for t, t_items in by_type.items():
            k = max(1, round(n * len(t_items) / len(all_items)))
            selected.extend(random.sample(t_items, min(k, len(t_items))))
        # Trim or fill to exactly n
        random.shuffle(selected)
        selected = selected[:n]
    else:
        selected = all_items

    print(f"Selected {len(selected)} items (stratified by type)")
    type_dist = Counter(str(x["type"]) for x in selected)
    print(f"Type distribution: {dict(type_dist)}")

    # ── Parse checkpoint list ──
    ckpt_list = []
    if args.include_base:
        ckpt_list.append(("base", BASE_MODEL_ID))
    for spec in args.ckpts.split(","):
        spec = spec.strip()
        if ":" in spec:
            # Split on first colon (to handle absolute paths like /path/to/...)
            name, path = spec.split(":", 1)
            ckpt_list.append((name.strip(), path.strip()))
        else:
            print(f"[WARN] Skipping malformed entry (no colon): {spec!r}", file=sys.stderr)

    if not ckpt_list:
        print("[ERROR] No valid checkpoints specified.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCheckpoints to evaluate: {[n for n,_ in ckpt_list]}")

    # ── Evaluate each checkpoint ──
    all_results = []
    for name, path in ckpt_list:
        result = evaluate_checkpoint(name, path, selected, device, verbose=args.verbose)
        all_results.append(result)

    # ── Print comparison table ──
    print_results_table(all_results)

    # ── Show example responses (last checkpoint) ──
    if args.show_examples > 0:
        last_name, last_path = ckpt_list[-1]
        print(f"\n{'═'*80}")
        print(f"  示例回答（{last_name}，前 {args.show_examples} 条）")
        print(f"{'═'*80}")
        model, processor = load_model_and_processor(last_path, device)
        sample = selected[:args.show_examples]
        for idx, item in enumerate(sample):
            gt  = item["gt_answer"].strip().strip("'\"").upper()
            imgs = item.get("images", [])
            q   = item["question"]
            qtype = str(item["type"])
            resp = run_vqa_inference(model, processor, q, imgs, device)
            pred = extract_answer_letter(resp)
            mark = "✓" if pred == gt else "✗"
            print(f"\n[{idx+1}] type={qtype}  gt={gt}  pred={pred}  {mark}")
            print(f"  Q: {q[:200]}{'...' if len(q)>200 else ''}")
            print(f"  Images: {len(imgs)} file(s)")
            print(f"  Response: {resp!r}")
        unload_model(model)


if __name__ == "__main__":
    main()
