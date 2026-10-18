"""The QA benchmarks of eval_external and how each backend runs them (one place, so every model gets the same
prompt, frames and decoding).

Prompts are the benchmarks' own (the backend's implementation of the official template), never our training
format.  Decoding is greedy for every model; each model's view also carries a greedy generation_config.json
(see adapters/models.py), so nothing a backend leaves unset differs between models.
"""

from __future__ import annotations

from pathlib import Path

# Same arguments as VLMEvalKit's own "Qwen2.5-VL-7B-Instruct" / "-3B-Instruct" entries (EASI v0.2.2 config.py),
# used for every Qwen2.5-VL-based model.  The class is Qwen2VLChat loaded with SDPA attention instead of
# flash-attn (qa_transfer/vlmeval_launch.py says why).
VLMEVAL_QWEN25VL = {
    "class": "Qwen2VLChatSDPA",
    "min_pixels": 1280 * 28 * 28,
    "max_pixels": 16384 * 28 * 28,
    "use_custom_prompt": False,
}
# Qwen2VLChat's own generation defaults (vlmeval/vlm/qwen2_vl/model.py), recorded in every manifest.
VLMEVAL_DECODING = {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.01, "top_p": 0.001, "top_k": 1,
                    "repetition_penalty": 1.0}

# lmms-eval: model "qwen2_5_vl" with these model_args (pretrained= is added per model); the decoding comes from
# each task's yaml (greedy) plus the view's generation_config.json.
LMMS_QWEN25VL_ARGS = {"max_num_frames": 32, "attn_implementation": "sdpa"}
# Our task variants (lmms_tasks/*.yaml): each includes the original task and changes only what its header says.
LMMS_INCLUDE = Path(__file__).resolve().parent / "lmms_tasks"

BENCHMARKS = {
    # name used in eval_external tables -> backend and backend-specific settings
    "VSI-Bench": {
        "backend": "vlmevalkit", "data": "VSI-Bench_32frame", "data_cfg": {}, "frames": 32, "items": 5130,
        "metric": "mean over 8 sub-tasks; MCQ accuracy, numeric MRA",
        "source": {"videos": "nyu-visionx/VSI-Bench (HF snapshot, revision recorded at run time)",
                   "questions": "lmms-lab-si/EASI-Leaderboard-Data/VSI-Bench.tsv md5 34544fd83241391d83eff087a1be7d83"},
        "prompt": {"impl": "vlmeval.dataset.vsibench.VsiBench.build_prompt",
                   "pre": "These are frames of a video.",
                   "mcq_post": "Answer with the option's letter from the given choices directly.",
                   "num_post": "Answer briefly and directly in one float number."},
    },
    "VSI-Bench@lmms": {
        "backend": "lmms_eval", "task": "vsibench_ee", "frames": 32, "items": 5130,
        "metric": "mean over 8 sub-tasks; MCQ accuracy, numeric MRA",
        "source": {"dataset": "nyu-visionx/VSI-Bench (HF snapshot, revision recorded at run time)",
                   "videos": "$HF_HOME/vsibench, unpacked from the same snapshot",
                   "task": "lmms_tasks/vsibench_ee.yaml = lmms-eval vsibench with dataset token off"},
        "prompt": {"impl": "lmms_eval/tasks/vsibench/utils.py vsibench_doc_to_text"},
    },
    "ViewSpatial": {
        "backend": "vlmevalkit", "data": "ViewSpatialBench",
        "data_cfg": {"class": "ViewSpatialBench", "dataset": "ViewSpatialBench"}, "frames": None, "items": 5712,
        "metric": "accuracy",
        "source": {"questions": "lmms-lab-si/EASI-Leaderboard-Data@52bb1dd ViewSpatialBench.tsv (md5 checked by VLMEvalKit)"},
        "prompt": {"impl": "vlmeval.dataset.ViewSpatialBench.build_prompt"},
    },
    "MindCube-Tiny": {
        "backend": "vlmevalkit", "data": "MindCubeBench_tiny_raw_qa",
        # EASI's table has 1,040 items; MindCube's own MindCube_tinybench.jsonl has 1,050.
        "data_cfg": {"class": "MindCubeBench", "dataset": "MindCubeBench_tiny_raw_qa"}, "frames": None, "items": 1040,
        "metric": "accuracy",
        "source": {"questions": "lmms-lab-si/EASI-Leaderboard-Data@52bb1dd MindCubeBench_tiny_raw_qa.tsv"},
        "prompt": {"impl": "vlmeval.dataset.MindCubeBench.build_prompt (raw QA)"},
    },
    "MindCube": {
        "backend": "vlmevalkit", "data": "MindCubeBench_raw_qa",
        "data_cfg": {"class": "MindCubeBench", "dataset": "MindCubeBench_raw_qa"}, "frames": None, "items": 21154,
        "metric": "accuracy",
        "source": {"questions": "lmms-lab-si/EASI-Leaderboard-Data@52bb1dd MindCubeBench_raw_qa.tsv"},
        "prompt": {"impl": "vlmeval.dataset.MindCubeBench.build_prompt (raw QA)"},
    },
    "MMSI": {
        "backend": "vlmevalkit", "data": "MMSIBench_wo_circular",
        "data_cfg": {"class": "MMSIBench", "dataset": "MMSIBench_wo_circular"}, "frames": None, "items": 1000,
        "metric": "accuracy",
        "source": {"questions": "lmms-lab-si/EASI-Leaderboard-Data@52bb1dd MMSIBench_wo_circular.tsv"},
        "prompt": {"impl": "vlmeval.dataset.MMSIBench.build_prompt"},
    },
    "SPBench-MV": {
        "backend": "vlmevalkit", "data": "SPBench-MV",
        "data_cfg": {"class": "SPBench", "dataset": "SPBench-MV"}, "frames": None, "items": 319,
        "metric": "accuracy (MCQ) and MRA (numeric), as SPBench defines",
        "source": {"questions": "lmms-lab-si/EASI-Leaderboard-Data@52bb1dd SPBench-MV.tsv"},
        "prompt": {"impl": "vlmeval.dataset.SPBench.build_prompt"},
    },
    "VSTI-Bench": {
        "backend": "lmms_eval", "task": "vstibench_ee", "frames": 32, "items": 5736,
        "metric": "mean over 9 question types; MCQ accuracy, numeric MRA",
        "source": {"dataset": "Journey9ni/vstibench@0903eb0 (labels after the 2026-07-13 correction)",
                   "task": "lmms_tasks/vstibench_ee/vstibench.yaml (VLM-3R's task, token off, revision pinned)"},
        "prompt": {"impl": "VLM-3R vstibench utils.vsibench_doc_to_text",
                   "pre": "These are frames of a video.",
                   "mcq_post": "Answer with the option's letter from the given choices directly.",
                   "num_post": "Please answer the question using a single word or phrase."},
    },
    "VSTI-Bench-prefix": {
        "backend": "lmms_eval", "task": "vstibench_prefix_ee", "frames": 32, "items": 6042,
        "metric": "mean over 9 question types; MCQ accuracy, numeric MRA",
        "source": {"dataset": "Journey9ni/vstibench@27ff0a2 (labels before the correction; what Spatial-Interactor used)",
                   "task": "lmms_tasks/vstibench_ee/vstibench_prefix.yaml"},
        "prompt": {"impl": "VLM-3R vstibench utils.vsibench_doc_to_text (same as VSTI-Bench)"},
    },
    "SAT-Syn": {
        "backend": "lmms_eval", "task": "sat_syn_ee", "frames": None, "items": 4001,
        "metric": "accuracy (2 options)",
        "source": {"dataset": "array/SAT@bda5dde val split (synthetic), options shuffled as nv-njb/SAT does "
                              "(qa_transfer/make_sat_syn.py; the procedure reproduces nv-njb's SAT-Real 150/150)"},
        "prompt": {"impl": "lmms_eval/tasks/sat/utils.py sat_doc_to_text (same as SAT-Real)"},
    },
    "SAT-Real": {
        "backend": "lmms_eval", "task": "sat", "frames": None, "items": 150,
        "metric": "accuracy (2 options)",
        "source": {"dataset": "nv-njb/SAT@85f0ea843878d3a15e3935da7c1ede53276d171f (array/SAT test, options shuffled "
                              "once with seed 42)"},
        "prompt": {"impl": "lmms_eval/tasks/sat/utils.py sat_doc_to_text",
                   "post": "Please answer directly with only the letter of the correct option and nothing else."},
    },
}
