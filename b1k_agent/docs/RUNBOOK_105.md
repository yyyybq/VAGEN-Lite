# Runbook for server 105: Steps 1–4

These steps need server 105: GPUs, the simulator, `/nas/baiqiao`. They were written on 2026-10-21 from a cloud session that cannot reach 105, so **none of them has been run yet**. Run them yourself, or from a Claude Code session started on 105 (for example `claude remote-control` in `/nas/baiqiao/b1k_agent`), in this order. Each step lists what to record in STATUS.md.

## 0. Get the code onto 105

```bash
# The code lives in the b1k_agent/ folder of branch claude/b1k-agent-baseline of github.com/yyyybq/VAGEN-Lite.
# Make it a standalone repo (history of that folder only):
cd /tmp && git clone --branch claude/b1k-agent-baseline https://github.com/yyyybq/VAGEN-Lite b1k_src
cd b1k_src && git subtree split --prefix=b1k_agent -b b1k_agent_only
git clone /tmp/b1k_src -b b1k_agent_only /nas/baiqiao/b1k_agent
cd /nas/baiqiao/b1k_agent && git log --oneline | head
```

The agent server needs Python ≥3.11 with numpy<2, pyyaml, websockets, msgpack, pillow and httpx. The openpi venv already has those exact versions, so no install is needed:

```bash
cd /nas/baiqiao/b1k_agent && PYTHONPATH=. /nas/baiqiao/openpi/.venv/bin/python -m pytest -q   # pytest: see scripts/setup_agent_env.sh if missing
```

If you prefer a separate env, `scripts/setup_agent_env.sh` creates a NEW venv `/nas/baiqiao/envs/b1k_agent` with pinned versions. It does not touch the existing envs.

## Step 1: Takeover inventory (read-only, no GPU)

```bash
bash /nas/baiqiao/b1k_agent/scripts/takeover_check.sh     # writes docs/takeover/<timestamp>.md
```

It finds the smoke-test watcher, the most recent logs under `/nas/baiqiao/b1k_runs` and their state (not_started / running / finished / failed), and collects:

- nvidia-smi, the driver and EGL/Vulkan info;
- the BEHAVIOR-1K and openpi commits and the env package versions;
- disk usage, a datasets summary and a checkpoint summary.

It never starts or stops anything. Then:

1. **Read the two scripts in `/nas/baiqiao/b1k_runs/`.** If the env setup they do (for example the libstdc++ preload) is not inside the conda env's activate hooks, put it in a small file and export `B1K_ENV_SETUP=/path/to/it`. `eval/launch.py` sources that file before the evaluator, so the launcher reuses your setup instead of duplicating it.
2. Run these and record the asset id under `assets/`. It is the `--repo-id` the executor server needs; the official docs use `behavior-1k/2026-challenge-demos`.
   ```bash
   python scripts/inspect_checkpoint.py /nas/baiqiao/checkpoints/pi05_turn_on_the_radio --out docs/takeover/ckpt.json
   python scripts/inventory_datasets.py /nas/baiqiao/BEHAVIOR-1K/datasets --out docs/takeover/datasets.json
   ```
3. Check that `<datasets>/2026-challenge-task-instances/metadata/{B100_task_misc.csv,task.jsonl}` exist. The evaluator needs them, and they give `task_id` and the default `max_steps`.

Record in STATUS.md: the watcher state with its evidence, commits and versions, the GPU model / memory / driver, the checkpoint facts (asset id, action dim, norm-stats keys) and the dataset coverage.

## Step 2: Smoke test (wait for the watcher, never start a second one)

- **Pass criteria:** non-empty RGB and depth from a headless render, one task reset, and several steps without a crash.
- **If it failed:** diagnose the rendering, not around it.
  - Isaac Sim 5.1 needs a recent driver (check the Isaac Sim 5.1 requirements page for the minimum on Linux) and Vulkan. Check `vulkaninfo --summary` and `nvidia-smi`.
  - For headless EGL, check `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` and `__EGL_VENDOR_LIBRARY_FILENAMES`.
  - Check `~/.omnigibson_appdata` and the kit logs (`~/.nvidia-omniverse/logs/Kit/...`). The takeover report collects these paths.
- Write the conclusion in STATUS.md.

## Step 3: One official baseline episode (turning_on_radio)

Use a Dev instance for this first run (public_test index 10 = instance id 311). Report instances 0–9 are run only with frozen configs.

```bash
cd /nas/baiqiao/b1k_agent
# 3a. dry run: prints the commands, the chosen free GPU and max_steps
python -m eval.launch --config configs/L0_exec_only.yaml --task turning_on_radio --mode public_test \
    --instances 10 --wrapper default --seed 0 --write-video --dry-run
# 3b. official baseline path, no agent in between: evaluator -> executor server
python -m eval.launch ... --direct --run-id s3_direct_radio_311 --detach
# 3c. same instance through the agent server in L0 passthrough
python -m eval.launch --config configs/L0_exec_only.yaml ... --run-id s3_L0_radio_311 --detach
python -m eval.launch status /nas/baiqiao/b1k_runs/logs/s3_L0_radio_311
```

- 3b reproduces the documented baseline: `serve_b1k`-equivalent server, `receding_horizon`, `action_horizon 16`, DefaultWrapper 224 RGB.
- 3c should give the same action stream. If the simulator is deterministic at a fixed seed, the official JSONs should match. Check this before trusting any agent run (Step 4 measures determinism).

### Checklist (compare against `docs/PROTOCOL.md` and the logs)

| Item | Expected (from code) | Where to check |
|---|---|---|
| Cameras sent | `robot_r1::robot_r1:{zed_link,left_realsense_link,right_realsense_link}:Camera:0::rgb`, RGBA uint8, 224×224 (DefaultWrapper) or 720/480 (RGBD) | agent `events.jsonl` (dropped keys = none), executor log |
| Camera order into π0.5 | head, left wrist, right wrist → `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`, resize_with_pad 224 | openpi `eval_b1k_wrapper.py process_input` |
| Depth | `depth_linear` metres, only with RGBDFullResWrapper; π0.5 baseline ignores depth | obs keys |
| Proprio → state | 61-D proprio → 23-D state: base_qvel 0:3, trunk 53:57, left arm 3:10, left gripper sum 24:26, right arm 28:35, right gripper sum 49:51 | openpi `configs/robots/b1k.py` |
| Actions | 23-D: base vel [-1,1] → ±0.75 m/s, ±1 rad/s; torso/arms absolute joint pos; grippers [-1 closed, +1 open] | `eval/r1pro.yaml`, `steps.jsonl` ranges |
| Normalization | z-score from `<ckpt>/assets/<repo-id>/norm_stats.json` (quantile norm off) | executor server log |
| Action chunk | model horizon 32, execute 16, re-infer (receding_horizon) | executor metadata `model_action_horizon`, `action_horizon` |
| Reset | `{"reset": True}` once per rollout; wrapper buffers cleared | agent log |
| Timeout | `max_steps = int(1.5 × mean human length)`; JSON `steps = max_steps+1` on timeout | evaluator log "Setting timeout to be …" |
| Termination | success ends the rollout at once; `q_score.final` = newly satisfied goal literals / total (1.0 on success) | official JSON |

Done when the official `eval_output/json/turning_on_radio_311_0.json`, the video and our logs exist (`STATUS.md`: "simulator-run").

## Step 4: Throughput and determinism

```bash
bash scripts/gpu_sampler.sh start <run_dir>     # samples nvidia-smi every 2 s
# run 3c again with the same seed (determinism), then 2-3 instances in parallel on one GPU (separate ports)
bash scripts/gpu_sampler.sh stop <run_dir>
python scripts/measure_throughput.py <run_dir> [<run_dir> ...] --out docs/throughput_<date>.md
```

Write the measured numbers into PROTOCOL.md: the single-episode wall clock, the steps/s, how many simulator+policy instances fit on one GPU, and the peak memory. They give the estimated hours for Dev, Report and Full. Every later experiment size follows from them.

## Step 5 onward

- Executor candidates and the skill diagnosis: `scripts/skill_probe.py` and `docs/PROTOCOL.md` §Executor candidates.
- Dev set: `docs/PROTOCOL.md` §Dev tasks.
- Ladder and leave-one-out configs: `python -m eval.ablation ...`.
- Batches: `python -m eval.batch configs/batches/...`.
- Tables: `python -m eval.aggregate ...`.
- Failure distribution: `python -m eval.failures ...`.
