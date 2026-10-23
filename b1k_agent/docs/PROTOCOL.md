# Evaluation protocol (initial version, 2026-10-21)

This file records what the official sources say and what the code does. Where they differ, both are written down. It also fixes our own protocol choices.

**Check date:** 2026-10-21. The website (behavior.stanford.edu) was not reachable from the session that wrote this, because a proxy policy blocked it. Its pages are generated from `docs/challenge/*.md` in the BEHAVIOR-1K repo, so those files were read at several refs instead.

**Labels:** [code] verified in code · [docs] stated in docs · [inferred] our reasoning · NOT FOUND.

**Detailed evidence** (file:line citations): `docs/research/R1–R9`, summarised here. `R9_critic.md` lists the errata.

## 1. Sources and versions

| Item | Value |
|---|---|
| BEHAVIOR-1K repo | https://github.com/StanfordVL/BEHAVIOR-1K |
| Tag on 105 | `v3.9.3-post2` = `a8247a8cc1633fe1ca0cc66aa07243d46c64f155` (2026-10-05). **No plain `v3.9.3` tag exists** (`git ls-remote --tags`, 2026-10-21) |
| Other refs read | `v3.9.3-post1` `bd049de3`; `main` `7d6718cb` (2026-10-14); branch `2026/eval` `38d2c9fb` (2026-10-19); branch `challenge/group-time-budget` `ad934aa6` |
| What post1/post2 are | updates 09/28: "use `v3.9.3-post1` … will use `v3.9.3-post1` for final hidden-test evaluations". Updates 10/07: post2 = "a single bug fix for custom robot evaluation" [docs, main] |
| openpi fork | https://github.com/wensi-ai/openpi branch `behavior` = `0cc8e355f7bac0976db1cc3139b1ff0379feea60` (2026-06-28) |
| Website pages | https://behavior.stanford.edu/challenge/{index,evaluation,baselines,dataset,submission,updates}.html ← `docs/challenge/*.md` |
| Submission deadline | 2026-10-16 AoE, **already passed**. We do not submit anyway |

## 2. Tasks

- **100 tasks.** The 2025 set's 50 tasks keep indices 0–49 in the same order; indices 50–99 are new [docs + code: `task_data.json`]. `turning_on_radio` is index 0.
- **Task identity at evaluation:**
  - The evaluator sends only an integer `task_id` (`B100_task_misc.csv` column "Task ID", in the data root's `2026-challenge-task-instances/metadata/`) [code `evaluator.py:591`, `eval_utils.py:60-77`]. No language and no BDDL text are sent.
  - "BDDL task definitions can be used and are identical during evaluation" [docs `evaluation.md:28`].
  - Our agent maps `task_id` → name → BDDL file and task prompt offline (`agent/task_meta.py`).

## 3. Track and allowed inputs

- **One track** [docs `evaluation.md:5-23`]. Allowed policy inputs: RGB, depth and proprioception.
- **Not allowed:** ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose, or other simulator-only information.
- **Training** may use privileged information.
- **LLM / external API calls are allowed** if the observation restrictions hold. The participant pays, and their latency counts toward the time budget [docs `evaluation.md:30`].
- **What the evaluator actually sends** [code `evaluator.py:576-592`, `_batch_obs`]:
  - Three `robot_r1::robot_r1:<link>:Camera:0::rgb` (RGBA uint8): head `zed_link`, wrists `left_/right_realsense_link`.
  - `::depth_linear` (float32 metres), only with `RGBDFullResWrapper`.
  - `robot_r1::proprio` (61-D).
  - `robot_r1::cam_rel_poses` (camera poses relative to the base, 21-D).
  - `task_id`.
  - Every value has a leading `num_envs` dimension.
- **Wrappers:**
  - `DefaultWrapper` is the CLI default: 224×224 RGB, no depth.
  - `RGBDFullResWrapper`: head 720×720, wrists 480×480, RGB plus depth. The docs call it the "official RGB + depth challenge-track" wrapper. **All documented example commands use the default wrapper** [docs vs code].
- **Our whitelist** (`agent/fields.py`) passes exactly these keys. Anything else is dropped and logged, and `tests/test_loop_matrix.py::test_truth_isolation` checks that.

## 4. Instances and our split

| Split | Evaluator args | Instance IDs | Our use |
|---|---|---|---|
| Report | `--mode public_test --instance-indices 0..9` | 301–310 | frozen configs only; 1 rollout per instance; no tuning, no memory selection, no re-runs to pick the best |
| Dev | `--mode public_test --instance-indices 10..19` | 311–320 | tuning, ablations, failure analysis |
| Dev (extra) | `--mode train --instance-indices <id>` | raw train IDs: 0–299 for tasks 0–49, 1–300 for tasks 50–99 | more Dev episodes. The demos were collected on train instances, so scores there are optimistic |
| Hidden | `--mode hidden_test` | 321–340 | not available to us (`scene_test/private`) |

Sources:
- "Use indices 0-9 for reported evaluation results. Indices 0-19 are public test instances" [docs `evaluation.md:66,282`].
- `TEST_INSTANCE_IDS = range(301, 341)`; public is the first 20 [code `eval_utils.py:18-21`, `evaluator.py:73-87`].
- Discrepancy: the docs call hidden instances "indices 20-39"; the code uses `--mode hidden_test` with indices 0–19.

## 5. Episode limits and time

- **Step cap:**
  - `max_steps = int(1.5 × mean human demo length)` per task, from `2026-challenge-task-instances/metadata/task.jsonl` [code `evaluator.py:281-286`].
  - The env truncates on step `max_steps+1`, so the JSON shows `steps = max_steps+1` on timeout [code, off-by-one].
  - Exact values for all 100 tasks are in `configs/tasks/task_catalog_2026.tsv`. It is a third-party snapshot generated from the official `task.jsonl` with the same formula, cross-checked against `task_data.json` and the dataset mean; verify it on 105.
  - turning_on_radio = **3224** (the shortest task); assembling_gift_baskets = 39090 (≈10.86 h at 1 s/step).
  - The agent uses the catalog value when `--max-steps` is not given.
- **Control:** 30 Hz; physics at 120 Hz [code `eval_utils.generate_basic_environment_config`].
- **Success** (all goal literals true) ends the episode immediately [code].
- **Timing rules for final evaluation** [docs, main updates 10/09; 2026/eval docs]:
  - The episode budget is `max_steps × 1 s` of accounted time, excluding scene startup.
  - The first query is uncharged, up to 600 s.
  - Each query must take ≤ 600 s.
  - Up to 3 reconnects are allowed, and the same observation is resent.
  - A failed or over-budget rollout scores 0.
  - **Only branch `2026/eval` implements this** [code]. v3.9.3-post1/post2 and main have no wall-clock limit.
  - With single-port `--num-envs N`, 2026/eval applies ONE group deadline of `max_steps` seconds; multi-port `--policy-endpoints` uses per-rollout accounting [code; it contradicts the group-time-budget docs].
- **Our handling:**
  - The agent budget clock starts after the first query.
  - Strong-model calls are skipped when less than `loop.provider_min_remaining_s` remains.
  - Every latency is logged.
  - Agent runs use `--num-envs 1`, one rollout per process.
  - The aggregator also shows Q with the over-budget = 0 rule applied.
- **Synchronous loop:** the simulator waits for every reply [code `_step_fn`]. Thinking never changes the world; it only costs budget.
- **Speed figures:**
  - The docs benchmark 13.5–24.6 FPS with random actions on an RTX 4090, and a scene load of 150–300 s [docs `evaluation.md:335-368`].
  - **Our own throughput: NOT MEASURED** (§11).

## 6. Actions and robot

- **Robot:** R1Pro, `OmniGibson/omnigibson/eval/r1pro.yaml`, `name: robot_r1`. The evaluator overrides the base mass (250 kg) and the head camera aperture (40) [code].
- **Action, 23-D** [code `eval_utils.ACTION_QPOS_INDICES`, `r1pro.yaml`]:

  | Slice | Part | Command |
  |---|---|---|
  | 0:3 | base | velocity (vx, vy, wz), robot frame; input [-1, 1] → ±0.75 m/s and ±1 rad/s |
  | 3:7 | torso | absolute joint positions (rad) |
  | 7:14 | left arm | absolute joint positions (rad) |
  | 14 | left gripper | smooth, [-1 closed … +1 open] |
  | 15:22 | right arm | absolute joint positions (rad) |
  | 22 | right gripper | smooth, [-1 closed … +1 open] |

  An all-zero action is **not** a no-op: it drives the arms and torso to 0 rad. The hold action is `agent/actions.hold_action`.
- `grasping_mode: assisted`.
- **Proprio, 61-D:**

  | Slice | Part |
  |---|---|
  | 0:3 | base_qvel (robot frame) |
  | 3:10 | left arm qpos |
  | 10:17 | left arm qvel |
  | 17:20 | left eef pos (base frame) |
  | 20:24 | left eef quat (base frame) |
  | 24:26 | left gripper qpos |
  | 26:28 | left gripper qvel |
  | 28:35 | right arm qpos |
  | 35:42 | right arm qvel |
  | 42:45 | right eef pos |
  | 45:49 | right eef quat |
  | 49:51 | right gripper qpos |
  | 51:53 | right gripper qvel |
  | 53:57 | trunk qpos |
  | 57:61 | trunk qvel |

  There is no global pose.

## 7. Policy server protocol

**Official client** (`network_utils.py`, identical message format at post2 and 2026/eval):

- It waits for HTTP `GET /healthz` = 200, connects over websocket, and reads one msgpack metadata frame.
- `{"reset": True}` is sent once per rollout batch, with no reply.
- Each observation request gets a reply `{"action": float32 (B, 23)}`. A text frame is treated as an error.
- Optional chunk replay (`--replay-action-chunk-size K`, default 0): the server may return `action_chunk`. We do not, so the client falls back to one request per step.

**2026/eval additions:**

- the metadata frame must arrive within 10 s;
- `/healthz` must answer within 2 s even during inference;
- after a reconnect, the current observation is resent without a reset.

**Our server (`agent/server.py`):**

- runs `act` in a worker thread so health checks and pings keep working;
- detects a resent observation and replies with the cached action, without stepping;
- keeps episode state across reconnections.

Tests: `tests/test_protocol.py`.

## 8. Metrics and aggregation

- **Per-episode Q** (official, never recomputed by us) [code `metrics/task_metric.py:6-28`]:
  - `q_score.final` = 1.0 on success.
  - Otherwise: `max over goal options of (#goal literals FALSE at episode start and TRUE at episode end) / (#literals in that option)`.
  - Literals that are already true at the start count in the denominator but can never earn credit.
  - Credit is measured once, at the step where the episode ends (success or timeout), not as a maximum over time [code `evaluator.py:125-137`]. A literal satisfied and later undone earns nothing.
  - Breaking a literal that started true costs nothing, except that it can prevent full success.
  - The docs' wording "(predicates satisfied at episode end) / (total)" [docs `evaluation.md:231`] omits the "newly" condition.
- **Other fields in the result JSON:** `success`, `steps`, `agent_distance{base,left,right}`, `normalized_agent_distance`, `time{simulator_steps, simulator_time, normalized_time}`. On 2026/eval also `failure_reason` and a time breakdown.
  - File name: `<out>/json/{task}_{instance_id}_{rollout}.json`, with the resolved id (301…).
  - Video: `<out>/videos/…mp4`, 30 fps, one frame per executed step, layout 448×672.
- **Official aggregation** [docs]: mean over the instances of a task, then mean over 100 tasks. Missing tasks and instances count 0.
  - Code `score_utils.compute_final_q_score` divides each task by 20 (the public split size) although the docs say report 10 instances, so with 10 files the in-repo scorer halves the score [code vs docs].
  - The 2026 leaderboard code is NOT FOUND.
- **Our aggregation** (`eval/aggregate.py`, from logs only):
  - task score = mean of official `q_score.final` over that task's declared (instance, seed) set, with missing = 0;
  - overall = mean over tasks;
  - subsets are always labelled with their task list and never mixed with full-set means;
  - intervals: task-stratified bootstrap;
  - infrastructure errors are counted separately and excluded from Q (after at most 2 re-runs);
  - a second Q column applies the 2026 budget rule.
- **Secondary metrics:** full success rate; per-literal progress (from diagnostics, offline); predicate regressions (offline, via `diagnostics/replay_truth.py`); recovery attempts and successes; repeated invalid actions; strong-model calls, tokens and cost; wall-clock time.

## 9. Evaluator configuration we use

| Setting | Choice | Why |
|---|---|---|
| Evaluator code | v3.9.3-post2 (installed on 105) for Steps 3–4. Add the `2026/eval` evaluator (`38d2c9fb`, separate git worktree, `PYTHONPATH=<worktree>/OmniGibson`) once post2 runs, and use it for Report runs | post2 is the installed official tag. Only 2026/eval enforces the timing rules; obs, action, wrappers and metrics are identical to post2 [code diff] |
| Wrapper | Step 3 runs both: `DefaultWrapper` (what every baseline command uses) and `RGBDFullResWrapper` (the "official" RGB-D track). Freeze one before Dev, with the evidence | DECISION PENDING: π0.5 was trained on 720/480 frames resized to 224, so full-res + resize may match training better; depth is needed only by M-approach |
| `--max-steps` | omitted (official 1.5× default) | rule |
| `--num-envs` | 1 per evaluator process for agent runs | 2026/eval single-port groups share one deadline |
| Chunk replay | off (default) | rule default; temporal ensembling forbids it |
| Tiled rendering | off | final evaluation does not use it [docs] |
| Video | on | required by the rules; needed for failure analysis |
| Seeds | evaluator uses `DEFAULT_EVAL_SEED = 0` [code]. Our `run.seed` seeds only our side. Determinism will be measured (spec 6.2) | |

## 10. Baseline executor (π0.5, OpenPI route)

- **Official command** [docs `baselines.md:171-183`]:
  ```
  serve_b1k.py --robot b1k/R1Pro --task b1k/$TASK --repo-id $REPO_ID --policy.config pi05_b1k \
      --policy.dir $CKPT --control_mode receding_horizon --action_horizon 16 --port 8000
  ```
  with `REPO_ID=behavior-1k/2026-challenge-demos`.
- **Model and data path** [code, openpi fork]:
  - `pi05_b1k` = π0.5 with a 32-action chunk, starting from `gs://openpi-assets/checkpoints/pi05_base/params`.
  - Images: head → left wrist → right wrist, `resize_with_pad` to 224.
  - State: 23-D from proprio (base_qvel, trunk, arms, summed gripper widths).
  - Torso and arm actions are trained as deltas and returned as absolute values.
  - Normalisation: z-score from `<ckpt>/assets/<repo-id>/norm_stats.json`.
  - Serving: `receding_horizon` runs 16 of the 32 actions, then re-infers.
- **Language:** one fixed prompt per server process: `TASK_REGISTRY["b1k"]`, which holds only `turning_on_radio`: "Turn on the radio receiver that's on the table in the living room."
- **The training prompt was probably different** [code, pipeline; R9]:
  - The official replay writes the snake_case task name as the LeRobot task string (`lerobot_data_wrapper.py:359,477`).
  - openpi reads `meta/tasks.parquet` as the prompt, and its tokenizer maps `_` to spaces.
  - So the checkpoint most likely saw only "turning on radio", while it is served with the NL sentence. The released parquet was not inspected (HF blocked).
  - Implications:
    - L0 uses the official serving prompt (baseline fidelity).
    - The Step-5 probe must include the snake_case prompt.
    - Prompt swapping is not expected to steer this checkpoint.
- **GPU co-location:** the documented π0.5 server reserves 85% of a GPU (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`), and a BEHAVIOR simulator needs about 10.5 GB [3p, RAX docs]. Run them on separate GPUs, or lower the fraction. `eval/launch.py` defaults to separate GPUs.
- **Observation payload:** ≈7.5 MiB per query with `RGBDFullResWrapper` (3 RGBA images + 3 depth maps); ≈0.57 MiB with `DefaultWrapper`. On localhost it is negligible. For remote serving it counts against the 1 s/step budget.
- **Key mismatch** [code]:
  - The fork reads `robot::proprio` and `robot::robot:…` keys, but the official default robot is `robot_r1`.
  - As shipped, `serve_b1k.py` would fail with `KeyError('robot::proprio')` against the default evaluator.
  - Our `serve_b1k_prompted.py --rename_robot_from robot_r1` renames key names only. An unmerged fork PR #5 renames the config instead. **Check how the existing `/nas/baiqiao/b1k_runs` scripts handle this.**
- **Checkpoint on 105:** `/nas/baiqiao/checkpoints/pi05_turn_on_the_radio`, presumably the Drive checkpoint `1KojwNUz…` from baselines.md. **NOT INSPECTED.** `scripts/inspect_checkpoint.py` lists its `assets/<id>` (= `--repo-id`) and the norm-stats dimensions.
- The train-config name is not stored in the checkpoint; it is `pi05_b1k`.

## 11. Throughput and run-time estimates

**NOT MEASURED** (nothing has run yet). Step 4 measures:

- single-episode wall clock;
- scene load;
- steps/s;
- simulator + policy instances per GPU;
- peak memory.

Tool: `scripts/measure_throughput.py`.

Planning formula once measured, per (task, instance, seed):
- `T_ep ≈ T_load + steps_executed / FPS_eff + Σ provider latency`;
- `steps_executed ≤ max_steps`; failing episodes run to `max_steps`.

Upper bound from the docs alone, 1 GPU, 1 instance:
- the budget rule caps an episode at about `max_steps` seconds;
- turning_on_radio is therefore ≤ ~54 min of policy and simulator time plus load;
- a typical 2025 task is 10k–30k steps, i.e. hours at 1 step/s, but far less at the documented 13–25 FPS of the simulator alone.

## 12. Executor candidates (Step 5; details in R5 notes)

| # | Candidate | Coverage | Language | Status |
|---|---|---|---|---|
| B1 | Official 2026 π0.5 `turning_on_radio` (Drive) | 1 task | fixed task prompt | on 105 (presumably) |
| B2 | Official 2026 GR00T N1.7 `turning_on_radio` (Drive) | 1 task | task prompt | not downloaded |
| A1 | `gs://openpi-assets/checkpoints/pi05_base` (11.59 GiB, public GCS) | none (needs fine-tuning) | — | generic pretrain only |
| D2 | Comet (2025, 2nd place) `sunshk/openpi_comet` pi05-b1kpt50-cs32 | tasks 0–49 | global NL instruction (`fine_grained_level=0`; the fork has an optional subtask-instruction mode) | HF; 2025 conventions (base-velocity frame, qvel) changed in 2026 → validate |
| D1 | RLC (2025, 1st place) `IliaLarchenko/behavior_submission` / `behavior_50t_checkpoint` | tasks 0–49 | **task-ID embeddings, no text** | HF; cannot take sub-instructions |
| C2 | `nvidia/GR00T-N1.6-BEHAVIOR1k` | tasks 0–49 | task instruction | HF |
| F1 | `kmy17518/gr00t-n1.7-b1k-multitask` (community) | 100 tasks (per card) | task prompt | search snippet only; verify existence and license |
| — | official 100-task weights | — | — | **NOT FOUND** |

Classes are kept apart and never interchanged:
- generic π0.5 pretrain (A1);
- 2025 50-task adapted (D1, D2, C2);
- per-task (B1, B2, community E*);
- 100-task 2026 (F1, community; none official).

No released executor is trained on sub-task or skill language. That makes the Step-5 skill-instruction diagnosis (`scripts/skill_probe.py`) a precondition for M-skill and for instruction-type recovery.

Selection rule (spec): can be downloaded, loaded, and run through the official interface on 105, in that order.

## 13. Dev tasks (Step 6)

**No pure-navigation task exists** in the 100 BDDL goals: none mentions the agent, `inroom` or `grasped` [R6, derived]. The navigation probe is therefore a navigation-dominant task.

Credit caps:
- "cap" = the best Q reachable without full success, given goal literals that are already true at the start.
- Caps are heuristic, from the BDDL `:init`.

**Proposal A, full 2026 coverage** (R6 §5):

| Role | Task (idx) | Scene / rooms | ~max_steps | Goal literals | Cap |
|---|---|---|---:|---|---:|
| navigation-dominant | `turning_out_all_lights_before_sleep` (66, new) | house_single_floor, 8 room types | ~15.0k | 5 toggles; each room visit +0.2 | 1.0 |
| single-object | `turning_on_radio` (0) | house_double_floor_lower, living_room | ~3.2k | 1 | 1.0 |
| multi-object | `picking_up_trash` (1) | kitchen + living_room | ~7.9k | 3 cans inside the trash can | 1.0 |
| articulated | `store_honey` (83, new) | Rs_int; jar in kitchen, cabinet in entryway | ~10.2k | 1 (inside cabinet) | 1.0 |
| long-horizon | `putting_dishes_away_after_cleaning` (11) | kitchen | ~16.4k | 8 plates inside the cabinet + cabinet closed | ≈0.8 |

**Proposal B, a 2025 50-task executor** (indices 0–49 only): replace 66 by `bringing_water` (17, two rooms with a fridge, cap 0.67) or `carrying_in_groceries` (14), and replace 83 by `rearranging_kitchen_furniture` (8, cap 0.75) or `make_microwave_popcorn` (40).

**Proposal C, the Cracking-paper set** (cheaper: about 28.9k steps per sweep, with a published but non-official comparison): `picking_up_trash` (1), `make_rose_centerpieces` (56), `composting_waste` (90), `installing_smoke_detectors` (63), `make_microwave_popcorn` (40).

Keep `turning_on_radio` as the pipeline smoke task regardless; it is the only task with an official π0.5 checkpoint.

**Choice rule:** freeze A, B or C after Step 5, from the executor's task coverage. Use public_test indices 10–19 plus train instances; never 0–9.

**Approximate cost of one instance of each Proposal A task:** about 52.8k simulator steps. That is roughly 65 min at the documented 13.5 FPS, plus 150–300 s of scene load per task [inferred; to be replaced by Step-4 measurements].

**Scoring implications** for M-goal / M-plan (R6, R7):
- Secure the literals that start false first.
- Never undo achieved ones; credit is read at the final step only.
- End with a "close what was opened / restore initially-true literals" sweep. Initially-true literals give no partial credit but are required for the 1.0 of full success, and the jump from the cap (0.67–0.92) to 1.0 is large.

## 14. Page vs code discrepancies (keep checking)

1. **Version:** the pages say `v3.9.3` (no such tag). The updates say v3.9.2, then v3.9.3-post1, then post2 for custom robots.
2. **Hidden instances:** "indices 20-39" in the docs vs `--mode hidden_test 0-19` in the code. The number of hidden instances is 20 in the code vs "10 more" in the docs.
3. **Q-score wording:** "satisfied at end" (docs) vs "newly satisfied, max over options, 1.0 on success" (code).
4. **Scorer divisor:** 20 per task (code) vs 10 reported instances (docs).
5. **Timing rules:** announced in the docs, implemented only on the unreleased `2026/eval` branch. Single-port group deadline `max_steps` (code) vs `N × max_steps` (group-time-budget docs).
6. **Example commands:** they all use `DefaultWrapper`, not the "official" `RGBDFullResWrapper`.
7. **openpi fork:** robot name `robot` vs evaluator `robot_r1`. The fork doc omits `--repo-id` (norm stats would not be found).
8. **`obs_modalities`** in the robot config are overwritten by the wrapper's `camera_spec`.
9. **JSON wording:** "each successful rollout produces a JSON" (docs) vs a JSON for every rollout (code). The sample JSON shows `instance_id: 0`; the code writes 301….

## 15. Open questions

- Which evaluator branch and mode do the organisers actually use for hidden tests? It does not affect us beyond the timing rule.
- Is the NL task instruction officially allowed at test time? It is implied by the baselines and by `task_id`, but not stated.
- Exact per-task `max_steps`, from `task.jsonl` on 105.
- Determinism of a fixed-seed episode on 105 (spec 6.2).
- How the existing `/nas/baiqiao/b1k_runs` scripts set up the environment, and whether they already solve the robot-name mismatch.
