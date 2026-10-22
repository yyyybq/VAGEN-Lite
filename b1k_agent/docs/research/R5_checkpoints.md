# R5: Candidate executor checkpoints for the BEHAVIOR-1K 2026 Challenge

Research date: 2026-10-21. Read-only research; nothing was run against a simulator or GPU.

## 0. Sources and evidence labels

Local clones (read-only):
- **B1K** = `/home/user/stanfordvl/behavior-1k`, HEAD = tag `v3.9.3-post2` (`a8247a8c`, 2026-10-05). Other refs: `origin/main` (`7d6718cb`), `origin/2026-eval` (`38d2c9fb`), `origin/group-time-budget` (`ad934aa6`).
- **OPENPI** = `/home/user/wensi-ai/openpi`, branch `behavior`, `0cc8e355` (2026-06-28). This is the only commit in the clone (`git log` shows one merge commit).

Downloaded for reading only, into scratchpad `dl/`. No code from these was executed.
- `wensi-ai/Isaac-GR00T` (the GR00T fork linked from baselines.md). Shallow clone of all branches: `behavior` = `ace36d93` (default HEAD, 2026-07-02), `main` = `65cc4a19`, `multi-task` = `58e54602` (2026-07-12), `my` = `e30c6c6b` (2026-09-15), `andi/vector` = `c17188b2` (2026-07-24).
- `NVIDIA/Isaac-GR00T`. Blobless clone; branch `n1d6` = `9b37aa1c` (2026-06-30).
- `IliaLarchenko/behavior-1k-solution` @ `ca556f74` (2026-01-24): 1st place in 2025.
- `mli0603/openpi-comet` @ `4bb2aa7b` (raw files): 2nd place in 2025.
- `StanfordVL/b1k-baselines` tutorials (raw), which are included by `docs/challenge/archive/2025/baselines.md`.
- `starVLA/starVLA` @ master (raw README files).
- GCS JSON API listing of the public bucket `gs://openpi-assets` (reachable through `storage.googleapis.com`).

Blocked hosts: huggingface.co, drive.google.com and arxiv.org. Hugging Face model cards are known only from WebSearch snippets.

Evidence labels used throughout this report:
- **[code]**: verified by reading code or config.
- **[docs]**: stated in a docs or README file that I read.
- **[snippet]**: taken from a WebSearch result snippet only; not verified.
- **[inferred]**: my own reasoning from the cited facts.
- **NOT FOUND**: the fact is not present in the places I looked, which are listed.

---

## 1. TL;DR: candidate checkpoints

| # | Candidate | Category | Task coverage | Robot / interface | Obtainable? | Evidence |
|---|---|---|---|---|---|---|
| A1 | `gs://openpi-assets/checkpoints/pi05_base` | generic pi0.5 base | generic pretrain (no BEHAVIOR) | none (needs fine-tuning) | yes, public GCS, 11.59 GiB | [code] + GCS listing |
| B1 | Official 2026 π0.5 baseline `turning_on_radio` (Google Drive `1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2`) | per-task 2026 | 1 task (`turning_on_radio`) | R1Pro, OpenPI fork `pi05_b1k` | Drive link (size unknown) | [docs] |
| B2 | Official 2026 GR00T N1.7 baseline `turning_on_radio` (Google Drive `1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789`) | per-task 2026, GR00T route | 1 task | R1Pro, GR00T fork `NEW_EMBODIMENT` | Drive link (size unknown) | [docs] |
| C1 | `nvidia/GR00T-N1.7-3B` | generic GR00T base | generic pretrain | needs fine-tuning (R1Pro via `NEW_EMBODIMENT`) | HF; backbone `nvidia/Cosmos-Reason2-2B` is gated | [docs] |
| C2 | `nvidia/GR00T-N1.6-BEHAVIOR1k` | 2025 50-task adapted (GR00T N1.6) | 50 tasks (2025 = 2026 tasks 0-49) | R1Pro, tag `BEHAVIOR_R1_PRO`, own sim env, 82-D state | HF | [docs] NVIDIA n1d6 README |
| D1 | `IliaLarchenko/behavior_submission` (4 ckpts) and `IliaLarchenko/behavior_50t_checkpoint` | 2025 50-task adapted (pi0.5 variant "PI_BEHAVIOR") | 50 tasks; task-ID embeddings, **no text** | `robot_r1::` keys, `task_id` obs | HF | [docs] RLC README; [code] RLC wrapper |
| D2 | `sunshk/openpi_comet/pi05-b1kpt50-cs32` (also `pi05-b1kpt12-cs32`; `sunshk/openpi_comet_pytorch`) | 2025 50-task adapted (stock pi0.5) | tasks 0-49 (pt50) / 12 tasks (pt12) | `robot_r1::` keys, global NL instruction prompt | HF | [docs] Comet README; [code] Comet config |
| E* | Community per-task / skill fine-tunes (fastwalker1118, JackLiu0406/b1k-checkpoints, Hoshipu, Shawn3636, 0Corvid0) | per-task | 1 to 25 tasks | mostly RLC "PiBehavior" lineage | HF (some gated) | [snippet] |
| F1 | `kmy17518/gr00t-n1.7-b1k-multitask` | 100-task 2026 (GR00T N1.7, community) | 100 tasks (per card) | R1Pro (`NEW_EMBODIMENT`) | HF (per snippet) | [snippet] + [code] related fork branch |
| F2 | `JackLiu0406/meta-SFT-checkpoints` (meta100, step69999) | 100-task 2026 (pi0.5 PiBehavior, community) | 100 tasks | RLC lineage | **private** per snippet | [snippet] |
| -- | **Official 100-task 2026 weights** | -- | -- | -- | **NOT FOUND** | baselines.md (all 4 refs), OPENPI, GR00T fork (all branches), search |

---

## 2. What the official docs say (B1K `docs/challenge/baselines.md`)

`docs/challenge/baselines.md` is byte-identical at HEAD (`v3.9.3-post2`), `origin/main`, `origin/2026-eval` and `origin/group-time-budget`. `git diff HEAD <ref> -- docs/challenge/baselines.md` returned 0 lines for all three refs. **[code/docs]**

Baseline methods (`docs/challenge/baselines.md:3-6`) **[docs]**:
```
For the 2026 BEHAVIOR Challenge, we provide starter training and evaluation pipelines for two baseline methods:
- **[π0.5](https://www.physicalintelligence.company/blog/pi05)** — fine-tuned with a challenge fork of [OpenPI](https://github.com/wensi-ai/openpi)
- **[GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B)** — fine-tuned with a challenge fork of [Isaac-GR00T](https://github.com/wensi-ai/Isaac-GR00T)
```
The challenge index repeats this: `docs/challenge/index.md:115-116` ("Baselines | π0.5 (pi0.5) and GR00T N1.7") and `index.md:156`.

Provided checkpoints (`docs/challenge/baselines.md:12-18`) **[docs]**:
```
## Provided checkpoints
We provide fine-tuned checkpoints so participants can run evaluation without training:
| Task               | π0.5                                     | GR00T N1.7                               |
| `turning_on_radio` | [checkpoint](https://drive.google.com/file/d/1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2/view?usp=sharing) | [checkpoint](https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view?usp=sharing) |
```
- Only **one task** is covered: `turning_on_radio`, which is task 0.
- No Hugging Face model repo is given for either baseline checkpoint.
- No revision, size, norm-stats asset id or license is stated for either Drive file. **NOT FOUND** in baselines.md, OPENPI, or the GR00T fork.
- drive.google.com is blocked here (curl returned code 000), so the file contents and sizes are unknown.

Placeholders (`baselines.md:32-36`) **[docs]**:
```bash
export TASK_NAME=turning_on_radio              # any challenge task
export DATA_ROOT=~/2026-challenge-demos        # local LeRobot v3 root (data/, meta/, videos/)
export DATASET_PATH=$DATA_ROOT                 # OpenPI dataset_root
export REPO_ID=behavior-1k/2026-challenge-demos  # OpenPI repo_id / norm stats (HF hub id, not a local path)
```

π0.5 training: `baselines.md:110` **[docs]**:
> The `pi05_b1k` config fine-tunes the π0.5 base model (`gs://openpi-assets/checkpoints/pi05_base/params`) on the R1Pro robot with a 32-step action horizon.

π0.5 serving: `baselines.md:171-183` **[docs]**:
```bash
uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task b1k/$TASK_NAME \
    --repo-id $REPO_ID \
    --policy.config pi05_b1k \
    --policy.dir $PATH_TO_CKPT \
    --control_mode receding_horizon \
    --action_horizon 16 \
    --port 8000
```

GR00T: `baselines.md:205-206` says "The N1.7 backbone `nvidia/Cosmos-Reason2-2B` is gated." `baselines.md:257` says:
> fine-tunes the GR00T N1.7 base model (`nvidia/GR00T-N1.7-3B`) on the R1Pro robot with a 16-step action horizon, tuning the projector and diffusion action head while keeping the vision encoder and LLM frozen. The R1Pro embodiment ... is defined in `examples/b1k/r1pro.py` and registered under the `NEW_EMBODIMENT` tag.

GR00T serving: `baselines.md:288-295` runs `python scripts/b1k/serve_b1k.py --model-path $PATH_TO_CKPT --modality-config-path examples/b1k/r1pro.py --embodiment-tag NEW_EMBODIMENT --host 127.0.0.1 --port 8000`, "applying temporal ensembling by default". **[docs]**

Evaluation for both baselines: `baselines.md:306-311`:
`python -m omnigibson.eval.eval --task-name $TASK_NAME --host 127.0.0.1 --port 8000 --output-dir $LOG_PATH --write-video`
No `--robot-config` is passed, so the default `OmniGibson/omnigibson/eval/r1pro.yaml` is used (`evaluator.py:55,330`). **[docs+code]**

**2025 baseline numbers in baselines.md: NOT FOUND.**
- The 2026 `baselines.md` contains no success rates or Q-scores.
- `docs/challenge/archive/2025/baselines.md:1-12` only `include-markdown`s three tutorials from `StanfordVL/b1k-baselines`. I fetched all three (`tutorials/openpi.md`, `openvla.md`, `il_lib.md`) and none contains numbers.
- `b1k-baselines/README.md` is 2 lines.
- The 2025 numbers that do exist in the repo are in the leaderboard JSONs (§9).

---

## 3. Category A: generic π0.5 base weights

**A1. `gs://openpi-assets/checkpoints/pi05_base`**
- **Where it is referenced:** `OPENPI/README.md:68` lists it as the "Base π₀.₅ model ... for fine-tuning". The baseline config loads `params` from it at `OPENPI/src/openpi/training/config.py:769`:
  `weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`. **[code]**
- **Size:** 29 objects totalling 12,441,749,581 bytes (11.59 GiB), objects dated 2025-09-08. This comes from the public GCS JSON API listing `storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/`, retrieved 2026-10-21. **[verified via GCS API]**
- **Norm stats:** the bundled assets are `assets/{arx, arx_mobile, droid, fibocom_mobile, franka, trossen, trossen_mobile, ur5e, ur5e_dual}/norm_stats.json` (same GCS listing). There are **no R1Pro/B1K norm stats** in the base, so fine-tuning must compute new ones. **[verified via GCS API]**
- **Other BEHAVIOR checkpoints in the bucket:** none. The `checkpoints/` prefixes in `gs://openpi-assets` are pi05_base, pi05_droid, pi05_droid_jointpos, pi05_libero, the pi0* variants, polaris and roboarena. **[verified via GCS API]**
- **Architecture:** `Pi0Config` defaults are `paligemma_variant="gemma_2b"`, `action_dim=32`, `action_horizon=50`. With `pi05=True`, `max_token_len` is 200 and `discrete_state_input=True` (`OPENPI/src/openpi/models/pi0_config.py:21-41`). **[code]**
- **License:** the OPENPI repo ships an Apache-2.0 `LICENSE` and a `LICENSE_GEMMA.txt` (Gemma Terms of Use). The README states no separate weights license. A HF snippet says `lerobot/pi05_base` was tagged `license: gemma` ([source](https://huggingface.co/lerobot/pi05_base/commit/975848a74c17ca1f646570aec93c761941f20be2)). **[code + snippet]**
- **Usefulness as an executor:** none without fine-tuning, because it has no R1Pro action space or BEHAVIOR data. **[inferred]**

---

## 4. Category B1: official 2026 π0.5 per-task checkpoint and OpenPI fork interface

**Checkpoint:** Drive `1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2`, `turning_on_radio` only (`baselines.md:18`). Revision, size and step are **NOT FOUND**.

Interface as defined by the OpenPI fork (`0cc8e355`). All points are **[code]** unless marked.

- **Robot.** `src/openpi/configs/robots/b1k.py:6-50` defines R1Pro, registered as `b1k/R1Pro`.
- **Cameras.** Three RGB streams, each with dataset `resolution=[240, 240]`:
  - `image_0` = head `robot::robot:zed_link:Camera:0::rgb`
  - `image_1` = left wrist `robot::robot:left_realsense_link:Camera:0::rgb`
  - `image_2` = right wrist `robot::robot:right_realsense_link:Camera:0::rgb`
  - Source: `b1k.py:10-27`. The serving wrapper resizes and pads to 224x224 (`src/openpi/shared/eval_b1k_wrapper.py:17,82`).
- **Proprio / state.** The state is 23-D, selected from the evaluator's 61-D `::proprio` vector (`b1k.py:39-46`):
  - `base_qvel` [0:3]
  - `trunk_qpos` [53:57]
  - `left_arm_qpos` [3:10]
  - `left_gripper_qpos` [24:26], summed to 1 value
  - `right_arm_qpos` [28:35]
  - `right_gripper_qpos` [49:51], summed to 1 value
  - Finger summing: `src/openpi/policies/b1k_policy.py:34-41`.
  - pi0.5 tokenizes the state into the prompt (`discrete_state_input`, `pi0_config.py:40-41`).
- **Action.** 23-D, `action_dim=23` (`b1k.py:30-38`):
  - base [0:3]
  - torso [3:7], `needs_delta_comp`
  - left_arm [7:14], `needs_delta_comp`
  - left_gripper [14]
  - right_arm [15:22], `needs_delta_comp`
  - right_gripper [22]
  - The model is trained on torso/arm **deltas relative to current joint state** and converted back to **absolute** joint targets at output (`config.py:435-442`; `transforms.py:248-290`). The model pads to 32 dims and the output is truncated to 23 (`b1k_policy.py:110-112`).
- **Action units [code + inferred].** The evaluator's default `r1pro.yaml:35-80` uses:
  - arms and trunk: absolute `JointController` position commands with `command_input_limits: null`, i.e. raw joint positions in rad;
  - base: `HolonomicBaseJointController` velocity with input limits [-1,1] mapped to output [-0.75,0.75] (x, y) and [-1,1] (yaw);
  - grippers: `MultiFingerGripperController mode: smooth`.
  - The challenge update at `updates.md:45` says base `state[0:3]` is now robot-local velocity, which "matches the action convention used by the R1Pro base controller".
- **Norm stats.**
  - Computed with `uv run scripts/compute_norm_stats.py pi05_b1k --data.repo_id=$REPO_ID ...` and written to `outputs/assets/pi05_b1k/$REPO_ID` (`baselines.md:101-106`).
  - Saved into each checkpoint under `assets/<asset_id>/` (`src/openpi/training/checkpoints.py:71-77`).
  - At serve time, loaded from `<ckpt>/assets/<repo_id>`, where `repo_id = args.repo_id or args.task` (`scripts/b1k/serve_b1k.py:53-57`; `src/openpi/policies/policy_config.py:59-64`).
  - Normalization is mean/std, not quantile: `use_quantile_norm=False` (`config.py:455`).
- **Chunk length.**
  - Model `action_horizon=32` (`config.py:758`).
  - Serving default `--action_horizon 16` (re-plan every 16 steps) with `max_len = config.model.action_horizon` (`serve_b1k.py:36-37,68-75`).
  - Control modes: `receding_horizon` (default), `receding_temporal`, `temporal_ensemble` (`eval_b1k_wrapper.py:356-365`).
- **Language.**
  - Training uses `prompt_from_task=True` (`config.py:764`). The prompt is the LeRobot `tasks.parquet` task string looked up by `task_index` (`src/openpi/training/data_loader.py:150-153`; `src/openpi/training/lerobot_compat.py:17-25`; `transforms.py:356-368`).
  - Serving uses a single fixed prompt: `TASK_REGISTRY[task_bucket][task_name]` (`serve_b1k.py:46-47`). It is baked into the wrapper as `self.text_prompt` and sent with every inference (`eval_b1k_wrapper.py:88-96`).
  - The registry has only one entry (`src/openpi/configs/tasks/b1k.py:4-6`):
    ```python
    TASKS = {
        "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
    }
    ```
  - **There is no per-request prompt override path.** The websocket server passes the obs dict straight to `policy.act(obs)` (`src/openpi/serving/websocket_b1k_server.py:65-74`).
- **Server protocol.**
  - Responds `{"action": ..., "server_timing": {...}}` only (`websocket_b1k_server.py:76-86`).
  - There is no `action_chunk` field, so evaluator chunk replay (`--replay-action-chunk-size`) would fall back to one request per step (`docs/challenge/evaluation.md:107-122`).
  - It handles a `"reset"` message (`websocket_b1k_server.py:66-68`) and `/healthz` (`:106-114`).
- **How the docs run it:** see §2 (`baselines.md:166-185`).
- **Licensing of B1:** **NOT FOUND**. It is presumably derived from pi05_base and the BEHAVIOR dataset terms. **[inferred]**

Caveats found by reading the code (**[code] + [inferred]**; not executed):
1. **Observation-key namespace mismatch.**
   - The OpenPI fork expects `robot::proprio` and `robot::robot:zed_link:Camera:0::rgb`: `RobotConfig(name="robot", ...)` in `b1k.py:7,12`, and `obs[f"{self.robot.name}::proprio"]` in `eval_b1k_wrapper.py:72`.
   - The official evaluator's default robot is named `robot_r1` (`OmniGibson/omnigibson/eval/r1pro.yaml:2`; `docs/challenge/evaluation.md:168` "The default name is `robot_r1`").
   - It flattens observations as `<robot.name>::...` (`eval_utils.py:235-247`, `evaluator.py:576-592`), with no renaming before sending (`policies.py:88-92`).
   - The GR00T fork and both 2025 winners use `robot_r1::...` (§5, §6).
   - Unless the evaluator is run with a custom robot config named `robot`, the OpenPI fork as checked in would likely `KeyError` on `robot::proprio`. This needs verification on a live run.
2. **Language train/serve mismatch risk.**
   - Training prompt = `tasks.parquet` string.
   - The GR00T fork documents that LeRobot's `tasks.parquet` holds the **snake_case task id** (e.g. `turning_on_radio`), while `meta/tasks.jsonl` holds the NL description (`wensi-ai/Isaac-GR00T@e30c6c6:gr00t/data/b1k_prompts.py` docstring; `getting_started/b1k.md:294`).
   - The OpenPI server prompts with the NL description from `TASK_REGISTRY`.
   - If the GR00T fork is right about `tasks.parquet`, the π0.5 baseline is served with a different string than it was trained on. **[inferred; tasks.parquet not inspected — HF blocked]**
3. **Only one task is registered.** Serving any `--task b1k/<other>` raises `KeyError` until the task is added to `TASK_REGISTRY` (`serve_b1k.py:47`). **[code]**
4. **Single-task defaults.** The default config is single-task: `repo_id="turning_on_radio"` and a hard-coded `dataset_root="/viscam/u/shiyuc/openpi/2026-challenge-demos/b1k/turning_on_radio"` (`config.py:760-763`). The data loader has no task filter, so pointing `dataset_root` at the full 100-task root would train multi-task (`data_loader.py:140-155`). **[code]**
5. **State definition drift.**
   - The fork commit (2026-06-28) predates the dataset fix of 07/27/2026, which made `observation.state[0:3]` robot-local base velocity (`docs/challenge/updates.md:37,45`).
   - It also predates the 08/24/2026 qvel fix of `state[10:17]`, `[26:28]`, `[35:42]`, `[51:53]`, `[57:61]` (`updates.md:7,15`).
   - The R1Pro state used by pi0.5 includes `base_qvel` [0:3].
   - Any checkpoint trained before 07/27 saw the old base-velocity convention. The training date of the Drive checkpoint is unknown. **[inferred]**
6. **Docs vs code CLI.** baselines.md passes `--policy.config ...` directly, while OPENPI `docs/b1k.md:246-251` inserts a `policy:checkpoint` subcommand. In this code `Args.policy` is a plain `Checkpoint` dataclass, not a union (`serve_b1k.py:14-31`), so the baselines.md form matches the code. **[code]**

---

## 5. GR00T route (C1, B2, C2)

### C1. `nvidia/GR00T-N1.7-3B` (generic base)
- Named in `baselines.md:6,257,265` as the base for fine-tuning. **[docs]**
- Fork README facts **[docs]** (`wensi-ai/Isaac-GR00T@ace36d93:README.md`):
  - `:71`: "GR00T N1.7 is fully commercially licensable under Apache 2.0."
  - `:97`: the VLM backbone is Cosmos-Reason2-2B (Qwen3-VL).
  - `:253`: lists `nvidia/GR00T-N1.7-3B` as "Base model (3B params)".
- The fork code `LICENSE` is Apache-2.0. **[code]**
- The backbone is gated (`baselines.md:205-206`). **[docs]**
- The `my`-branch guide says "the trainable action head is ~1.6B of the 3.1B parameters" (`wensi-ai/Isaac-GR00T@e30c6c6:getting_started/b1k.md:85`). **[docs]**
- A HF snippet says some community mirrors describe the license differently ([search](https://huggingface.co/SamLegion888/GR00T-N1.7-3B)). **[snippet]**

### B2. Official 2026 GR00T N1.7 `turning_on_radio` checkpoint
- Drive `1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789` (`baselines.md:18`). Size and step: **NOT FOUND**.
- In the fork's `my` branch the guide still reads "We provide a GR00T N1.7 checkpoint for: turning_on_radio task [here](add checkpoint link)." (`e30c6c6:getting_started/b1k.md:91-93`). **[docs]**

Interface, `wensi-ai/Isaac-GR00T` branch `behavior` (`ace36d93`, the default branch, cloned by `baselines.md:198`). All points are **[code]** unless marked.
- **Embodiment.** `examples/b1k/r1pro.py:41-118` registers `NEW_EMBODIMENT` with robot name `robot_r1`. Observation keys are `robot_r1::robot_r1:zed_link:Camera:0::rgb` and the left/right realsense equivalents (`r1pro.py:42-47`), which **match the official evaluator**.
- **State** = slices of the 61-D proprio: `base_qvel[0:3]`, `torso[53:57]`, `left_arm[3:10]`, `left_gripper[24:26]`, `right_arm[28:35]`, `right_gripper[49:51]` (`examples/b1k/r1pro.json:2-27`).
  - The fingers are 2-D here, not summed. The documented totals are 3+4+7+2+7+2 = 25 dims. **[inferred arithmetic]**
- **Action.** 23-D absolute joint targets (`r1pro.py:3-24` docstring; `r1pro.json:28-53`).
  - base, left_gripper and right_gripper are ABSOLUTE.
  - torso and arms are RELATIVE to their state key during learning (`r1pro.py:73-109`).
- **Chunk.** `delta_indices=list(range(16))`, a 16-step chunk (`r1pro.py:63-65`), consistent with `baselines.md:257`.
- **Serving.**
  - `scripts/b1k/serve_b1k.py` defaults to `control_mode="temporal_ensemble"` (`:32`).
  - The wrapper defaults are `action_horizon=10`, `max_len=50`, `temporal_ensemble_max=5` (`gr00t/eval/eval_b1k_wrapper.py:73-86`).
  - Images are resized and padded to 224x224 (`eval_b1k_wrapper.py:72,142`).
- **Language (critical).**
  - Training key: `annotation.human.task_description` with `original_key: "task_index"` (`r1pro.json:65-69`; `r1pro.py:111-114`).
  - Serving on the `behavior` branch builds the wrapper **without a text prompt** (`serve_b1k.py:78-83`), so the wrapper default `text_prompt="pick up the object and place it on the table"` is used for every request (`eval_b1k_wrapper.py:70,154`).
  - The `my` branch later documents that "Checkpoints trained before `--prompt-source` existed saw task names under the `task_description` key" (`e30c6c6:getting_started/b1k.md:315`).
  - The official baseline therefore effectively ignores language at serve time. **[code + docs]**
- **Norm stats.**
  - `meta/stats.json` is "generated automatically on the first training run" (`baselines.md:217`). **[docs]**
  - Serve-time files include `statistics.json`, `processor_config.json` and `embodiment_id.json`, among others (`e30c6c6:getting_started/b1k.md:261`). **[docs]**
- **Size.** An eval-only checkpoint copy is "~7 GB each" (`e30c6c6:getting_started/b1k.md:261`). A full resumable checkpoint is "34 GB" or "35.7 GB" (`:248`, `:270`). **[docs]**

Other branches of the GR00T fork (not referenced by the official docs) **[code]**:
- **`my` (`e30c6c6`, 2026-09-15).**
  - Adds `gr00t/data/b1k_prompts.py` and `examples/b1k/tasks.jsonl`: 100 lines in the form `{"task_index", "task_name", "task"}`, described as a "verbatim copy of the dataset's meta/tasks.jsonl".
  - The prompt is resolved **per request from the evaluator's `task_id`** (`gr00t/eval/eval_b1k_wrapper.py:16,95-97,183-188` on that branch).
  - It also adds `--text-prompt` to fix the prompt verbatim, plus `--task-name` and `--prompt-source task_name|task_description` (`scripts/b1k/serve_b1k.py:43-93` on that branch).
  - It adds HF uploader scripts (`scripts/b1k/hf_checkpoint_uploader.py`, `hf_resume_checkpoint_uploader.py`).
  - `--task-names` restricts training to selected tasks, and dropping it trains "on every task under the root" (`getting_started/b1k.md:111-117,188`).
- **`multi-task` (`58e54602`, 2026-07-12).** Commit message: "cap v3.0 per-worker parquet/video caches to bound multi-task training memory".
- **`andi/vector` (`c17188b2`).** Commit message: "Fix B1KPolicyWrapper batched (N-env) inference".

### C2. `nvidia/GR00T-N1.6-BEHAVIOR1k` (2025 50-task, GR00T N1.6)
Source: `NVIDIA/Isaac-GR00T@9b37aa1c` (branch `n1d6`).
- **What it is.** `examples/BEHAVIOR/README.md:12`: "We provide a checkpoint: `nvidia/GR00T-N1.6-BEHAVIOR1k` which is post-trained on all 50 tasks." **[docs]**
- **Reported results** (`README.md:14-69`) **[docs]**:
  - The table reports "Task Progress" (described as "a denser metric than `Q Score`").
  - Average: N1.6 **26.30%** vs Pi0.5 **11.30%** (`:69`).
  - `turning_on_radio`: 43.33% vs 16.67% (`:25`).
  - Per-task post-training: `turning_on_radio` Task Progress 80.56%, Q 0.70 (`:76`).
- **Embodiment.** Tag `BEHAVIOR_R1_PRO` (`gr00t/data/embodiment_tags.py:52` on n1d6). **[code]**
- **Inputs** (`gr00t/configs/data/embodiment_configs.py:211-301` on n1d6) **[code]**:
  - video `observation.images.rgb.{head,left_wrist,right_wrist}_256_256`;
  - an **82-D state** including `robot_pos`, `robot_ori_cos/sin`, 2-D orientation, linear/angular velocity, arm qpos with sin/cos, EEF pos/quat, gripper qpos and trunk qpos.
- **Action.** A **32-step** chunk (`delta_indices=list(range(0,32))`) over base, torso, left_arm, left_gripper, right_arm, right_gripper. Torso and arms are RELATIVE; base and grippers are ABSOLUTE. **[code]**
- **Language.** Key `annotation.human.coarse_action`. At eval the env sets it to the task instruction: `obs["annotation.human.coarse_action"] = env.task_instruction` (`gr00t/eval/sim/BEHAVIOR/behavior_env.py:253,309`). **[code]**
- **How the docs run it** (`README.md:103-148`) **[docs]**:
  - It uses NVIDIA's own gym env and `rollout_policy.py`, with BEHAVIOR-1K on branch `feat/task-progress`.
  - The client uses `--n_action_steps 8`.
  - Training data is `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` (`:84`).
- **Compatibility [inferred].** This is **not** the official 2026 websocket evaluator interface. The 82-D state includes global robot pose, which the official evaluator's `robot_r1::proprio` does not obviously provide (see `r1pro.yaml:13-28` proprio list). An adapter would need its own state reconstruction, so use with the official evaluator is unverified.

---

## 6. Category D: 2025 50-task adapted weights (cover 2026 tasks 0-49)

Task-ID compatibility **[code]**: the first 50 entries of the 2026 `docs/challenge/task_data.json` are identical, in the same order, to the 50 entries of `docs/challenge/archive/2025/task_data.json` (checked programmatically: `2026[:50] == 2025` → True). 2025-trained task-ID-conditioned models therefore map onto 2026 task IDs 0-49 only. Tasks 50-99 are new.

Main caveat **[docs + inferred]**: the 2026 dataset and evaluator changed definitions after 2025.
- `state[0:3]` base velocity is now robot-local (`updates.md:45`).
- The qvel fields were corrected (`updates.md:15`).
- The evaluator uses a heavier base (`EVAL_BASE_LINK_MASS = 250.0`) and a head camera horizontal aperture of 40.0 (`evaluator.py:53-54`).
- The default timeout is now 1.5x mean human length (`evaluation.md:78,282`); the 2025 docs used 2x (`archive/2025/evaluation.md:192`).

2025 models may therefore see distribution shift. Their 2025 scores are not 2026 scores.

### D1. Robot Learning Collective (1st place 2025)
Source: `IliaLarchenko/behavior-1k-solution@ca556f74`, Apache-2.0 `LICENSE`.
- **Checkpoints** **[docs]**:
  - `README.md:186`: "All checkpoints are available on HuggingFace in the same repository: https://huggingface.co/IliaLarchenko/behavior_submission".
  - It holds 4 checkpoints, each fine-tuned on a task subset: ckpt1 has 20 tasks, ckpt2 has 16, ckpt3 has 13, ckpt4 has task 40 (`:191-196`).
  - The task→checkpoint switch is `task_checkpoint_mapping.json` (`:198-205`).
  - The intermediate 50-task checkpoint is https://huggingface.co/IliaLarchenko/behavior_50t_checkpoint, with the authors' guess of "around 15-20% q-score" (`:207-208`).
- **Model** ("PI_BEHAVIOR", built on pi0.5) **[docs]**:
  - "Replaced language model with 50 trainable task embeddings (no text processing)" (`README.md:15`).
  - Stage/subtask prediction is used as "System 2" (`:27-29`).
  - It predicts 30 actions, executes 26 and keeps 4 for inpainting (`:32`).
  - It trains on 224x224 RGB plus proprio only (`:25`).
- **Model code** **[code]**:
  - `num_tasks: int = 50`, `action_horizon: int = 30`, `action_dim: int = 32` (`src/b1k/models/pi_behavior_config.py:51-56`).
  - The wrapper reads `obs["robot_r1::proprio"]` and the `robot_r1::robot_r1:*:Camera:0::rgb` keys (`src/b1k/shared/eval_b1k_wrapper.py:107-111`).
  - It reads `obs["task_id"]` to switch tasks and checkpoints (`:193-196`).
  - It sends `tokenized_prompt = [task_id, current_stage]`, i.e. **no natural language** (`:164-171`).
  - The official 2026 evaluator also sends `task_id` (`evaluator.py:591`).
- **Language conditioning:** none. It is task-ID plus stage. It **cannot be prompted with subgoal text** without retraining. **[code]**
- **Submodule pin.** The repo's `openpi` submodule is `https://github.com/wensi-ai/openpi.git` branch `behavior` (`.gitmodules`). **[code]**

### D2. Team Comet (2nd place 2025, NVIDIA Research)
Source: `mli0603/openpi-comet@4bb2aa7b`, Apache-2.0 `LICENSE`.
- **Model zoo** (`README.md:113-118`) **[docs]**:
  - `pi05-b1kpt12-cs32`: tasks `0,1,6,17,18,22,30,32,34,35,40,45`, chunk 32 → https://huggingface.co/sunshk/openpi_comet/tree/main/pi05-b1kpt12-cs32
  - `pi05-b1kpt50-cs32`: tasks `0-49`, chunk 32 → https://huggingface.co/sunshk/openpi_comet/tree/main/pi05-b1kpt50-cs32
  - `README.md:71` claims "Q Score of 0.345 (Public Validation)" for the post-challenge models.
  - A PyTorch copy is at https://huggingface.co/sunshk/openpi_comet_pytorch ([snippet](https://huggingface.co/sunshk/openpi_comet_pytorch/blob/main/README.md)).
- **Configs** **[code]**:
  - `pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k` uses `Pi0Config(pi05=True, action_horizon=32)`, starts from `gs://openpi-assets/checkpoints/pi05_base/params`, and runs 50k steps (`src/openpi/training/config.py:721-747`).
  - It uses `fine_grained_level=0  # 0, 1, 2` (`:732`). The README defines 0 = global instruction, 1 = subtask instruction, 2 = skill instruction (`README.md:147`).
  - The released pt50 was therefore configured for **global task instructions** only.
- **Serving** **[code]**:
  - The wrapper reads `robot_r1::proprio` and `robot_r1::robot_r1:*:Camera:0::rgb` (`src/openpi/shared/eval_b1k_wrapper.py:93-111`).
  - The prompt is `metadata[task_name]["task"]` from `scripts/task_mapping.json` (`:43-46,154`). That file has 50 tasks, each with `task` NL text plus a `skill` list; the union is 34 skill strings.
  - It has an **optional hierarchical mode**: with `fine_grained_level>0` a "reasoner" client generates a subtask string that replaces `batch["prompt"]` (`:58-63,156-163`). Its prompt template asks "What's the next skill to perform?" (`:18-25`).
- **How the docs run it:** `uv run scripts/serve_b1k.py --task_name=$TASK_NAME --control_mode=receeding_horizon --max_len=32 policy:checkpoint --policy.config=pi05_b1k-base --policy.dir=$PATH_TO_CKPT` (`README.md:238-245`). **[docs]**
- **Size and license of the HF weights:** **NOT FOUND**. A snippet notes no license field was visible. **[snippet]**

### D3. Other 2025 items
- **Official 2025 baseline checkpoints (Drive)** **[docs]**:
  - Pi0 `turning_on_radio` and `picking_up_trash`, "trained for 50k steps" (`b1k-baselines/tutorials/openpi.md:33-37`).
  - Trained from `gs://openpi-assets/checkpoints/pi0_base/params` (`:57`).
  - Served via `scripts/serve_b1k.py --task_name=$TASK_NAME policy:checkpoint --policy.config=pi0_b1k` (`:69`).
  - A WB-VIMA `turning_on_radio` checkpoint (`il_lib.md:146`).
  - These are pi0 (not pi0.5), single-task, and use the older `omnigibson/learning/eval.py` evaluator.
- **StarVLA (2025, q=0.0019 public).** The repo is linked from `docs/gen_leaderboard.py:21-24`. Its Behavior README says "1) Download the checkpoint:``" with an empty link (`starVLA@master:examples/simBenchmarks/Behavior/README.md:38`). Checkpoint: **NOT FOUND**.

---

## 7. Category E: per-task / per-skill weights

Official per-task checkpoints are B1 and B2 above (2026, `turning_on_radio` only) and the 2025 Pi0/WB-VIMA single-task ones (§6 D3). **[docs]**

Official skill-conditioned weights: **NOT FOUND**.
- I found no checkpoint in the official docs or forks trained on per-skill language.
- The 2026 dataset *does* ship "Language annotations for each episode" (`docs/challenge/dataset.md:22-23`) and "per-episode language annotations for all 20,000 demonstrations" (`updates.md:51`).
- dataset.md lists **31 unique skills** (`dataset.md:103`; list at `:118+`).
- **[inferred]** One internal inconsistency: dataset.md gives "Total Skills 270,600" and "Avg. Skills per Trajectory 27.06" for 20,000 trajectories (`dataset.md:91-108`). 270,600 / 20,000 = 13.53. These values equal the 2025 table (10,000 × 27.06 = 270,600; `archive/2025/dataset.md:82-84`), so they look copied from 2025.

Community per-task checkpoints. All are **[snippet]** only and unverified:
- `fastwalker1118/b1kcheckpoints`: params-only single-task fine-tunes of `IliaLarchenko/behavior_50t_checkpoint` on 2026 demos (task 5, task 10). The card reports 0.0% vs 5.0% for "official checkpoint_1" on task 10 over 20 instances ([link](https://huggingface.co/fastwalker1118/b1kcheckpoints)).
- `JackLiu0406/b1k-checkpoints`: single-task eval checkpoints from the RLC code path, params and assets only ([link](https://huggingface.co/JackLiu0406/b1k-checkpoints)).
- `Hoshipu/behavior1k-task16-moving-boxes`: pi0.5 PiBehavior fine-tune on task 16 using the 100-task norm assets ([link](https://huggingface.co/Hoshipu/behavior1k-task16-moving-boxes)).
- `0Corvid0/pi05-b1k-eval-checkpoints`: gated; pi0.5 eval checkpoints over 25 tracked 2026 tasks ([link](https://huggingface.co/0Corvid0/pi05-b1k-eval-checkpoints)).
- `Shawn3636/pi05-rft-behavior1k` (4 tasks) and `Shawn3636/pi05-pcd-sft-step10k` (6-task, point cloud) ([link1](https://huggingface.co/Shawn3636/pi05-rft-behavior1k), [link2](https://huggingface.co/Shawn3636/pi05-pcd-sft-step10k)).
- `madokalif/pi05-b1k-base-pytorch-sft`: a PyTorch pi0.5 base used for LoRA starts ([link](https://huggingface.co/madokalif/pi05-b1k-base-pytorch-sft)).

---

## 8. Category F: 100-task 2026 weights

**Official: NOT FOUND.** Places checked:
- `docs/challenge/baselines.md` at all 4 refs;
- `docs/challenge/*.md` and `docs/challenge/archive/**`;
- OPENPI (`behavior` branch);
- the `wensi-ai/Isaac-GR00T` fork (all 5 branches);
- the `gs://openpi-assets` bucket listing;
- WebSearch for "behavior-1k 2026 100 task pi0.5", "huggingface.co behavior-1k organization models", and "2026 BEHAVIOR Challenge leaderboard baseline". The search found no official BEHAVIOR-team model repo; the HF org page surfaced only datasets ([snippet](https://huggingface.co/behavior-1k/datasets)).

Community candidates (all **[snippet]** only):
- **F1. `kmy17518/gr00t-n1.7-b1k-multitask`** ([link](https://huggingface.co/kmy17518/gr00t-n1.7-b1k-multitask)).
  - Snippet: "multi-task fine-tune of nvidia/GR00T-N1.7-3B on the BEHAVIOR-1K 2026 challenge dataset (100 tasks, 20,000 episodes, LeRobot v3.0)".
  - Final snapshot `checkpoint-238000`. The author notes the schedule was "still near peak LR at the final snapshot".
  - **[inferred]** The username matches the author of the `my` branch commits in `wensi-ai/Isaac-GR00T` (`git log origin/my` author `kmy17518`). That branch documents uploading `<exp>/checkpoint-<step>/` eval-only copies of about 7 GB each to a public HF repo (`getting_started/b1k.md:246-286`) and per-request `task_id`→prompt serving. It is likely the matching serving code, but this is not verified.
- **F2. `JackLiu0406/meta-SFT-checkpoints`** ([link](https://huggingface.co/JackLiu0406/meta-SFT-checkpoints)).
  - Snippet: "continued meta-training ... across all 100 activities of the 2026 challenge set", using the stock PiBehavior (RLC) architecture.
  - Recommended checkpoint `step69999`. The repo is **private**.
  - It warns to use "qvel-fixed" norm stats and to update a validation guard in `b1k/transforms.py` or tasks 50+ are dropped.

---

## 9. 2025 numbers (as stated in repo files; none are in baselines.md)

- **baselines.md (2026 or 2025 archive):** no numbers. **NOT FOUND** (§2).
- **2025 leaderboard data** in `docs/challenge_submissions/*.json`, rendered by `docs/gen_leaderboard.py:28-87` **[code]**. Overall `q_score` / `task_sr` at lines 8-9 of each file:

  | File | q_score | task_sr |
  |---|---|---|
  | `standard.hidden.Robot_Learning_Collective.Independent.20251114.json` | 0.25985 | 0.124 |
  | `standard.public.Robot_Learning_Collective...` | 0.26050 | 0.112 |
  | `standard.hidden.Comet.NVIDIA_Research.20251117.json` | 0.25144 | 0.114 |
  | `standard.public.Comet...` | 0.18300 | 0.144 |
  | `standard.hidden.SimpleAI_Robot...` | 0.15911 | 0.108 |
  | `standard.public.StarVLA.Independent.20251117.json` | 0.00187 | 0.0 |

- **NVIDIA N1.6 BEHAVIOR table:** average Task Progress 26.30% for N1.6 vs 11.30% for Pi0.5 (`NVIDIA/Isaac-GR00T@n1d6:examples/BEHAVIOR/README.md:69`). This is a different metric from Q. **[docs]**
- **RLC README:** "1st Place | Public Leaderboard: 26% | Private Leaderboard: 26%" and "achieving a 26% success rate" (`README.md:3,5`). The leaderboard JSON shows this is a **Q-score**, not task success (task_sr 0.112 / 0.124). **[code vs docs]**
- **Comet README:** "Q-score of **0.2514 (Held-out Test)**" (`README.md:43`), matching the JSON. **[docs]**

---

## 10. Interface checks against the official evaluator (relevant to every candidate)

- **Obs flattening.** The evaluator flattens obs as `<robot.name>::...` (`OmniGibson/omnigibson/eval/utils/eval_utils.py:235-247`). The default robot name is `robot_r1` (`r1pro.yaml:2`). It adds `"<robot>::cam_rel_poses"` and `"task_id"` (`evaluator.py:576-592`). **[code]**
- **Proprio layout.** The 61-D proprio follows `r1pro.yaml:13-28` in order: base_qvel 3, arm_left_qpos 7, arm_left_qvel 7, eef_left_pos 3, eef_left_quat 4, gripper_left_qpos 2, gripper_left_qvel 2, then the right-side equivalents, then trunk_qpos 4 and trunk_qvel 4. This gives offsets 0:3, 3:10, 24:26, 28:35, 49:51 and 53:57, which match the slices used by OpenPI, GR00T and RLC. **[code + arithmetic, inferred ordering]**
- **No prompt from the evaluator.** The evaluator sends no language prompt; only `task_id` is task information. Every executor's prompt is decided server-side. **[code: `evaluator.py:591`, `policies.py:88-92`]**
- **Chunk replay.** This is opt-in (`--replay-action-chunk-size`). It requires the server to return `action_chunk`, and none of the baseline servers do (§4). **[docs/code]**
- **Images.** The default eval wrapper delivers 224x224 RGB (`OmniGibson/omnigibson/eval/wrappers/default_wrapper.py:4,23-28`). The OpenPI and GR00T wrappers also resize and pad to 224. **[code]**

---

## 11. Discrepancies (page vs code)

1. **Observation keys.** The OpenPI fork R1Pro config uses robot name `robot` / `robot::robot:...` keys (`OPENPI/src/openpi/configs/robots/b1k.py:7,12,18,24`; `eval_b1k_wrapper.py:72`). The official evaluator default is `robot_r1` (`r1pro.yaml:2`; `evaluation.md:168`). baselines.md presents the π0.5 server plus default evaluator as working end-to-end (`baselines.md:166-185,297-312`).
2. **GR00T baseline language.** The GR00T baseline (`behavior` branch) is served with the constant prompt "pick up the object and place it on the table" (`eval_b1k_wrapper.py:70`; `serve_b1k.py:78-83`) but trained on `task_index`-derived text (`r1pro.json:65-68`). baselines.md does not mention this.
3. **π0.5 baseline language.** Training uses the `tasks.parquet` string, while serving uses the `TASK_REGISTRY` NL description. Per the GR00T fork docs `tasks.parquet` holds the snake_case id (`b1k_prompts.py` docstring; `getting_started/b1k.md:294`), which would be a train/serve mismatch. This is unverified because the parquet could not be read.
4. **Only one task registered in OpenPI.** `TASK_REGISTRY` has only `turning_on_radio` (`configs/tasks/b1k.py:4-6`), yet baselines.md says `TASK_NAME` can be "any challenge task" (`baselines.md:33`).
5. **Norm-stats asset id.** baselines.md has `--repo-id $REPO_ID=behavior-1k/2026-challenge-demos` (`baselines.md:36,177`), but the fork config default is `repo_id="turning_on_radio"` (`config.py:760`) and `docs/b1k.md:69` gives `turning_on_radio` as the example. The provided checkpoint loads norm stats from `<ckpt>/assets/<repo_id>` (`policy_config.py:64`), so serving with `--repo-id behavior-1k/2026-challenge-demos` only works if the Drive checkpoint was trained with that asset id. Unknown.
6. **Serve CLI.** OPENPI `docs/b1k.md:246-251` uses `policy:checkpoint`, which does not match the non-union `Args.policy` in `serve_b1k.py:31`. The baselines.md form (`--policy.config`) matches the code.
7. **Control-mode spelling.** The OpenPI fork wrapper uses `receding_horizon`, while the GR00T fork wrapper uses `receeding_horizon` / `receeding_temporal` (`gr00t/eval/eval_b1k_wrapper.py:361-364` on branch `behavior`). The GR00T default `temporal_ensemble` is unaffected.
8. **"26%" headline.** The RLC README says "26% success rate", but the leaderboard JSON gives q_score 0.26 and task_sr 0.112-0.124.
9. **Copied skill stats.** dataset.md skill statistics for 2026 appear copied from 2025 (270,600 total / 27.06 average does not fit 20,000 trajectories).
10. **Missing GR00T link in fork docs.** The GR00T fork `my`-branch guide still says "(add checkpoint link)" for the `turning_on_radio` checkpoint (`getting_started/b1k.md:93`), while baselines.md gives a Drive link.

---

## 12. Open questions

- Contents of the two Drive checkpoints: size, training step, training date (before or after the 07/27 base_qvel fix and 08/24 qvel fix), norm-stats asset id, and whether they were trained single-task or on all tasks.
- Exact `meta/tasks.parquet` task strings in `behavior-1k/2026-challenge-demos`: snake_case id or NL description. HF is blocked.
- Whether the OpenPI fork baseline works with the default evaluator (`robot::` vs `robot_r1::`). This needs a dry run with a recorded observation, or `--robot-config` with `name: robot`, which `evaluation.md:168` allows.
- License and size of community checkpoints (kmy17518, comet, RLC, N1.6-BEHAVIOR1k). Model cards could not be fetched.
- Whether `kmy17518/gr00t-n1.7-b1k-multitask` is public, its prompt source (`task_name` vs `task_description`), and its per-task performance.
- Whether Comet's hierarchical (`fine_grained_level` 1/2) checkpoints were ever released; the released pt50 config uses level 0.
- Whether the official organizers plan a 100-task baseline. Nothing in `updates.md` at HEAD or main mentions one.
- The scaling between base action values and m/s: controller input limits [-1,1] map to output [-0.75,0.75], and `update_lerobot_base_qvel.py:13-17` compares `action[t]` to `base_qvel[t+1]`.

---

## 13. Implications for a strong-planner + off-the-shelf-executor agent

1. **No official executor covers 100 tasks.**
   - The only official fine-tuned weights are single-task `turning_on_radio`, for both π0.5 and GR00T.
   - An executor covering all tasks must be trained, or taken from community 100-task weights (kmy17518 GR00T N1.7 multitask is public per snippet; JackLiu0406 meta100 is private).
2. **No released executor is trained on subgoal or skill language.**
   - Official π0.5 and GR00T use a task-level prompt (or effectively none).
   - RLC uses task-ID embeddings only.
   - Comet pt50 uses the global instruction.
   - N1.6-BEHAVIOR1k uses a `coarse_action` key that is fed the task instruction at eval.
   - A planner that emits free-form subgoal text will therefore be off-distribution for every off-the-shelf checkpoint.
   - The planner can safely control: which **task prompt / task_id** to present, when to **reset or re-plan** the executor, and possibly **checkpoint switching** per task (as RLC does).
   - Skill-conditioned control needs fine-tuning on the 2026 per-episode annotations (31 skills).
3. **Prompt injection needs server-side changes.**
   - The OpenPI fork wrapper fixes the prompt at server start.
   - The GR00T `my` branch resolves the prompt per request from `task_id`, or via `--text-prompt`.
   - A modular agent's policy server must own prompt selection and overwrite the prompt per query. The evaluator never sends text.
4. **Use the official key namespace.**
   - Obs keys are `robot_r1::...`; `task_id` is available every step.
   - The executor adapter should read `robot_r1::proprio` (61-D) and the three `robot_r1::robot_r1:*:Camera:0::rgb` streams.
   - The OpenPI fork's `robot::` keys must be patched (or the robot renamed via `--robot-config`).
5. **Respect chunking and timing.**
   - π0.5: 32-step chunk, re-plan every 16 steps.
   - GR00T: 16-step chunk with temporal ensembling (inference every step), which is costlier against the 1 FPS average budget.
   - RLC: 30-step chunk, executes 26 in 20 steps.
   - If the server returns `action_chunk`, chunk replay can cut websocket round trips. Temporal ensembling must not be combined with chunk replay (`evaluation.md:126`).
6. **Norm stats are per checkpoint.**
   - OpenPI loads them from `<ckpt>/assets/<repo_id>` and GR00T from the checkpoint's `statistics.json`.
   - A multi-checkpoint executor must keep each checkpoint's own stats and asset id.
   - Recompute stats only after the 07/27 and 08/24 dataset fixes.
7. **2025 weights cover only tasks 0-49.**
   - RLC's 50 task embeddings cannot index tasks 50-99.
   - The state convention changed (base velocity frame, qvel fix), so 2025 checkpoints need validation or a short fine-tune on 2026 data before they are trusted.
8. **Budget for size and hardware.**
   - pi05_base is 11.59 GiB on GCS.
   - GR00T N1.7 eval checkpoints are about 7 GB each (34-35.7 GB with optimizer state).
   - Comet states inference needs more than 8 GB GPU memory (`README.md:80`).

---

## 14. Search result URLs used (WebSearch snippets only)
- https://huggingface.co/kmy17518/gr00t-n1.7-b1k-multitask
- https://huggingface.co/JackLiu0406/meta-SFT-checkpoints
- https://huggingface.co/JackLiu0406/b1k-checkpoints
- https://huggingface.co/fastwalker1118/b1kcheckpoints
- https://huggingface.co/Hoshipu/behavior1k-task16-moving-boxes
- https://huggingface.co/0Corvid0/pi05-b1k-eval-checkpoints
- https://huggingface.co/Shawn3636/pi05-rft-behavior1k
- https://huggingface.co/Shawn3636/pi05-pcd-sft-step10k
- https://huggingface.co/madokalif/pi05-b1k-base-pytorch-sft
- https://huggingface.co/sunshk/openpi_comet_pytorch/blob/main/README.md
- https://huggingface.co/sunshk/openpi_comet/blob/main/README.md
- https://huggingface.co/lerobot/pi05_base/commit/975848a74c17ca1f646570aec93c761941f20be2
- https://mintlify.wiki/NVIDIA/Isaac-GR00T/examples/behavior
- https://mixpeek.com/model/nvidia/GR00T-N1.6-BEHAVIOR1k
- https://huggingface.co/behavior-1k/datasets
- https://huggingface.co/SamLegion888/GR00T-N1.7-3B
- https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html
