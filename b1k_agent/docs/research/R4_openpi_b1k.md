# R4 — OpenPI fork (wensi-ai/openpi @ `behavior`) — BEHAVIOR-1K integration

Scope: everything B1K-related in `OPENPI=/home/user/wensi-ai/openpi` (branch `behavior`, commit `0cc8e355`, 2026-06-28, the fork linked from the official baselines page), cross-checked against the official evaluator in `B1K=/home/user/stanfordvl/behavior-1k` (HEAD = `v3.9.3-post2`, `a8247a8`) plus `origin/2026-eval`.
I could not run anything (no GPU or simulator). Every claim below is tagged:
- **[code]**: verified by reading code
- **[docs]**: stated in a docs/markdown file
- **[inferred]**: my reasoning from code; not executed
- **[snippet]**: web-search snippet only
- **NOT FOUND**: I looked and did not find it

Path shorthands used below: `OP:` = OPENPI, `B1K:` = B1K at HEAD, `B1K@2026-eval:` = `git show origin/2026-eval:<path>`, `PRn:` = open PR head fetched into a scratch clone (`/tmp/claude-0/-home-user-VAGEN-Lite/9977f130-b514-5e2e-ab6e-7d7795066892/scratchpad/openpi_refs`, `git fetch refs/pull/N/head`).

---

## 0. TL;DR (most decision-relevant)

1. **One B1K train config only: `pi05_b1k`** [code] `OP:src/openpi/training/config.py:756-774`. It is π0.5 (`Pi0Config(action_horizon=32, pi05=True)`), uses the R1Pro robot (`robot_config_name="b1k/R1Pro"`), and starts from `gs://openpi-assets/checkpoints/pi05_base/params`. The default `repo_id` is `"turning_on_radio"` and `dataset_root` is a hard-coded Stanford cluster path. No per-task or multi-task variants exist; whether it is single-task or multi-task depends only on which LeRobot dataset you point it at.
2. **Language-conditioned, but with one fixed prompt per server process.**
   - Training prompt [code]: `PromptFromLeRobotTask` reads the LeRobot `meta/tasks` table (task_index → string) (`OP:src/openpi/training/data_loader.py:150-153`, `OP:src/openpi/transforms.py:355-370`).
   - Serving prompt [code]: `TASK_REGISTRY["b1k"][task_name]` (`OP:scripts/b1k/serve_b1k.py:46-47`). `B1KPolicyWrapper` stamps it on every request and never reads a prompt from the observation (`OP:src/openpi/shared/eval_b1k_wrapper.py:88-97`).
   - **The registry has exactly ONE entry** [code]: `"turning_on_radio": "Turn on the radio receiver that's on the table in the living room."` (`OP:src/openpi/configs/tasks/b1k.py:4-6`). It matches `B1K:docs/challenge/task_data.json:6`.
3. **Model I/O**
   - Images [code]: 3 cameras in the order head → left wrist → right wrist, mapped to `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`, resized with padding to 224×224.
   - State [code]: 23-D, taken from the 61-D B1K proprio at these indices:
     - base_qvel 0:3
     - trunk_qpos 53:57
     - left arm 3:10
     - left gripper sum(24:26)
     - right arm 28:35
     - right gripper sum(49:51)
   - Actions [code]: 23-D in chunks of 32. Torso and arms are trained as **deltas relative to current state** and converted back to absolute on output. Base velocity and grippers are absolute.
   - Normalization [code]: z-score, with `use_quantile_norm=False` forced (`config.py:455`).
4. **Serving** [code]
   - `scripts/b1k/serve_b1k.py` runs a custom websocket server (`openpi/serving/websocket_b1k_server.py`). It calls `policy.act(obs)` and returns `{"action": (B,23)|(23,), "server_timing": {...}}`.
   - A message with a `"reset"` key resets the wrapper and gets no reply.
   - Default control is `receding_horizon` with `action_horizon=16`: run inference, execute 16 of the 32 predicted actions, then replan.
   - Other modes are `receding_temporal` and `temporal_ensemble` (ensemble of up to 5 chunks, weights `exp(0.005*i)`, grippers taken from the newest chunk).
5. **Observation-key mismatch with the official evaluator** [inferred from code, corroborated by an open PR]
   - The fork's R1Pro config uses robot `name="robot"`, so it reads `robot::proprio` and `robot::robot:zed_link:Camera:0::rgb` (`OP:src/openpi/configs/robots/b1k.py:7-24`, `eval_b1k_wrapper.py:72,79`).
   - The official evaluator's default robot is `name: robot_r1` (`B1K:OmniGibson/omnigibson/eval/r1pro.yaml:2`), so it sends `robot_r1::proprio` and `robot_r1::robot_r1:zed_link:Camera:0::rgb`. This holds on HEAD, main, 2026-eval and group-time-budget.
   - Unmerged PR #5 on the fork (`andi/vector`, 2026-07-24) changes the names to `robot_r1` to fix exactly this.
   - Either patch the OpenPI robot config or evaluate with a custom robot config named `robot`.
6. **Norm-stats asset id pitfall** [code]
   - When serving, `asset_id = --repo_id or --task` (`serve_b1k.py:53-57`), and norm stats load from `<ckpt>/assets/<asset_id>/norm_stats.json` (`policy_config.py:59-64`).
   - The fork's own doc example omits `--repo_id`, so it would look up `assets/b1k/<TASK>/`. Training writes `assets/<train repo_id>/` instead (`checkpoints.py:71-77`).
   - The official B1K doc passes `--repo-id $REPO_ID` with `REPO_ID=behavior-1k/2026-challenge-demos` (`B1K:docs/challenge/baselines.md:36,177`).

---

## 1. Inventory of B1K-related files in the fork

From `grep -ril "b1k\|behavior\|r1pro\|omnigibson"` [code]:

| File | Role |
|---|---|
| `docs/b1k.md` (310 lines) | Fork tutorial: train, norm stats, serve |
| `src/openpi/training/config.py` | `DataConfig` B1K fields, `LeRobotB1KDataConfig`, `pi05_b1k`, B1K validation params |
| `src/openpi/policies/b1k_policy.py` | `B1KInputs` / `B1KOutputs`, `extract_state_from_proprio` |
| `src/openpi/configs/robots/{__init__,base_config,b1k}.py` | Robot registry, `RobotConfig`, R1Pro definition |
| `src/openpi/configs/tasks/{__init__,b1k}.py` | `TASK_REGISTRY` (prompt table) |
| `src/openpi/shared/eval_b1k_wrapper.py` | `B1KPolicyWrapper` (receding horizon / temporal ensembling) |
| `src/openpi/serving/websocket_b1k_server.py` | B1K websocket server, with its own msgpack+numpy+torch codec |
| `src/openpi/training/data_loader.py` | `create_b1k_dataset`, `create_b1k_data_loader` |
| `src/openpi/training/lerobot_compat.py` | LeRobot API shim and `tasks_from_metadata` |
| `scripts/b1k/{serve_b1k.py, train_b1k.py, train_b1k.sh, train_b1k.sbatch.sh}` | Entry points |
| `scripts/compute_norm_stats.py` | Adds `create_b1k_dataloader` (used when `dataset_root` is set) |
| `examples/simple_client/test_b1k_client.py` | **Stale.** Imports `OpenPIWrapper`, which no longer exists, and uses R1 keys `robot_r1::robot_r1:eyes:Camera:0::rgb` (lines 8, 13, 28-30). |
| `examples/simple_client/main.py` | `EnvMode.B1K = 'b1k'` (line 24), but there is no B1K entry in `obs_fn` (lines 119-124), so it is unusable for B1K. |
| `pyproject.toml:70` | `lerobot = { git = "https://github.com/wensi-ai/lerobot", branch = "release/b1k" }`. `uv.lock:1755-1757` pins lerobot 0.5.2 @ `c43f5811`. |

Notes:
- `README.md` has **no** B1K section [code]: grep for b1k/behavior/r1pro gave no hits.
- The `behavior` branch is a 1-commit shallow history: `0cc8e35 2026-06-28 Merge pull request #4 from 13RENDA/brenda/pi05`.
- Other remote refs (via `git ls-remote`): `2025-challenge` (83ab06a), `andi/vector`, `i3l`, `main`, `molmo`, `s2rg`, and PR heads #1–#7. Merge refs exist for #5 and #6, so they are probably open [inferred].
- There is no `.gitmodules`, so the official doc's `git submodule update --init --recursive` (`B1K:docs/challenge/baselines.md:87`) does nothing [inferred].

---

## 2. Training configuration

### 2.1 `pi05_b1k` (the only B1K config) [code] `OP:src/openpi/training/config.py:755-774`

```python
    #  ================= behavior configs =================
    TrainConfig(
        name="pi05_b1k",
        model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
        data=LeRobotB1KDataConfig(
            repo_id="turning_on_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/viscam/u/shiyuc/openpi/2026-challenge-demos/b1k/turning_on_radio",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=10_000,
        num_train_steps=50_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
```

Full list of config names [code] (`config.py` `grep name="`): `pi0_aloha`, `pi05_aloha`, `pi0_aloha_towel`, `pi0_aloha_tupperware`, `pi0_droid`, `pi0_fast_droid`, `pi05_droid`, **`pi05_b1k`**, `pi0_libero`, `pi0_libero_low_mem_finetune`, `pi0_fast_libero`, `pi0_fast_libero_low_mem_finetune`, `pi05_libero`, `pi0_aloha_pen_uncap`, `pi05_aloha_pen_uncap`, `pi0_fast_full_droid_finetune`, `pi05_full_droid_finetune`, `pi05_droid_finetune`, `pi0_aloha_sim`, `debug`, `debug_restore`, `debug_pi05`, plus roboarena/polaris configs (`config.py:1100-1102`).
- **NOT FOUND:** any `pi05_b1k_<task>` or multi-task B1K config.

Model defaults that apply [code] (`OP:src/openpi/models/pi0_config.py:18-41`):
- `action_dim=32` (state and actions are zero-padded to 32)
- `max_token_len = 200 if pi05`
- `discrete_state_input = pi05` (True), so the state is put into the text prompt as 256-bin tokens (see §5)
- `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`
- Sampling uses `num_steps=10` flow-matching steps by default (`OP:src/openpi/models/pi0.py:217-228`)
- For π0.5 there is no continuous state token; it is added only `if not self.pi05` (`pi0.py:151-155`)

TrainConfig defaults [code] (`config.py:566-647`):
- `project_name="B1K"`, `batch_size=32` (scripts override to 64), `ema_decay=0.99`, `keep_period=5000`
- B1K-specific: `val_log_interval`, `val_batch_size`, `val_num_batches=10`, `val_repo_id`, `val_episodes_index`
- Checkpoints save **EMA params** as `params` when EMA is on (`OP:src/openpi/training/checkpoints.py:149-156`).

`DataConfig` B1K fields [code] (`config.py:103-110`): `data_cls` (default `_lerobot_compat.LeRobotDataset`), `dataset_root`, `dataset_kwargs`.
- `DataConfig.episodes_index` is read at `data_loader.py:174` and `train_b1k.py:59` but **is not declared** on `behavior`. Open PR #6 adds it (`PR6: config.py +2 lines`).

### 2.2 `LeRobotB1KDataConfig` [code] `config.py:375-456`

```python
@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    robot_config_name: str = tyro.MISSING
    extra_delta_transform: bool = True
    action_sequence_keys: Sequence[str] = ("action",)
    ...
    def create(self, assets_dirs, model_config) -> DataConfig:
        robot_config = ROBOT_REGISTRY[self.robot_config_name]
        repack_mapping = {}
        for i in range(3):  # Support up to 3 cameras (image_0, image_1, image_2)
            image_key = f"image_{i}"
            if image_key in robot_config.observations:
                repack_mapping[f"observation/{image_key}"] = robot_config.observations[image_key].dataset_key
        repack_mapping.update({
                "observation/state": "observation.state",
                "actions": robot_config.action_key,
                "prompt": "prompt",
        })
        ...
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1KInputs(robot_config=robot_config, model_type=model_config.model_type)],
            outputs=[b1k_policy.B1KOutputs(action_dim=robot_config.action_dim)],
        )
        if self.extra_delta_transform:
            delta_mappings = self._build_delta_mappings(robot_config)
            data_transforms = data_transforms.push(
                inputs=[_transforms.MappedDeltaActions(delta_mappings)],
                outputs=[_transforms.MappedAbsoluteActions(delta_mappings)],
            )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(robot_config.action_key,),
            use_quantile_norm=False,
        )
```

- **Normalization is z-score (mean/std)** for B1K [code] (`config.py:455`). This overrides the generic rule `use_quantile_norm = model_type != PI0` (`config.py:199`), which would have chosen quantile norm for π0.5.
- `_build_delta_mappings` (`config.py:381-404`) pairs each action group with `needs_delta_comp=True` with the next state slice of the same width. For R1Pro this gives [inferred, by hand-evaluating the code on the R1Pro config]:
  - torso: action 3..6 ↔ state 3..6
  - left_arm: action 7..13 ↔ state 7..13
  - right_arm: action 15..21 ↔ state 15..21
  - base (0..2) and grippers (14, 22) stay absolute.
- `MappedDeltaActions` subtracts the current state from **every** step of the chunk (`actions[..., idx] -= state[..., sidx][..., None, :]`). `MappedAbsoluteActions` adds it back (`OP:src/openpi/transforms.py:247-290`) [code].

### 2.3 Dataset loading [code] `OP:src/openpi/training/data_loader.py:130-155`

```python
def create_b1k_dataset(data_config, action_horizon):
    if isinstance(data_config.repo_id, list):
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id[0], root=os.path.join(data_config.dataset_root, data_config.repo_id[0]))
        dataset_kwargs = {"repo_ids": data_config.repo_id, **data_config.dataset_kwargs}
    else:
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(repo_id=data_config.repo_id, root=data_config.dataset_root)
        dataset_kwargs = {"repo_id": data_config.repo_id, **data_config.dataset_kwargs}
    dataset = data_config.data_cls(
        root=data_config.dataset_root,
        delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys},
        **dataset_kwargs)
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))])
    return dataset
```

- Action chunk = 32 consecutive frames at the dataset fps [code].
- The B1K evaluator steps at `action_frequency: 30` (`B1K:OmniGibson/omnigibson/eval/utils/eval_utils.py:204`) [code]. So one chunk ≈ 1.07 s if the dataset is 30 fps [inferred]. **NOT FOUND:** the dataset fps value is not in docs.
- `tasks_from_metadata` handles LeRobot v3's DataFrame `tasks` table (`OP:src/openpi/training/lerobot_compat.py:17-25`):

```python
    if isinstance(tasks, pd.DataFrame):
        return {int(row["task_index"]): str(task_name) for task_name, row in tasks.iterrows()}
```

- Multi-repo (`repo_id` as list) is supported via `MultiLeRobotDataset`-style kwargs. In that case norm stats use only the first asset_id (`config.py:207-209`, `checkpoints.py:76`) [code].
- **NOT FOUND:** the dataset's per-episode language `annotations` folder (`B1K:docs/challenge/dataset.md:22-23`) is not used anywhere in the fork. Grep for annotation/subtask/skill in `src` and `scripts` found nothing relevant.

### 2.4 Training scripts [code]

- `scripts/b1k/train_b1k.py`: `main(_config.cli())` (last line), using `tyro.extras.overridable_config_cli` (`config.py:1110-1111`).
  - Uses `create_b1k_data_loader`, logs camera views to W&B (`train_b1k.py:431-436`), and adds validation loss.
- `scripts/b1k/train_b1k.sh` defaults: `CONFIG_NAME=pi05_b1k`, `NUM_GPUS=8`, `--batch_size=64`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, exp name `openpi_<date>`.
  - It **hard-codes** `source /home/ubuntu/jiajun-stanford-lab/Research/openpi/.venv/bin/activate` (`train_b1k.sh:91`).
- `scripts/b1k/train_b1k.sbatch.sh`: 4×H200, `--resume --batch_size=64 --num_train_steps=100000`.
  - It **omits the line-continuation backslash** after `--num_train_steps=100000` (sbatch file lines 47-48), so `"$@"` runs as a separate command and pass-through args are dropped [inferred].

### 2.5 Norm stats computation [code] `OP:scripts/compute_norm_stats.py`

- When `dataset_root` is set, uses `create_b1k_dataloader` (lines 141-144).
- Stats are computed on `["state","actions"]` **after** repack plus data transforms (lines 110-118, 150-157). So stats are on the 23-D extracted state and on **delta** actions for torso and arms [inferred from transform order].
- Stores mean, std, q01, q99 (`OP:src/openpi/shared/normalize.py:8-14,73-86`).
- Output path is `config.assets_dirs / asset_id`, i.e. `./outputs/assets/pi05_b1k/<repo_id>/norm_stats.json` (lines 159-162; `config.py:649-652`).
- CLI signature: `def main(config_name: str, max_frames: int | None = None)` with `tyro.cli(main)` (lines 134, 166). The script **does not accept** `--data.*` overrides [code]. See §9 for the doc discrepancy.

---

## 3. Robot config (R1Pro) and task/prompt registry

### 3.1 R1Pro [code] `OP:src/openpi/configs/robots/b1k.py:1-50`

```python
R1Pro = RobotConfig(
    name="robot",
    robot_type="R1Pro",
    observations={
        "image_0": ObservationConfig(name="head",
            obs_key="robot::robot:zed_link:Camera:0::rgb",
            dataset_key="observation.rgb.zed_link_camera_0", resolution=[240, 240]),
        "image_1": ObservationConfig(name="left_wrist",
            obs_key="robot::robot:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.left_realsense_link_camera_0", resolution=[240, 240]),
        "image_2": ObservationConfig(name="right_wrist",
            obs_key="robot::robot:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.right_realsense_link_camera_0", resolution=[240, 240]),
    },
    action_key="action",
    action_dim=23,
    action=[
        StateActionConfig(name="base", indices=list(range(3))),
        StateActionConfig(name="torso", indices=list(range(3, 7)), needs_delta_comp=True),
        StateActionConfig(name="left_arm", indices=list(range(7, 14)), needs_delta_comp=True),
        StateActionConfig(name="left_gripper", indices=[14], is_eef=True),
        StateActionConfig(name="right_arm", indices=list(range(15, 22)), needs_delta_comp=True),
        StateActionConfig(name="right_gripper", indices=[22], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="base_qvel", indices=list(range(0, 3))),
        StateActionConfig(name="trunk_qpos", indices=list(range(53, 57))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(3, 10))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(24, 26)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(28, 35))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(49, 51)), is_eef=True),
    ],
)
register_robot("b1k/R1Pro", R1Pro)
```

- `resolution=[240,240]` is **unused**: no reference to `.resolution` in `src` or `scripts` [code, grep]. `ObservationConfig.resolution` is commented `# [width, height]` (`base_config.py:31`).
- Proprio indices match B1K's official `PROPRIOCEPTION_INDICES["R1Pro"]` [code] (`B1K:OmniGibson/omnigibson/eval/utils/eval_utils.py:121-141`):
  - base_qvel 0:3
  - arm_left_qpos 3:10
  - gripper_left_qpos 24:26
  - arm_right_qpos 28:35
  - gripper_right_qpos 49:51
  - trunk_qpos 53:57
  - total proprio length 61
- Action layout matches B1K `ACTION_QPOS_INDICES["R1Pro"]` [code] (`eval_utils.py:106-117`): base 0:3, torso 3:7, left_arm 7:14, left_gripper 14:15, right_arm 15:22, right_gripper 22:23.
- The B1K eval robot controllers are [code] (`B1K:OmniGibson/omnigibson/eval/r1pro.yaml`):
  - arms and trunk: `JointController`, `motor_type: position`, `use_delta_commands: false` (lines 36-51, 75-80)
  - base: `HolonomicBaseJointController`, velocity (lines 63-64)
  - So the 23-D output is absolute joint targets plus base velocity, which matches the fork's "delta→absolute" output conversion [inferred].
- `RobotConfig.__post_init__` validates `sum(len(a.indices)) == action_dim` (`base_config.py:77-82`) [code].
- Lineage [code]: the docstrings mention `i3l` (`robots/__init__.py` example `ROBOT_REGISTRY["i3l/A1"]`; `tasks/b1k.py:3` comment `# i3l task prompts`).

### 3.2 Task → prompt table [code] `OP:src/openpi/configs/tasks/b1k.py:1-9` (complete file)

```python
from . import TASK_REGISTRY

# i3l task prompts
TASKS = {
    "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
}

# Register in global registry
TASK_REGISTRY["b1k"] = TASKS
```

- This is the **only** prompt table. It covers 1 of the 100 challenge tasks [code].
- The string is identical to the `instruction` for `turning_on_radio` in `B1K:docs/challenge/task_data.json:6` [code].
- In that JSON, 100 tasks have unique ids but only **50** have an `instruction` field. The other 50 do not (e.g. `freeze_fruit`, `cook_a_brisket`, `sorting_bottles_cans_and_paper`, …) [code, parsed with python].
- So a 100-task prompt table cannot come entirely from `task_data.json` [inferred]. The other candidate source is the dataset's `meta/tasks.parquet`, whose contents are **NOT FOUND** (HF blocked; no B1K code writes the task strings).
- Registry auto-import [code]: `OP:src/openpi/configs/tasks/__init__.py` imports every module in the folder, so adding `tasks/foo.py` registers a new bucket.

---

## 4. Policy transforms (input mapping, cameras, state, outputs)

### 4.1 `B1KInputs` / `B1KOutputs` [code] `OP:src/openpi/policies/b1k_policy.py:22-112`

```python
def extract_state_from_proprio(proprio_data, robot_config):
    state = []
    for proprio in robot_config.proprio:
        if proprio.is_eef:
            # Sum the gripper finger positions to get a single width value
            state.append(proprio_data[..., proprio.indices].sum(axis=-1, keepdims=True))
        else:
            state.append(proprio_data[..., proprio.indices])
    return np.concatenate(state, axis=-1)

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

class B1KInputs(transforms.DataTransformFn):
    def __call__(self, data):
        proprio_data = data["observation/state"]
        state = extract_state_from_proprio(proprio_data, self.robot_config)
        ...
        for camera_id in self.robot_config.observations:
            images.append(_parse_image(data[f"observation/{camera_id}"]))
            image_masks.append(np.True_)
        while len(images) < 3:
            images.append(np.zeros_like(images[0])); image_masks.append(np.False_)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
        inputs = {"state": state, "image": dict(zip(names, images)), "image_mask": dict(zip(names, image_masks))}
        if "actions" in data: inputs["actions"] = action
        if "prompt" in data: inputs["prompt"] = data["prompt"]
        return inputs

class B1KOutputs(transforms.DataTransformFn):
    action_dim: int
    def __call__(self, data):
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
```

- The **23-D state** order is [inferred from the proprio list order]:
  - [base_qvel(3)]
  - [trunk_qpos(4)]
  - [left_arm_qpos(7)]
  - [left_gripper_width(1)]
  - [right_arm_qpos(7)]
  - [right_gripper_width(1)]
  - This matches the action layout index by index, which is why the delta mapping in §2.2 lines up.
- Camera order is head → left wrist → right wrist, mapped to `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` [code].

### 4.2 Full inference pipeline order [code] `OP:src/openpi/policies/policy_config.py:75-89`

```python
transforms=[*repack_transforms.inputs, transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,          # B1KInputs, MappedDeltaActions(no-op w/o actions)
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs],        # InjectDefaultPrompt(None), ResizeImages(224,224), TokenizePrompt(discrete_state_input=True), PadStatesAndActions(32)
output_transforms=[*data_config.model_transforms.outputs, # none for PI05
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,         # MappedAbsoluteActions, then B1KOutputs (Group.push prepends outputs: transforms.py:49-59)
            *repack_transforms.outputs],
```

- `Policy.infer` returns `{"state": ..., "actions": (32, 23) after B1KOutputs, "policy_timing": {"infer_ms": ...}}` (`OP:src/openpi/policies/policy.py:67-106`) [code].
- `ModelTransformFactory` for PI05 is at `config.py:138-150` [code].
- The JAX RNG defaults to `jax.random.key(0)` (`policy.py:65`) [code].
- A PyTorch checkpoint is detected by the presence of `model.safetensors` (`policy_config.py:48-57`) [code].

---

## 5. Prompt formation and language conditioning

- **The policy is language-conditioned** [code]. For π0.5, `PaligemmaTokenizer.tokenize` builds (`OP:src/openpi/models/tokenizer.py:22-29`):

```python
cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
if state is not None:
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

- The state used here is the **normalized** 23-D state, before padding: `TokenizePrompt` runs before `PadStatesAndActions` (`config.py:142-148`) [code].
- With z-score normalization, values outside [-1, 1) saturate to bin −1 or 255 [inferred from `np.digitize` semantics].
- The prompt is truncated to `max_token_len=200` with a warning (`tokenizer.py:40-46`) [code]. There is no lowercasing for π0.5; only `_` → space.
- `TokenizePrompt` raises `ValueError("Prompt is required")` if there is no prompt (`transforms.py:298-300`) [code].
- **Training prompt** = LeRobot task string for the frame's `task_index` (`transforms.py:355-370`) [code].
- **Serving prompt** = `TASK_REGISTRY[bucket][name]` [code]:
  - It is passed both as `default_prompt` (`serve_b1k.py:59-61`) and as `B1KPolicyWrapper.text_prompt`.
  - `process_input` writes `"prompt": self.text_prompt` into every per-env dict (`eval_b1k_wrapper.py:88-97`).
  - The incoming observation is **never** read for a prompt. The evaluator's `task_id` (`B1K:OmniGibson/omnigibson/eval/evaluator.py:591`) is ignored [code].
- Whether the *provided* checkpoint actually uses language beyond a constant string is unknown [inferred]. It was fine-tuned per task (`turning_on_radio`) according to `B1K:docs/challenge/baselines.md:16-18`, so its training prompt was presumably constant.

---

## 6. Serving: `serve_b1k.py`, `B1KPolicyWrapper`, websocket server

### 6.1 `scripts/b1k/serve_b1k.py` (complete logic) [code] lines 14-87

```python
@dataclasses.dataclass
class Args:
    robot: str
    task: str
    policy: Checkpoint            # Checkpoint(config: str, dir: str)
    repo_id: str | None = None    # LeRobot repo id whose checkpoint assets contain the norm stats.
    control_mode: str = "receding_horizon"
    action_horizon: int = 16      # Number of actions to execute before replanning.
    port: int = 8000
    record: bool = False

def main(args):
    task_bucket, task_name = args.task.split("/")
    task_prompt = TASK_REGISTRY[task_bucket][task_name]
    config = _config.get_config(args.policy.config)
    norm_stats_repo_id = args.repo_id or args.task
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=norm_stats_repo_id, robot_config_name=args.robot))
    policy = _policy_config.create_trained_policy(config, args.policy.dir, default_prompt=task_prompt)
    policy_metadata = policy.metadata
    if args.record: policy = _policy.PolicyRecorder(policy, "policy_records")
    policy = B1KPolicyWrapper(policy=policy, robot=args.robot, text_prompt=task_prompt,
        control_mode=args.control_mode, action_horizon=args.action_horizon, max_len=config.model.action_horizon)
    server = websocket_b1k_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=policy_metadata)
    server.serve_forever()
```

- `max_len` = model action horizon = 32 for `pi05_b1k` [code].
- `metadata` = `train_config.policy_metadata`, which is `None` for `pi05_b1k`, so the server's first message is `{}` [code] (`policy.py:54,108-110`; `websocket_b1k_server.py:38,59`).
- Serving does not touch the dataset [inferred]: `LeRobotB1KDataConfig.create` only builds transforms and tries to load norm stats from the local assets dir, logging "not found" and skipping if absent (`config.py:202-217`). The real stats come from the checkpoint (`policy_config.py:59-64`).
- The generic `scripts/serve_policy.py` has **no** B1K env mode (`EnvMode` = ALOHA, ALOHA_SIM, DROID, LIBERO; `serve_policy.py:14-20` in-file numbering) [code]. It also cannot serve the wrapper, because the generic server calls `policy.infer` and `B1KPolicyWrapper` only has `act` (`websocket_policy_server.py:61`) [code].

### 6.2 `B1KPolicyWrapper` [code] `OP:src/openpi/shared/eval_b1k_wrapper.py`

Constructor fields:
- `obs_size=(224, 224)` (line 17)
- `temporal_ensemble_max = 5` (line 27)
- `gripper_indices` = action indices with `is_eef`, i.e. [14, 22] (lines 29-33)

Input processing (lines 68-98):

```python
prop_state = obs[f"{self.robot.name}::proprio"]
if prop_state.ndim == 1: prop_state = prop_state[None, :]
for camera_key in sorted(self.robot.observations.keys()):
    camera_obs = obs[self.robot.observations[camera_key].obs_key][..., :3]   # drop alpha
    if camera_obs.ndim == 3: camera_obs = camera_obs[None, ...]
    observations.append(resize_with_pad(camera_obs, *self.obs_size))        # openpi_client.image_tools (PIL bilinear, letterbox)
...
processed_input = [{ "observation/image_0": img_obs[i, 0], "observation/image_1": img_obs[i, 1],
    "observation/image_2": img_obs[i, 2], "observation/state": prop_state[i], "prompt": self.text_prompt } for i in range(batch_size)]
```

Control modes (`act`, lines 356-365):
- **`receding_horizon`** (default, lines 118-162):
  - Infers when `step_counter % action_horizon == 0` or when the buffered chunk is exhausted (lines 129-131).
  - Stores `min(T, max_len)` actions and returns `action_buffer[b, 0, idx]`, then increments.
  - With defaults it runs one `Policy.infer` per env every 16 steps and uses the first 16 of the 32 actions [inferred].
- **`receding_temporal`** (lines 164-274):
  - Replans every `action_horizon` steps and keeps up to 5 overlapping chunks.
  - Output = weighted mean with `exp(0.005*i)` weights over active chunks (lines 241-245). Gripper dims come from the latest chunk (lines 249-251).
  - On the very first call it infers twice: init (lines 175-188), then `step_counter==0` replans (191-199) [inferred].
- **`temporal_ensemble`** (lines 276-354):
  - Infers **every step**, ensembles up to 5 chunks with the same weights, and takes grippers from the newest chunk's first action (lines 330-331).
- Returns `torch.from_numpy(final_actions)`: shape (B, 23), or (23,) if the input was unbatched (lines 160-162).
- `reset()` (lines 42-49) clears all buffers and calls `self.policy.reset()`. `BasePolicy.reset` is a no-op (`packages/openpi-client/src/openpi_client/base_policy.py:10-12`).
- Batching [code]: per-env `Policy.infer` calls run in a Python loop (lines 136-138), not as one batched forward. PR #5 adds `Policy.infer_batch`.

### 6.3 Websocket protocol [code] `OP:src/openpi/serving/websocket_b1k_server.py`

```python
async def _handler(self, websocket):
    packer = Packer()
    await websocket.send(packer.pack(self._metadata))          # 1st frame: metadata dict ({} for pi05_b1k)
    while True:
        result = unpackb(await websocket.recv(), strict_map_key=False)
        if "reset" in result:
            self._policy.reset()
            continue                                              # NO reply to reset
        obs = deepcopy(result)
        action = self._policy.act(obs)
        action = {"action": action.cpu().numpy()}
        action["server_timing"] = {"infer_ms": infer_time * 1000}   # + "prev_total_ms" after first step
        await websocket.send(packer.pack(action))
```

- Health check: `GET /healthz` → `200 OK` (lines 106-114).
- `compression=None`, `max_size=None` (lines 45-52).
- On an exception it logs the traceback, closes with `INTERNAL_ERROR` and re-raises (lines 92-103). Unlike the generic server, it does **not** send the traceback string first (compare `websocket_policy_server.py:78-83`).
- Codec (lines 136-180) is `msgpack` with:
  - ndarray / torch.Tensor → `{b"__ndarray__": True, b"data": bytes, b"dtype": dtype.str, b"shape": shape}`
  - numpy scalars → `{b"__npgeneric__": True, ...}`
  - object, void and complex dtypes are rejected
  - It matches `packages/openpi-client/src/openpi_client/msgpack_numpy.py:21-57` (numpy only) and B1K's own `network_utils` Packer.
- **No action-chunk support** on `behavior` [code]:
  - The `__action_chunk_size__` key is ignored, and the response has no `action_chunk`.
  - So B1K's `--replay-action-chunk-size K` falls back to one request per step with a warning (`B1K:OmniGibson/omnigibson/eval/utils/network_utils.py:146-162`; `B1K:docs/challenge/evaluation.md:105-126`).
- **Shared state across connections** [code + PR evidence]: one `self._policy` (the wrapper with its buffers) serves all connections. PR #5 adds a per-connection `copy.copy` of the wrapper because "concurrent eval env slots all share the single self._policy and corrupt each other's action plans" (`PR5: websocket_b1k_server.py` diff comment).
  - This matters for `origin/2026-eval`'s `MultiWebsocketPolicy` (one client per env) if several clients hit one server [inferred] (`B1K@2026-eval:OmniGibson/omnigibson/eval/policies.py:126-216`).

### 6.4 openpi-client package [code]

- `packages/openpi-client` 0.1.0 depends on `dm-tree`, `msgpack>=1.0.5`, `numpy>=1.22.4,<2.0.0`, `pillow>=9.0.0`, `websockets>=11.0` (`pyproject.toml`).
- `WebsocketClientPolicy.infer` packs obs, receives, and raises on a `str` reply (`websocket_client_policy.py:46-54`).
- `ActionChunkBroker` slices a chunk one step at a time (`action_chunk_broker.py:10-50`).
- `image_tools.resize_with_pad` (PIL, bilinear, letterbox; it is a no-op when the size already matches) is at `image_tools.py:15-58`.

---

## 7. Compatibility with the official evaluator (B1K side)

| Item | OpenPI `behavior` expects | Official evaluator sends | Status |
|---|---|---|---|
| Proprio key | `robot::proprio` (`eval_b1k_wrapper.py:72`, `name="robot"` at `robots/b1k.py:7`) | `robot_r1::proprio` (default robot `name: robot_r1`, `B1K:.../eval/r1pro.yaml:2`; obs flattened `"{robot}::{key}"` by `flatten_obs_dict`, `eval_utils.py:235-247`; env obs keyed by robot name `envs/env_base.py:583,602`; proprio key `"proprio"` at `robots/robot.py:1526`) | **MISMATCH** [inferred]. Fixed only in unmerged PR #5 (`name="robot_r1"`). |
| Camera keys | `robot::robot:{zed_link,left_realsense_link,right_realsense_link}:Camera:0::rgb` | `robot_r1::robot_r1:...:Camera:0::rgb` (sensor names `"{robot.name}:{link}:Camera:0"`, `robots/robot.py:1431`; `r1pro.yaml:3-7`) | **MISMATCH** [inferred] |
| Image size | Anything (resized to 224 with padding) | 224×224 with `DefaultWrapper` (`eval/wrappers/default_wrapper.py:4,22-28`); 720 head / 480 wrist with `RGBDFullResWrapper` (`docs/challenge/evaluation.md:136-140`) | OK |
| Proprio layout | 61-D R1Pro layout | Same (`eval_utils.py:121-141`) | OK [code] |
| Batch | (B, …) or unbatched | `_batch_obs` stacks (num_envs, …) (`evaluator.py:411-421`); 2026-eval Multi client sends (1, …) (`B1K@2026-eval:.../policies.py:200`) | OK [inferred] |
| Reset | `{"reset": ...}`, no reply | Client sends `{"reset": True}` and doesn't wait (`network_utils.py:172-179`) | OK |
| Response | `{"action", "server_timing"}` | Client requires `"action"` (`network_utils.py:137-145`) | OK |
| Action chunks | not supported | optional (`--replay-action-chunk-size`) | Fallback (warning) |
| Extra obs keys | ignored | `task_id`, `robot_r1::cam_rel_poses` (`evaluator.py:589-591`) | OK (ignored) |

Evaluator time limits (relevant because policy latency counts):
- On `origin/2026-eval`, per-query timeout `POLICY_RESPONSE_TIMEOUT = 600` s, and per-rollout `MultiWebsocketPolicy.set_time_budget` charges policy time plus sim time (`B1K@2026-eval:OmniGibson/omnigibson/eval/utils/network_utils.py` diff; `policies.py:153-228`) [code].

---

## 8. Checkpoint layout and how to read a downloaded checkpoint

Saved by orbax `CheckpointManager` with items `assets`, `train_state`, `params` [code] (`OP:src/openpi/training/checkpoints.py:40-53,65-87`; `max_to_keep=1`, `keep_period`):

```text
outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/
├── wandb_id.txt                       # train_b1k.py:259-268
└── <STEP>/                            # <- pass this as --policy.dir
    ├── params/                        # orbax PyTree; item {"params": <EMA params if ema_decay>}; restored via model.restore_params(ckpt/"params") (policy_config.py:57; models/model.py:286-325)
    ├── train_state/                   # optimizer state etc. (not needed for serving)
    └── assets/
        └── <ASSET_ID>/norm_stats.json # ASSET_ID = data_config.asset_id (= AssetsConfig.asset_id or repo_id; list -> first)
```

- PyTorch alternative: `<STEP>/model.safetensors`, which switches to the torch path (`policy_config.py:48-55`) [code].
- `norm_stats.json` schema [code] (`normalize.py:120-131`):
  `{"norm_stats": {"state": {"mean": [...23], "std": [...23], "q01": [...], "q99": [...]}, "actions": {...}}}`
  - `Unnormalize` pads mean with 0 and std with 1 up to 32 dims (`transforms.py:170-173`).
- **The train config name is NOT stored in the checkpoint** [code, inferred from `save_state` items]. It must come from the CLI (`--policy.config pi05_b1k`).
- To read a downloaded checkpoint's configuration [inferred]:
  - List `<STEP>/assets/` to learn `ASSET_ID`, then pass `--repo-id <ASSET_ID>`.
  - Check `norm_stats.json` lengths: 23-D means the R1Pro 23-D state/action layout.
  - Note that `params/` gives no prompt or task info. The prompt must come from `TASK_REGISTRY`.
- Doc evidence [docs]:
  - The fork doc says norm stats live in `outputs/assets/<CONFIG_NAME>/<REPO_ID>/` and checkpoints in `outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>/` (`OP:docs/b1k.md:72,171,213,296`).
  - The official doc uses `REPO_ID=behavior-1k/2026-challenge-demos` (`B1K:docs/challenge/baselines.md:36,106`), so the provided checkpoint would most likely have `assets/behavior-1k/2026-challenge-demos/norm_stats.json` [inferred]. **NOT VERIFIED:** the Google Drive checkpoint (`baselines.md:18`) could not be downloaded.

---

## 9. Exact commands (docs) and doc-vs-code discrepancies

### 9.1 Official BEHAVIOR page (generated from `B1K:docs/challenge/baselines.md`) [docs]

```bash
# setup (baselines.md:85-90)
git clone -b behavior https://github.com/wensi-ai/openpi.git $OPENPI_DIR
cd $OPENPI_DIR && git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync && source .venv/bin/activate && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# norm stats (baselines.md:100-104)
uv run scripts/compute_norm_stats.py pi05_b1k --data.repo_id=$REPO_ID --data.base_config.dataset_root=$DATASET_PATH
# train (baselines.md:118-123)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi05_b1k --exp_name=$EXP_NAME --overwrite --batch_size=64 \
    --data.repo_id=$REPO_ID --data.base_config.dataset_root=$DATASET_PATH
# serve (baselines.md:171-182)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro --task b1k/$TASK_NAME --repo-id $REPO_ID \
    --policy.config pi05_b1k --policy.dir $PATH_TO_CKPT \
    --control_mode receding_horizon --action_horizon 16 --port 8000
# evaluate (baselines.md:301-312)
conda activate behavior && cd $PATH_TO_BEHAVIOR_1K
python -m omnigibson.eval.eval --task-name $TASK_NAME --host 127.0.0.1 --port 8000 --output-dir $LOG_PATH --write-video
```

Placeholders [docs] (`baselines.md:31-37`):
- `TASK_NAME=turning_on_radio`
- `DATA_ROOT=~/2026-challenge-demos`
- `DATASET_PATH=$DATA_ROOT`
- `REPO_ID=behavior-1k/2026-challenge-demos` ("HF hub id, not a local path")

### 9.2 Fork tutorial (`OP:docs/b1k.md`) [docs]

- Serve (lines 246-252): `uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/<TASK_NAME> policy:checkpoint --policy.config <CONFIG_NAME> --policy.dir <CHECKPOINT_DIR>`
- Flag table (lines 273-279): `--repo_id` (default "task bucket/name"), `--control_mode receding_horizon`, `--action_horizon 16`, `--port 8000`, `--record false`
- Norm stats (line 165): `uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>`
- Train overrides (lines 216-226) include `--data.repo_id`, `--data.base_config.dataset_root`, `--data.robot_config_name`.

### 9.3 Discrepancies

1. **Robot naming** [inferred]: the fork doc says the server "accepts observations keyed by the R1Pro `obs_key` definitions" (`docs/b1k.md:269`). Those keys use `robot::`, while the official evaluator's default robot name is `robot_r1` (`B1K:docs/challenge/evaluation.md:168,195`). As shipped, the server would raise `KeyError('robot::proprio')`. An open PR (#5) renames them to `robot_r1`.
2. **`--repo_id` default** [code]: the fork doc example omits `--repo_id`, so `asset_id="b1k/<TASK>"` (`serve_b1k.py:53`). Training writes `assets/<repo_id>/`, e.g. `turning_on_radio` or `behavior-1k/2026-challenge-demos`. Norm-stats loading would fail with `FileNotFoundError` (`normalize.py:141-146`). The official doc passes `--repo-id`, which is correct.
3. **`policy:checkpoint` subcommand** in `docs/b1k.md:249,259` [inferred]: `serve_b1k.Args.policy` is a plain `Checkpoint` field, not a `Checkpoint | Default` union as in `serve_policy.py:55`. tyro would create no subcommand, so the token would probably be rejected. The official page omits it.
4. **compute_norm_stats CLI** [inferred]: `main(config_name: str, max_frames=None)` via `tyro.cli(main)` (`compute_norm_stats.py:134,166`) has no `--data.*` arguments, and the config name is a flag (`--config-name`, as in the fork doc), not a positional. The official command `compute_norm_stats.py pi05_b1k --data.repo_id=... --data.base_config.dataset_root=...` probably fails. The official page's alternative (edit `repo_id`/`dataset_root` in the `pi05_b1k` block) works.
5. **`--data.base_config.dataset_root` override** [inferred]: `DataConfigFactory.base_config` is annotated `tyro.conf.Suppress[DataConfig | None]` (`config.py:185`), so tyro hides it. Both docs' `--data.base_config.dataset_root` override for training (`docs/b1k.md:223`, `baselines.md:123,135`) probably fails. Editing `config.py` is the safe route.
6. **tyro flag style** [inferred, unverified]: the official doc mixes `--repo-id` and `--control_mode`; the fork doc uses `--repo_id`. tyro 0.9.22 is pinned (`uv.lock:4795-4796`). I believe it accepts both forms but could not run it.
7. **Submodules** [inferred]: the official doc runs `git submodule update`, but the fork has no `.gitmodules` (harmless).
8. **Dataset layout** [docs vs code]: the fork doc tells you to point at a per-task dataset root (`docs/b1k.md:70`), as the hard-coded config does (`config.py:763`). The official doc points at the whole 100-task LeRobot v3 root with `REPO_ID=behavior-1k/2026-challenge-demos`. Whether the forked lerobot can load a root where only `chunk-000` was downloaded is **NOT FOUND** (lerobot fork not cloned).
9. **Stale example** [code]: `examples/simple_client/test_b1k_client.py:8` imports the nonexistent `OpenPIWrapper`.
10. **sbatch script** [inferred]: missing backslash (sbatch lines 47-48), and it always passes `--resume`. If `--overwrite` were forwarded, `TrainConfig.__post_init__` would raise "Cannot resume and overwrite at the same time" (`config.py:666-668`).

---

## 10. Open PRs / other branches (unmerged, but informative)

- **PR #5** (`andi/vector`, head `36f8177`, 2026-07-24, "Preserve policy semantics in batched B1K inference"), which adds:
  - `robot_r1` naming
  - `Policy.infer_batch`: one batched forward, with per-example RNG noise kept identical to sequential calls
  - per-connection wrapper copies
  - `reset_connection_state`
  - `__action_chunk_size__` support with validation: only `receding_horizon`, `K <= action_horizon` and `K <= max_len`; it builds `action_chunk` by calling `act` K times
  - Source: diff vs `behavior`.
- **PR #6** (`8360c47`, 2026-08-24): declares `DataConfig.episodes_index`.
- **PR #7** (`c8deb66`, 2026-09-11, third-party, "Add policy stage monitoring and offline navigation evaluation tools"), based on the older 2025 code:
  - Adds `StageTrackingMixin` with `start_stage`/`query_stage`/`observe_stage`/`finish_stage`. Starting a stage "clears its cached action chunks and temporal ensemble, and sets the model prompt to the supplied instruction" (`PR7: src/openpi/policies/stage_tracking.md`).
  - Adds a navigation-arrival rule: XY ≤ 1.1 m, speed ≤ 0.03 m/s, yaw rate ≤ 0.05 rad/s, sustained for more than 1 s.
  - This is prior art for planner-driven prompt switching.
- **`2025-challenge` branch** (83ab06a, 2025-12-30): `scripts/serve_b1k.py` read the prompt from the dataset metadata (`BehaviorLerobotDatasetMetadata(repo_id="behavior-1k/2025-challenge-demos", tasks=[task_name]...); prompt = list(metadata.tasks.values())[0]`, lines 74-83), defaulting to `turning_on_radio`.

---

## 11. Web context (snippets only, low weight)

- Community HF checkpoints exist for π0.5 on BEHAVIOR-1K 2026 data [snippet]:
  - a task-16 single-task fine-tune: https://huggingface.co/Hoshipu/behavior1k-task16-moving-boxes
  - 25-task orbax checkpoints served with `--policy.dir`: https://huggingface.co/0Corvid0/pi05-b1k-eval-checkpoints
- 2025 runner-up "Openpi Comet" (π0.5-based; task balancing; rejection-sampling fine-tuning) [snippet]: https://arxiv.org/pdf/2512.10071
- Some community π0.5 variants reportedly use task embeddings instead of text [snippet]: https://huggingface.co/Shawn3636/pi05-pcd-sft-step10k
- No official web page beyond the repo docs was reachable (behavior.stanford.edu and HF are blocked for fetch).
