# R2 — Official BEHAVIOR-1K 2026 evaluator code (HEAD = v3.9.3-post2)

Scope: what the official evaluator at `B1K` HEAD (tag `v3.9.3-post2`, commit `a8247a8`) actually does, read from code. Nothing was executed (no simulator here).

Conventions:
- `B1K` = `/home/user/stanfordvl/behavior-1k`. Unless prefixed, a path like `eval/evaluator.py:123` means `B1K/OmniGibson/omnigibson/eval/evaluator.py` line 123. `OG/` = `B1K/OmniGibson/omnigibson/`.
- `ref:path:line` means `git -C B1K show ref:path` (refs: `origin/2026-eval`, `origin/main`, `origin/group-time-budget`).
- `OPENPI` = `/home/user/wensi-ai/openpi` (branch `behavior`, `0cc8e35`).
- Labels: **[code]** verified by reading code; **[docs]** stated in `B1K/docs/challenge/*.md` (these are the official web pages); **[inferred]** my deduction; **NOT FOUND** where I looked and found nothing.

Files in the eval dir (`OG/eval/`, 3058 lines): `eval.py` (199), `evaluator.py` (736), `policies.py` (97), `r1pro.yaml` (126), `utils/{dataset_utils,eval_utils,light_utils,network_utils,obs_utils,score_utils}.py`, `wrappers/{default_wrapper,rgbd_full_res_wrapper}.py`, empty `__init__.py` files. No hydra/yaml eval config: the entry point uses argparse (`eval/eval.py:36-113`) and builds an OmegaConf dict in code (`eval/eval.py:154-169`). The only yaml it loads is the robot config (`eval/r1pro.yaml`, or `--robot-config`).

---

## 0. TL;DR for building a planner + executor policy server

1. The evaluator is a **websocket client**. Your code is the **server**: it serves HTTP `GET /healthz` → 200, accepts a WS connection, **sends a msgpack metadata dict first**, then loops: receive a msgpack obs dict → send back a msgpack `{"action": ndarray}`. It also receives `{"reset": True}` messages and **must not reply** to them. [code] `eval/utils/network_utils.py:75-111,113-163,172-179,215-247`.
2. Every obs value has a leading batch dim `N=num_envs` (it is 1 for single-env runs). Keys (default R1Pro, robot name `robot_r1`): three `robot_r1::robot_r1:<link>:Camera:0::rgb` (N,H,W,4) uint8, plus `::depth_linear` (N,H,W) float32 meters when you use `RGBDFullResWrapper`, `robot_r1::proprio` (N,61) float32, `robot_r1::cam_rel_poses` (N,21), `task_id` (N,1) int64. **No language instruction and no BDDL text are sent.** Only the integer `task_id` arrives. [code] `eval/evaluator.py:411-421,576-592`.
3. The R1Pro action is 23-D: `base[0:3]` = robot-frame velocity (vx, vy, wz). The base input is clipped to [-1,1] and scaled to ±0.75 m/s and ±1.0 rad/s. `torso[3:7]` = **absolute** joint positions (rad). `left_arm[7:14]` = absolute joint positions. `left_gripper[14]` is in [-1,1]. `right_arm[15:22]` = absolute joint positions. `right_gripper[22]` is in [-1,1]. [code] `eval/utils/eval_utils.py:105-117`, `eval/r1pro.yaml:35-82`. **An all-zero action is not a no-op**: it drives the arms and torso to joint angle 0 (§2.6).
4. At HEAD the loop is **synchronous** with **no wall-clock timeout** in the evaluator. The simulator waits for every reply (`eval/evaluator.py:423-437`). Official final-eval rules (docs on `main`) and the unreleased `2026/eval` branch add a wall-clock budget of `max_steps × 1 s` per rollout, a 600 s per-query limit, a free first query (up to 600 s), and 3 reconnects (§4.3).
5. The default episode length is `int(1.5 × mean human-demo length)` steps (`eval/evaluator.py:281-286`). Because of an off-by-one, the env is truncated on step **max_steps+1** (§4.1). Success (all BDDL goal predicates true) ends the rollout immediately.

---

## 1. CLI / config parameters of the eval entry (`eval/eval.py`)

### 1.1 argparse flags [code] `eval/eval.py:36-113`

| Flag | Type / default | Meaning (code) | Cite |
|---|---|---|---|
| `--task-name` | str, **required** | BEHAVIOR task name; must be in `available_tasks.yaml` | `eval.py:38`, `evaluator.py:245-247` |
| `--host` | `"127.0.0.1"` | policy WS server host | `eval.py:39` |
| `--port` | int `8000` | policy WS server port | `eval.py:40` |
| `--robot-config` | str `None` | YAML/JSON with one complete robot dict (`model`, `name`, optional `eval.camera_sensor_names`, `eval.use_presampled_pose_key`); `None` → bundled `eval/r1pro.yaml` | `eval.py:41-49,138-142`, `evaluator.py:55,327-341` |
| `--instance-indices` | int list, `[0]` | `train`: direct instance IDs. `public_test`/`hidden_test`: indices in `range(20)` into the split | `eval.py:50-59`, `evaluator.py:73-87` |
| `--mode` | `train`/`public_test`/`hidden_test`, default `public_test` | instance split | `eval.py:60-65` |
| `--num-rollouts` | int `1` | repeat the same batch; `rollout_id` = 0..n-1 | `eval.py:66,181-190` |
| `--num-envs` | int `1` | must equal `len(instance_indices)`, else `ValueError` | `eval.py:67-72,132-136` |
| `--replay-action-chunk-size` | int `0` | K>1 opts into the action-chunk protocol (§6.4) | `eval.py:73-81,149` |
| `--max-steps` | int `None` | episode timeout in steps; None → `1.5 × mean human length` | `eval.py:82-87`, `evaluator.py:281-289` |
| `--env-wrapper` | `"omnigibson.eval.wrappers.DefaultWrapper"` | hydra `_target_` of the obs wrapper | `eval.py:88-92,156` |
| `--policy` | `websocket`/`local`, default `websocket` | `local` = zero actions (smoke test) | `eval.py:93-98,144-152` |
| `--output-dir` | `"/tmp/b1k_eval"` | results root; JSON in `<out>/json`, videos in `<out>/videos` | `eval.py:99,171-175` |
| `--write-video` / `--no-write-video` | `False` | MP4 per rollout | `eval.py:100-105` |
| `--video-fps` | int `30` | video FPS | `eval.py:106` |
| `--headless` / `--no-headless` | `True` | sets `gm.HEADLESS`; when headless it also disables the 1280x720 viewer camera unless `OMNIGIBSON_KEEP_VIEWER_CAMERA=1` | `eval.py:107-112,119-125` |

Hard-coded in the cfg built by `eval.py` [code] `eval/eval.py:154-169`:

```python
cfg = OmegaConf.create(
    {
        "env_wrapper": {"_target_": args.env_wrapper},
        "policy_name": args.policy,
        "model": model_cfg,
        "headless": args.headless,
        "partial_scene_load": True,          # always True from the CLI
        "max_steps": args.max_steps,
        "write_video": args.write_video,
        "mode": args.mode,
        "seed": seed,
        "num_envs": args.num_envs,
        "task": {"name": args.task_name},
        "robot": robot_config,
    }
)
```

The websocket `model_cfg` is `{"_target_": "omnigibson.eval.policies.WebsocketPolicy", "host", "port", "action_chunk_size"}`. At HEAD `allow_reconnect` is not passed, so it stays `False` (`eval.py:144-150`, `policies.py:63`). The local `model_cfg` is `{"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}` (`eval.py:152`).

**Seeds** [code]: `seed_everything(DEFAULT_EVAL_SEED)` runs once in `main()` with `DEFAULT_EVAL_SEED = 0` (`eval.py:127`, `eval_utils.py:16`). It seeds `random`, `numpy`, `torch` (and CUDA), and warp. It sets `cudnn.benchmark=False` and `cudnn.deterministic=True`. `deterministic_torch` stays False, so `use_deterministic_algorithms` is not called (`eval_utils.py:24-57`). There is no per-instance or per-rollout reseed. `cfg.seed` and `cfg.headless` are stored but `BatchedEvaluator` never reads them (grep of `evaluator.py` finds no use).

### 1.2 Docs vs CLI [docs] `docs/challenge/evaluation.md:37-101`
- The docs say to use instance indices `0 1 2 3 4 5 6 7 8 9` for reported results. They also say "Indices `0-19` are public test instances; indices `20-39` are hidden instances" (`evaluation.md:66`). In code, hidden instances are reached with `--mode hidden_test` plus indices in `range(20)`. Indices 20-39 fail the assert under the default mode (`evaluator.py:84-86`). See discrepancies.
- The docs call `RGBDFullResWrapper` the "official RGB + depth challenge-track evaluation" wrapper (`evaluation.md:82,140`). The CLI default is still `DefaultWrapper`, which gives 224×224 RGB with no depth (`eval.py:89-90`).

### 1.3 Fixed environment config [code] `eval/utils/eval_utils.py:191-232`, `eval/evaluator.py:241-298`
- `env.action_frequency=30`, `rendering_frequency=30`, `physics_frequency=120` (`eval_utils.py:203-207`). So each env step = 1/30 s of sim time with 4 physics substeps [inferred from 120/30].
- `scene.type=InteractiveTraversableScene`, `scene_model=task_cfg["scene_model"]`, `include_robots=False` (`eval_utils.py:208-214`).
- `task.type=BehaviorTask`, `activity_definition_id=0`, `activity_instance_id=0`, `online_object_sampling=False`, `highlight_task_relevant_objects=False`, `termination_config.max_steps=5000` (overwritten later), `include_obs=False` (`eval_utils.py:215-230`). `evaluator.py:290` sets `include_obs=False` again.
- Partial scene: `cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS[task_name]` (`evaluator.py:253-254`). The value comes from column `"Rooms to inlcude"` of `DATA_PATH/2026-challenge-task-instances/metadata/B100_task_misc.csv` (`eval_utils.py:60-71`).
- `gm.USE_GPU_DYNAMICS=False`, `gm.ENABLE_TRANSITION_RULES=True` (`evaluator.py:58-59`). `ToggleableMachineRule`, `MixingToolRule` and `CookingSystemRule` are disabled (`evaluator.py:242-243`; list at `B1K/joylo/gello/utils/og_teleop_cfg.py:438`).
- `env.automatic_reset` defaults to `False` (`OG/envs/env_base.py:1027`). `num_rerenders_on_reset` defaults to 3 (`env_base.py:1033`). `origin/2026-eval` sets it to 10 (`origin/2026-eval:OmniGibson/omnigibson/eval/evaluator.py:378`).
- Robot tweaks (applied to the env, not visible to the policy): for `r1`/`r1pro`, `base_footprint_link.mass = 250.0` (`evaluator.py:53,343-352`), and the head camera `horizontal_aperture = 40.0` (`evaluator.py:54,354-359`).
- Robot start pose: `position`/`orientation` from `available_tasks.yaml` (`evaluator.py:338-339`), then overwritten per instance by the TRO `robot_poses` (§3).
- Task metadata files are loaded from `gm.DATA_PATH`: `2026-challenge-task-instances/metadata/{available_tasks.yaml, B100_task_misc.csv, task.jsonl}` (`B1K/joylo/gello/utils/og_teleop_utils.py:1326-1341`, `eval_utils.py:63`, `score_utils.py:25-26`). **NOT FOUND** in the repo (`find B1K -name B100_task_misc.csv -o -name task.jsonl -o -name available_tasks.yaml` returns nothing). They ship with the dataset.

---

## 2. The per-step loop

### 2.1 Control flow [code]
`main` → `BatchedEvaluator(cfg)` → `evaluator.run(instances, ...)` once per rollout (`eval.py:178-190`). `run` → `evaluate_instances_batched(load_fn, step_fn, record_fn)` (`evaluator.py:636-697`):

```python
# evaluator.py:115-139
env_idx_to_instance = {env_idx: instance for env_idx, instance in enumerate(instances)}
load_fn(dict(env_idx_to_instance))
active = {env_idx: True for env_idx in env_idx_to_instance}
step = 0
while any(active.values()):
    active_env_indices = sorted(env_idx for env_idx, is_active in active.items() if is_active)
    terminated, truncated = step_fn(active_env_indices)
    step += 1
    hit_cap = max_steps is not None and step >= max_steps     # run() never passes max_steps -> None
    for env_idx in active_env_indices:
        term = bool(terminated[env_idx]); trunc = bool(truncated[env_idx]) or hit_cap
        if term or trunc:
            results[...] = record_fn(env_idx=..., instance=..., step=step, terminated=term, truncated=trunc)
            active[env_idx] = False
```

Each step queries the policy once, then steps the env once:

```python
# evaluator.py:423-437
def _step_fn(self, active_env_indices):
    action_dim = self.instance_eval_states[0].env_accessor.robot.action_dim
    batched_action = self.policy.forward(obs=self._batch_obs())  # (num_envs, action_dim)
    if batched_action.ndim == 1:
        batched_action = batched_action.unsqueeze(0)
    actions = th.zeros((self.num_envs, action_dim), dtype=th.float32)
    for env_idx in active_env_indices:
        actions[env_idx] = batched_action[env_idx].to(actions.dtype)
    terminated, truncated, _ = self._apply_actions(actions, active_env_indices)
    return terminated, truncated
```

- **Synchronous** [code]: `policy.forward` blocks on `ws.send` + `ws.recv` (`network_utils.py:131-133`) before `env.step` is called (`evaluator.py:430-436`). Simulated time does not advance while the policy computes. At HEAD there is no evaluator-side wall-clock limit; the only timers are the websocket keepalive settings `ping_interval=60, ping_timeout=300` (`network_utils.py:98-105`). [inferred] If a synchronous server blocks its asyncio event loop for more than about 300 s, it cannot answer pings, and the client may drop the connection. With `allow_reconnect=False` that raises `RuntimeError` (`network_utils.py:165-170`). Keep long LLM calls off the event-loop thread. (`websockets>=15.0.1` is pinned at `B1K/OmniGibson/setup.py:56`.)
- The policy is queried once per sim step; there are as many queries as steps unless chunk replay is on (§6.4). With N>1 envs, one call carries the whole batch (`evaluator.py:411-421`).
- Finished envs keep the last obs in the batch, so batch size stays N. They get an all-zero action and are still stepped by the shared sim, but their metrics are frozen (`evaluator.py:411-421,433-435`). This is tested at `B1K/OmniGibson/tests/test_evaluator.py:82-139`: step_calls == `[[1,2],[0,0]]`.
- `env.step(actions, n_render_iterations=1)` (`evaluator.py:386`). Inside: each robot's `apply_action(slice)` runs, then `og.sim.step()`, then `get_obs()`, then `task.step()` (termination), then `_current_steps += 1` (`env_base.py:718-807`).
- Light task special case: for `turning_out_all_lights_before_sleep`, after each step light visibility is re-synced, the scene is rendered 3 more times, and obs are fetched again (`evaluator.py:52,387-396`).

### 2.2 What dict is sent (obs construction) [code]

```python
# evaluator.py:576-592
def _preprocess_obs(self, obs, instance_eval_state):
    robot = instance_eval_state.env_accessor.robot
    obs = flatten_obs_dict(obs)                               # nested -> "a::b::c" keys
    base_pose = robot.get_position_orientation()
    cam_rel_poses = []
    for camera_name in self.robot_camera_names.values():     # order of eval.camera_sensor_names
        sensor_name = camera_name.split("::")[1]
        if sensor_name not in robot.sensors:
            continue
        camera = robot.sensors[sensor_name]
        cam_rel_poses.append(th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose)))
    if cam_rel_poses:
        obs[f"{robot.name}::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
    obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
    return obs

# evaluator.py:411-421
def _batch_obs(self):
    keys = self.instance_eval_states[0].obs.keys()
    return {key: th.stack([s.obs[key] for s in self.instance_eval_states], dim=0) for key in keys}
```

`flatten_obs_dict` joins nested keys with `::` (`eval_utils.py:235-247`). The raw obs is `{robot_name: {sensor_name: {modality: tensor}, "proprio": tensor}}` (`env_base.py:576-622`). Task obs (`"task"` key) is empty because `include_obs=False` (`OG/tasks/task_base.py:353-354`), and there are no external sensors. Sensor names are `f"{robot.name}:{link_name}:{prim_type}:{count}"` (`OG/robots/robot.py:1428-1433`).

**Resulting keys and shapes for the default R1Pro config** (`name: robot_r1`, `eval/r1pro.yaml:2-7`). N = num_envs; the batch dim is always there, even when N=1 (`evaluator.py:419`):

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` | (N,H,W,4) | uint8 | RGBA, 0-255 (`OG/sensors/vision_sensor.py:966`) |
| `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` | (N,H,W,4) | uint8 | |
| `robot_r1::robot_r1:zed_link:Camera:0::rgb` | (N,H,W,4) | uint8 | head |
| `...::depth_linear` (each camera; only with `RGBDFullResWrapper`) | (N,H,W) | float32 | `distance_to_image_plane` (`vision_sensor.py:81,968`), range [0, inf) in the obs space, not clipped by the evaluator |
| `robot_r1::proprio` | (N,61) | float32 [inferred dtype: obs space FLOAT32 `robot.py:1654-1656`] | layout §2.3 |
| `robot_r1::cam_rel_poses` | (N,21) | float32 [inferred] | 3 cameras × (pos xyz + quat xyzw) in the robot base frame, in `camera_sensor_names` order: left_wrist, right_wrist, head |
| `task_id` | (N,1) | int64 | `B100_task_misc.csv` "Task ID" |

**Resolutions per wrapper** [code]:
- `DefaultWrapper`: modalities `["rgb"]`; all 3 roles at **224×224** (`eval/wrappers/default_wrapper.py:3,22-28`).
- `RGBDFullResWrapper`: modalities `["rgb","depth_linear"]`; head **720×720**, wrists **480×480** (`eval/wrappers/rgbd_full_res_wrapper.py:19-22`, `eval_utils.py:10-11`).
- Resolutions are written into the robot `sensor_config` per link key (`"<link>:Camera:0"`) before the env is created. Only roles listed in `eval.camera_sensor_names` get an override (`evaluator.py:268-279`). Any other camera prim falls back to the robot config's class default `VisionSensor` 1080×1080 (`eval/r1pro.yaml:30-34`) [inferred from `robot.py:1408-1415`].
- Per the obs space the RGB images have 4 channels (RGBA). Servers should take `[..., :3]`, as OPENPI does (`OPENPI/src/openpi/shared/eval_b1k_wrapper.py:79`).
- Tensors are CPU in single-env mode. With `OMNIGIBSON_ENABLE_TILED_RENDERING` and N>1 they are CUDA tensors (`evaluator.py:601-602`, `OG/macros.py:166`); msgpack packing calls `.cpu()` anyway (`network_utils.py:299-300`). Docs on `main` say tiled rendering will **not** be used for final evaluation (`origin/main:docs/challenge/updates.md:53`).

**Depth units/range** [code + inferred]: `depth_linear` = Isaac `distance_to_image_plane`, in meters [inferred: standard Isaac/Replicator units; OmniGibson uses meters]. The evaluator does not clip it; obs space bound `[0, inf)` (`vision_sensor.py:968`); the default camera clipping range is `(0.001, 1e7)` (`vision_sensor.py:142`) unless the robot USD overrides it. The training dataset depth is log-quantized and clipped to `[MIN_DEPTH=0.01, MAX_DEPTH=10.0]` m with `DEPTH_SHIFT=3.5` (`eval/utils/obs_utils.py:24-62,65-96`; used by `OG/envs/lerobot_data_wrapper.py:13`). [inferred] A server should clip live depth to [0.01, 10] m so it matches training data.

**Camera intrinsics** (constants, not sent) [code] `eval_utils.py:80-90`: head 720×720 `fx=fy=306.0, cx=cy=360`; wrists 480×480 `fx=fy=388.6639, cx=cy=240`. `CAMERA_EXTRINSICS` is a copy-paste of the intrinsics, not real extrinsics (`eval_utils.py:92-102`). Use `cam_rel_poses` for extrinsics.

**Is a task name / instruction / BDDL sent?** [code] No. The only task information sent is `obs["task_id"]` (an int64 index) (`evaluator.py:591`). The baseline OpenPI server gets its text prompt at **startup** from `--task` (`OPENPI/scripts/b1k/serve_b1k.py:44-61`) and injects it per query (`OPENPI/src/openpi/shared/eval_b1k_wrapper.py:94`). [docs] "BDDL task definitions can be used and are identical during evaluation" (`evaluation.md:28`).

### 2.3 Proprio vector layout (R1Pro, 61-D) [code]
`get_proprioception()` concatenates `proprio_dict[k]` for `k in self._proprio_obs` in config order (`robot.py:1530-1539`). The config order is `eval/r1pro.yaml:13-28`. The indices match `PROPRIOCEPTION_INDICES["R1Pro"]` (`eval_utils.py:121-141`):

| Slice | Key | Meaning (code) |
|---|---|---|
| 0:3 | `base_qvel` | base velocity **in robot-local frame** `[vx, vy, wz]` (holonomic x/y joint velocities rotated by base yaw) (`robot.py:1601,1623-1641`) |
| 3:10 | `arm_left_qpos` | 7 left-arm joint positions (rad) (`robot.py:1577`) |
| 10:17 | `arm_left_qvel` | 7 left-arm joint velocities |
| 17:20 | `eef_left_pos` | left EEF position **relative to the robot base** (`robot.py:1582-1584,3056-3072`) |
| 20:24 | `eef_left_quat` | left EEF orientation quaternion (x,y,z,w) relative to the base (`OG/utils/transform_utils.py:950-972`) |
| 24:26 | `gripper_left_qpos` | 2 finger joint positions |
| 26:28 | `gripper_left_qvel` | 2 finger joint velocities |
| 28:35 | `arm_right_qpos` | 7 |
| 35:42 | `arm_right_qvel` | 7 |
| 42:45 | `eef_right_pos` | relative to base |
| 45:49 | `eef_right_quat` | xyzw, relative to base |
| 49:51 | `gripper_right_qpos` | 2 |
| 51:53 | `gripper_right_qvel` | 2 |
| 53:57 | `trunk_qpos` | 4 torso joint positions (`robot.py:1591`) |
| 57:61 | `trunk_qvel` | 4 torso joint velocities |

There is no global robot pose (`robot_pos`) in the proprio. A custom `proprio_obs` could add one, but the rules forbid using the "robot global pose" (`evaluation.md:23`).

### 2.4 Action layout (R1Pro, 23-D) [code + inferred]
`env.step` slices the flat action per robot, and `robot.apply_action` walks controllers in `_controllers` order, each taking `command_dim` entries (`env_base.py:745-759`, `robot.py:758-804`). The controller order comes from the R1Pro definition YAML `raw_controller_order` in the dataset `omnigibson-robot-assets/models/r1pro/r1pro.yaml` (`robot.py:215-222,2391`). That file is **NOT FOUND** in the repo; only `r1.yaml` is shown in the tutorial, with `["base","trunk","arm_left","gripper_left","arm_right","gripper_right"]` (`B1K/docs/tutorials/custom_robot_import.md:497`). The eval code's own constant:

```python
# eval_utils.py:105-117
ACTION_QPOS_INDICES = {"R1Pro": OrderedDict({
    "base": np.s_[0:3], "torso": np.s_[3:7], "left_arm": np.s_[7:14],
    "left_gripper": np.s_[14:15], "right_arm": np.s_[15:22], "right_gripper": np.s_[22:23]})}
```

OPENPI matches this (`action_dim=23`, same slices: `OPENPI/src/openpi/configs/robots/b1k.py:30-38`).

Controller semantics from `eval/r1pro.yaml:35-82` (with `action_normalize: false`, `r1pro.yaml:8`):

| Slice | Controller | Units / semantics |
|---|---|---|
| base 0:3 | `HolonomicBaseJointController`, `motor_type: velocity`, input limits `[-1,1]^3`, output limits `[[-0.75,-0.75,-1.0],[0.75,0.75,1.0]]` | **velocity**, robot-local frame `[vx, vy, wz]` ("absolute values defined in the robot's local frame", `OG/controllers/holonomic_base_joint_controller.py:12-13,142-159,165-167`). Input is clipped to [-1,1] then linearly mapped (`OG/controllers/controller_base.py:304-339`). So 1.0 → 0.75 m/s (x,y) and 1.0 rad/s (yaw) [inferred units m/s, rad/s]. Zero = stop. |
| torso 3:7 | `JointController`, `motor_type: position`, `use_delta_commands: false`, limits `null` | **absolute** joint position targets (rad) [rad inferred for revolute joints], no scaling, clipped to joint limits (`OG/controllers/joint_controller.py:280-290`) |
| left arm 7:14 | `JointController` position, absolute, no scaling | absolute joint positions |
| left gripper 14 | `MultiFingerGripperController`, `mode: smooth`, input/output limits `default` | 1-D command clipped to [-1,1] and linearly mapped to [mean lower finger limit, mean upper finger limit] (`OG/controllers/multi_finger_gripper_controller.py:85-88,159-176,186-198,440-441`). Motor type defaults to `position` (`robot.py:3440-3449`). [inferred] -1 ≈ closed, +1 ≈ open (grasping_direction default `"lower"`, `robot.py:133,200-201`). `grasping_mode: assisted` (`r1pro.yaml:29`): an assisted grasp is attempted whenever the finger control is below the upper joint limit (`robot.py:851-855`) |
| right arm 15:22 | as left arm | |
| right gripper 22 | as left gripper | |
| (camera) | `NullJointController` | [inferred] contributes no action dims, since the 23-D layout leaves no room for it |

**Zero action is not a no-op** [code + inferred]: the arm and torso use absolute position control with no scaling, so an all-zero action commands every arm and torso joint to 0 rad. The gripper value 0 maps to the midpoint between closed and open. Finished envs in a batch get exactly these zeros (`evaluator.py:433-435`), which does not matter because they are already scored. For your own "hold still" action, send base = 0, arms/torso = current `qpos` from proprio, and the gripper = its last command.

Reset joint configuration (`reset_joint_pos`, 28 low-level joints) [code] `eval/r1pro.yaml:83-126`: 6 virtual base joints = 0; torso `[1.025, -1.45, -0.47, 0.0]`; both arms all 0; fingers 0.05 (open). `robot.reset()` applies it before each instance load (`evaluator.py:463`, `robot.py:721-731`).

### 2.5 How actions come back [code] `network_utils.py:137-163`, `policies.py:88-92`, `evaluator.py:430-435`
- The response is a msgpack dict and must contain `"action"`. `th.from_numpy(deepcopy(action_dict["action"])).to(th.float32)`.
- Shape: `(N, action_dim)`. `(action_dim,)` is also accepted when N=1 (unsqueezed). Rows are indexed by env_idx.
- If `"action"` is missing, the client re-sends the same obs up to 2 more times, then raises (`network_utils.py:138-144`).
- If the server sends a **text** frame, the client raises `RuntimeError("Error in inference server:\n...")` (`network_utils.py:134-135`).
- `WebsocketPolicy.forward` returns the cached `last_action` when `obs["need_new_action"]` is present and False (`policies.py:88-92`). The evaluator never sets that key [code: grep], so this path is dead at HEAD.

### 2.6 Reset protocol [code]
- `load_batch` order: `robot.reset()` → load the TRO state per env → settle 25 physics steps with `keep_still()` → `scene.update_initial_file()` → `env.reset(env_indices)` → (light sync) → **`self.policy.reset()` once per batch** → preprocess the first obs → `metric.reset()` → open the video writers (`evaluator.py:448-574`).
- `env.reset` = `task.reset`, `_current_steps=0`, one `og.sim.step()`, 3 renders, `get_obs` (`env_base.py:838-903`). The first obs the policy sees is this post-reset obs.
- `WebsocketPolicy.reset` → `WebsocketClientPolicy.reset()`: it clears the cached chunk, connects if not yet connected (health check + handshake), and **sends `{"reset": True}` with no acknowledgment** (`network_utils.py:172-179`). The reference server handles it with `if "reset" in result: self._policy.reset(); continue`, and sends nothing back (`network_utils.py:226-228`).
- The first connection happens inside the first `policy.reset()`. The connection is reused across rollouts in the same process. Each `run()` (rollout) sends a new reset (`eval.py:181-189`).

---

## 3. Instance loading [code]

```python
# evaluator.py:73-87  (resolve_instance_ids)
if mode == "train": return [int(i) for i in instance_indices]
test_instances = TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES] if mode == "public_test" \
                 else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
assert set(instance_indices).issubset(set(range(num_split_instances)))
return [int(test_instances[i]) for i in instance_indices]
```

- `TEST_INSTANCE_IDS = list(range(301, 341))`, `NUM_PUBLIC_TEST_INSTANCES = 20`, `NUM_HIDDEN_TEST_INSTANCES = 20` (`eval_utils.py:18-21`). Public index i → instance ID 301+i (301..320). Hidden index i → 321+i (321..340).
- [docs] Report results on public indices 0-9 (IDs 301-310), 1 rollout each, with the default timeout. Indices 10-19 are for self-testing. 10 hidden instances are used for final evaluation of top teams (`evaluation.md:282,290`). Full submission = 100 tasks × 10 instances × 1 rollout (`submission.md:10`).
- TRO file path (`evaluator.py:464-481`, `OG/tasks/behavior_task.py:166-181`, `OG/utils/asset_utils.py:125-146`):
  `DATA_PATH/2026-challenge-task-instances/{scenes | scene_test/public | scene_test/private}/<scene>/json/<scene>_task_<activity>_instances/<scene>_task_<activity>_0_<instance_id>_template-tro_state.json`. The mode maps `train → scenes`, `public_test → scene_test/public`, `hidden_test → scene_test/private`. A missing file raises `FileNotFoundError`.
- Contents: a dict of `bddl_inst → object state`, plus `"robot_poses"`. The robot is placed at `get_presampled_robot_poses(...)[0]`: the generic `"robot"` key if present, else `eval.use_presampled_pose_key` or the robot model (`behavior_task.py:35-60`), using `frame="scene"` (`evaluator.py:485-497`). Each object gets `obj.load_state(tro_state, serialized=False)`, with a scene-offset fix for env_idx≠0 (`evaluator.py:498-507`).
- The base scene is the `activity_instance_id=0` template (`eval_utils.py:219`). The TRO state then overrides object states [inferred from `behavior_task.py:188-196` + `evaluator.py:483-507`].
- [docs] "Each instance differs in terms of initial object states and initial robot poses" (`evaluation.md:294`).
- Determinism: only global seeding at startup (§1.1). [docs] The simulator is nondeterministic, and cherry-picking is forbidden (`evaluation.md:286`, `origin/main:docs/challenge/updates.md:25`).

---

## 4. Termination [code]

### 4.1 Step limit
- `max_steps = int(human_stats["length"] * EVAL_TIMEOUT_MULTIPLIER)` with `EVAL_TIMEOUT_MULTIPLIER = 1.5` when `--max-steps` is not given (`evaluator.py:281-289`, `eval_utils.py:17`). `human_stats["length"]` = the mean episode length in frames (30 fps) over the task's demos, read from `task.jsonl` (`score_utils.py:29-46`; generated as `round(total_length/num_episodes, 4)`, `score_utils.py:215-233`).
- It is enforced by `Timeout(max_steps)` in `BehaviorTask._create_termination_conditions` (`behavior_task.py:237-250`): `return env.episode_steps >= self._max_steps` (`OG/termination_conditions/timeout.py:23`). It produces `truncated=True` (`env_base.py:784-790`).
- **Off-by-one** [code-read, not executed]: `task.step()` (the termination check) runs **before** `self._current_steps += 1` (`env_base.py:776,806`). Timeout therefore fires on the call where `episode_steps == max_steps`, which is the (max_steps+1)-th `env.step`. The result JSON will show `steps = max_steps + 1` on timeout.
- The `max_steps` argument of `evaluate_instances_batched` is never passed by `run()` (`evaluator.py:691-697`), so `hit_cap` is always False.
- [inferred] Per-task values cannot be computed here because `task.jsonl` is NOT FOUND in the repo. `docs/challenge/task_data.json` has a per-task `"duration"` (seconds; 63..869, mean 350.2). That fits `max_steps ≈ duration × 30 × 1.5`: the longest task `assembling_gift_baskets` has duration 869, giving about 39,105 steps ≈ 10.86 h at 1 s/step, which matches the "≈10.85 hours … longest task" in `origin/main:docs/challenge/updates.md:15`. Treat this as an approximation only.

### 4.2 Success
`PredicateGoal(SuccessCondition)` checks `compiled_task.check_goal(...)` per env every step (`OG/termination_conditions/predicate_goal.py:7,38-44`, `behavior_task.py:244-248`). When all goal predicates are satisfied, `terminated=True` and the evaluator records the rollout **immediately** (`evaluator.py:125-137`). `success` comes from `task.success[env_idx]` (`evaluator.py:163-165`, `task_base.py:389,421-427`).

### 4.3 Wall-clock timeouts
- **At HEAD: none in code.** There is no time budget in `evaluator.py`, `policies.py` or `network_utils.py` apart from the WS ping (`ping_interval=60, ping_timeout=300`, `network_utils.py:103-104`), the 2 s health-check request timeout and the 5 s retry sleeps (`network_utils.py:85,92,111`).
- **Official rules on `main`** [docs] (`origin/main:docs/challenge/updates.md:15-19`): each rollout must average ≥1 FPS, "enforced through an episode timeout of `max_steps × 1 second`, excluding scene startup". Each action-query round trip must finish within **600 s** and within the remaining episode time. Up to **3 reconnects**, and reconnection time counts toward the budget. The same IP/port is used throughout a rollout, so history-dependent policies work (`updates.md:57`).
- `origin/group-time-budget` docs rewrite this as a per-group budget of `N × max_steps` seconds when N rollouts run together (`origin/group-time-budget:docs/challenge/updates.md:15-19`).
- **Unreleased implementation on `origin/2026-eval`** [code on that ref]:
  - `POLICY_RESPONSE_TIMEOUT = 600`, `MAX_RECONNECTS_PER_ROLLOUT = 3` (`origin/2026-eval:.../eval/utils/network_utils.py:31-32`); `ping_timeout=None` (`:139`).
  - `POLICY_INITIALIZATION_TIMEOUT = 600.0` (`origin/2026-eval:.../eval/evaluator.py:58`).
  - Connect + reset get their own 600 s and are not charged. The **first query is uncharged** but bounded at 600 s (`initialize_fn`, `origin/2026-eval:docs/challenge/evaluation.md:111-113`).
  - After that, the episode deadline is `max_steps` seconds. Single-port: one group deadline (`time_limit_seconds=self.max_steps`, `evaluator.py:884`). Multi-port: per-rollout `set_time_budget(self.max_steps)` (`:796`), charging policy time + sim time / N (`origin/2026-eval:.../eval/policies.py:218-228`).
  - Failures record `q_score.final=0`, `normalized_agent_distance=0`, `normalized_time=1/1.5`, and `failure_reason ∈ {policy_connection_lost, group_time_budget_exceeded, policy_error, policy_query_timeout}` (`evaluator.py:838-852` on that ref).
  - The branch also adds `--policy-endpoints HOST:PORT ...` (one server per env), `--policy-scheme ws|wss`, `allow_reconnect=True` (`origin/2026-eval:.../eval/eval.py:37-66,169-189`), and chunk lengths `1 ≤ L ≤ K` (`network_utils.py:232` on that ref).

---

## 5. Outputs [code]

### 5.1 Per-rollout metrics JSON
Path: `<output-dir>/json/<task_name>_<instance_id>_<rollout_id>.json`. `instance_id` is the resolved ID (e.g. 301), not the index (`eval.py:171-172`, `evaluator.py:680-682`). A file is written for **every** finished rollout, successful or not.

```python
# evaluator.py:676-682
result = {"task": task_name, "instance_id": int(instance), "rollout_id": rollout_id, "steps": step}
result["success"] = success
for metric in instance_eval_state.metrics:          # [AgentMetric, TaskMetric]  (evaluator.py:374-378)
    result.update(metric.aggregate())
json.dump(result, f, indent=2, default=float)
```

Schema (from `OG/metrics/agent_metric.py:57-71`, `OG/metrics/task_metric.py:259-282`):

```json
{
  "task": "<str>", "instance_id": 301, "rollout_id": 0, "steps": <int>, "success": <bool>,
  "agent_distance": {"base": <m>, "left": <m>, "right": <m>},
  "normalized_agent_distance": {"base": human/agent, "left": ..., "right": ...},   // inf if agent==0
  "q_score": {"final": <float in [0,1]>},
  "time": {"simulator_steps": <int>, "simulator_time": steps*rendering_dt, "normalized_time": human_len/steps}
}
```

- `agent_distance.base` = sum of per-step world-frame base position deltas. `left`/`right` = sum of EEF world position deltas (`agent_metric.py:30-55`). The `left`/`right` key names come from `robot.arm_names`; `arm_left_qpos` in the proprio config implies the arm names are `left`/`right` (`robot.py:1577`).
- **Q-score** (`task_metric.py:7-29`): 1.0 on success. Otherwise, for each goal option, (number of predicates **not satisfied at episode start but satisfied at the end**) / (number of predicates in the option); the result is the max over options. The initial satisfaction is captured at `metric.reset()` right after env reset (`task_metric.py:244-253`, `behavior_task.py:206-235`).
- Leaderboard aggregation (`eval/utils/score_utils.py:271-397`): for each task, the per-task mean is `sum(q)/n_instances_per_task`, where n = 20 for public and 20 for hidden. Overall = mean over all tasks in `B100_task_misc.csv`. `task_sr` counts q==1. Time score is `1.5/0.5 − 1/(0.5·normalized_time)` = `3 − 2/normalized_time` (`score_utils.py:321-324`). Only `rollout_id` 0 files are accepted (`score_utils.py:297`). The input dir must be named `<track>.<testset>.<team>.<affiliation>.<date>/json/` with track ∈ {standard, privileged} and testset ∈ {public, hidden} (`score_utils.py:236-254`).

### 5.2 Video
- Only with `--write-video`. That flag requires the roles `left_wrist`, `right_wrist`, `head` in `eval.camera_sensor_names`, else a `ValueError` (`evaluator.py:318-325`).
- Path: `<output-dir>/videos/<task>_<instance_id>_<rollout_id>.mp4` (`evaluator.py:569-574`, `eval.py:173-175`).
- Format: PyAV `libx264`, `yuv420p`, resolution (H=448, W=672), `rate=--video-fps` (`evaluator.py:573`, `obs_utils.py:138-174`).
- Frame layout: `hstack([vstack([left_wrist 224x224, right_wrist 224x224]), head 448x448])`, RGB from `[..., :3]` (`evaluator.py:594-615`, `obs_utils.py:190-195`). One frame per executed step for active envs; the initial reset frame is not written (`evaluator.py:662-667`).

### 5.3 Logs
- Python logger to stdout only. There is no log file. Lines are emitted for: resolved instance IDs (`eval.py:131`); "Setting timeout to be ..." (`evaluator.py:283-288`); per instance "Instance X (env i) finished at step S: success=... q_score=..." (`evaluator.py:686-688`); the final "Eval summary: k/n success | mean q_score=..." (`eval.py:192-195`); totals and success rate on exit (`evaluator.py:703-711`).
- SIGINT triggers a clean shutdown (`evaluator.py:699-722`).

---

## 6. Websocket protocol (`eval/utils/network_utils.py`) — enough to write a server

### 6.1 Transport & handshake [code]
1. The client URI is `ws://{host}:{port}` (`network_utils.py:62`). `api_key` support exists (`Authorization: Api-Key ...`), but eval never sets it (`network_utils.py:97`).
2. **Health check first**: the client polls `GET http(s)://host:port/healthz` (timeout 2 s, retry every 5 s, **forever**) until `response.ok` (`network_utils.py:75-92`). The reference server answers it in `process_request` (`network_utils.py:268-276`).
3. Then it connects with `compression=None, max_size=None, ping_interval=60, ping_timeout=300` and **immediately calls `recv()` expecting a msgpack-packed metadata dict** from the server (`network_utils.py:95-108`). Any dict works; the reference server sends `self._metadata or {}` (`network_utils.py:198,219`). Connection errors are retried every 5 s, forever (`network_utils.py:109-111`).

### 6.2 Messages [code]
- Client → server per step: one **binary** frame = `msgpack.packb(obs_dict, default=pack_data)`, where `obs_dict: str → tensor`. When chunking is enabled it also carries `"__action_chunk_size__": K` (an int) (`network_utils.py:29,122-126`).
- Client → server at rollout start: `{"reset": True}`. **No reply expected** (`network_utils.py:178-179`).
- Server → client: binary msgpack dict `{"action": ndarray (N, action_dim) or (action_dim,), "server_timing": {"infer_ms": float, "prev_total_ms": float}}`. Only `"action"` (and optionally `"action_chunk"`) is read (`network_utils.py:236-246,137-163`).
- Errors: the reference server sends the traceback as a **text** frame only if `gm.DEBUG`, then closes with 1011 (`network_utils.py:252-265`). The client raises on any text frame.

### 6.3 Array encoding (msgpack-numpy variant) [code] `network_utils.py:298-342`

```python
def pack_data(obj):
    if isinstance(obj, th.Tensor):
        data = obj.detach().cpu().numpy()
        return {b"__ndarray__": True, b"data": data.tobytes(), b"dtype": data.dtype.str, b"shape": data.shape}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"): raise ValueError(...)
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

Packer = functools.partial(msgpack.Packer, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
```

- The marker keys are **bytes** (`b"__ndarray__"`). The dtype strings are numpy `.str`, e.g. `'|u1'`, `'<f4'`, `'<i8'`. The server unpacks with `strict_map_key=False` (`network_utils.py:225`). It is byte-compatible with openpi's `openpi_client.msgpack_numpy` [inferred from "Adapted from openpi", `network_utils.py:1-3`; OPENPI's `websocket_b1k_server.py` uses the same scheme].
- Arrays decoded with `np.ndarray(buffer=...)` are **read-only**. The reference server `deepcopy`s them (`network_utils.py:230`).
- Batching: each value has leading dim N (§2.2), and the reply's `action` must have N rows. There is one connection per evaluator process (at HEAD) for all N envs.

### 6.4 Optional action-chunk replay (HEAD semantics) [code] `network_utils.py:113-163`, [docs] `evaluation.md:105-127`
- Enabled only when `--replay-action-chunk-size K` with K>1 (`network_utils.py:67,70`).
- The request adds `"__action_chunk_size__": K`. The reply must contain `action_chunk` with shape **exactly** `(*action.shape[:-1], K, action_dim)`, and `action_chunk[..., 0, :]` must **exactly equal** `action`. Otherwise the client raises `RuntimeError` (`network_utils.py:146-157`).
- The client then returns cached `chunk[..., i, :]` for i=1..K-1 without contacting the server, so the server does not see those intermediate observations (`network_utils.py:114-117`).
- If the reply has no `action_chunk`, the client warns once and falls back to one request per step (`network_utils.py:158-162`).
- The cache is cleared on reset (`network_utils.py:173-174`).
- `origin/2026-eval` relaxes the shape rule to `1 ≤ L ≤ K` per reply (`origin/2026-eval:.../network_utils.py:226-238`).

### 6.5 Minimal compatible server (my sketch, derived from §6.1-6.3; not official code)

```python
# Sketch: mirrors WebsocketPolicyServer (network_utils.py:182-265) but keeps heavy work off the event loop.
import asyncio, http, functools, msgpack, numpy as np
import websockets.asyncio.server as _server
# reuse pack_data/unpack_data exactly as in §6.3
async def handler(ws):
    packer = msgpack.Packer(default=pack_data)
    await ws.send(packer.pack({"server": "my_planner_executor"}))        # metadata FIRST (client recv()s it)
    while True:
        msg = msgpack.unpackb(await ws.recv(), object_hook=unpack_data, strict_map_key=False)
        if "reset" in msg:                                               # no reply!
            policy.reset(); continue
        obs = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in msg.items()}  # writable copies
        act = await asyncio.to_thread(policy.act, obs)                   # (N,23) float32; keeps pings alive
        await ws.send(packer.pack({"action": np.asarray(act, dtype=np.float32)}))
def health(conn, req):
    return conn.respond(http.HTTPStatus.OK, "OK\n") if req.path == "/healthz" else None
async def main():
    async with _server.serve(handler, "0.0.0.0", 8000, compression=None, max_size=None,
                             process_request=health) as s:
        await s.serve_forever()
```

---

## 7. Policy classes and how a custom policy plugs in [code] `eval/policies.py`
- `LocalPolicy(action_dim=None)` (`policies.py:13-47`): `forward(obs)` returns `self.policy.act(obs).detach().cpu()` if `self.policy` is set, else zeros of shape `(batch, action_dim)`. `set_action_dim` is called by the evaluator (`evaluator.py:364-366`). `reset()` calls `self.policy.reset()`. With `--policy local` the result is zero actions (smoke test).
- `WebsocketPolicy(host, port, allow_reconnect=False, action_chunk_size=0)` (`policies.py:50-97`): it wraps `WebsocketClientPolicy`. It also has `update_host(host, port)`, and the `need_new_action` cache described in §2.5.
- `WebsocketClientPolicy` and `WebsocketPolicyServer` live in `network_utils.py:32-265`.
- Plug-in contract [code `evaluator.py:361-372,430`]: `cfg.model` is hydra-`instantiate`d from `_target_`. Any class works if it has `forward(obs: dict[str, Tensor(N,...)]) -> Tensor (N, action_dim)` and `reset()`. `set_action_dim(int)` is optional. The CLI restricts `--policy` to `websocket|local` (`eval.py:93-98`), so an in-process custom policy needs a programmatic `BatchedEvaluator(cfg)` with your own `model._target_`, or an assignment to `LocalPolicy.policy`. For official evaluation the supported path is a **websocket server** (`evaluation.md:37-50,103`).
- `origin/2026-eval` adds `MultiWebsocketPolicy` (one client per env, thread pool, per-env time budgets) and `PolicyBatchResult` (`origin/2026-eval:.../eval/policies.py:19-23,126-233`). It sends **per-env** obs with `unsqueeze(0)`, so the leading dim is 1 per port (`:200`).
- Custom obs wrapper: `--env-wrapper` must point to an `EnvironmentWrapper` subclass with a `@classmethod camera_spec()` returning `{"modalities": [...], "resolution": {role: (H, W)}}` (`evaluator.py:268-279`, `default_wrapper.py:21-28`). [docs] Submitted wrappers are inspected manually and may expose only RGB, depth and proprio (`evaluation.md:144`).

---

## 8. Privileged info / what is filtered [code + docs]
- [docs] The single 2026 track allows **RGB + depth + proprioception** only. Ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose and other simulator-only information are not allowed (`evaluation.md:5-26`). Privileged info is fine for training (`evaluation.md:28`).
- [code] There is **no key-level filter** in the evaluator. Restriction happens by construction:
  - (a) `include_obs=False`, so no task low-dim obs such as object poses or in-gripper flags (`evaluator.py:290`; what would be included is in `behavior_task.py:705-750`).
  - (b) Camera modalities are limited to the wrapper's `camera_spec` (`rgb`, plus `depth_linear` for RGBD) (`evaluator.py:268-269`).
  - (c) `proprio_obs` in `r1pro.yaml:13-28` has no global pose.
  - (d) No external sensors.
- On top of that the evaluator itself adds `robot_r1::cam_rel_poses` (camera poses relative to the base) and `task_id` (`evaluator.py:589-591`). These are organizer-provided and not global pose.
- A custom robot config or wrapper could enable more modalities (e.g. `seg_instance`, `robot_pos`). The docs say organizers inspect this manually (`evaluation.md:144`, `submission.md:37-42`).
- `score_utils.sanity_check_filename` still accepts `track ∈ {"standard","privileged"}` (`score_utils.py:250`), a leftover from 2025.

---

## 9. Page-vs-code discrepancies (and branch drift)
1. Hidden-instance indexing: the docs say "indices 20-39 are hidden" (`evaluation.md:66`). The code uses `--mode hidden_test` with indices 0-19 → IDs 321-340, and asserts `range(20)` (`evaluator.py:78-87`).
2. Docs: "Each successful rollout produces a JSON result" (`evaluation.md:129`). The code writes JSON for every rollout (`evaluator.py:669-682`).
3. Docs: in `RGBDFullResWrapper` "all other vision sensors are treated as wrist-resolution sensors" (`evaluation.md:216`). The code only resizes sensors named in `eval.camera_sensor_names`; others keep the robot config default of 1080×1080 (`evaluator.py:273-279`, `r1pro.yaml:30-34`).
4. Docs call RGBD "official", but the CLI default is `DefaultWrapper` (224 RGB, no depth) (`evaluation.md:82` vs `eval.py:89-90`).
5. Sample JSON in `submission.md:46-71` shows `"instance_id": 0`. Code writes the resolved ID, e.g. 301 (`evaluator.py:676`), and score_utils expects 301-320 / 321-340 in filenames (`score_utils.py:291-299`).
6. Off-by-one: the docs' `--max-steps` is the "episode timeout in simulator steps" (`evaluation.md:78`), but the Timeout fires on step max_steps+1 (`timeout.py:23`, `env_base.py:776,806`).
7. score_utils divides each task by `NUM_PUBLIC_TEST_INSTANCES=20` (`score_utils.py:300,340`), while docs say to report public indices 0-9 (10 instances) (`evaluation.md:282`). With only 10 files the local score would be halved [inferred]. The leaderboard scorer is not in the repo (NOT FOUND).
8. `CAMERA_EXTRINSICS` duplicates the intrinsics matrices (`eval_utils.py:92-102`); it is a placeholder.
9. `evaluate_instances_batched(max_steps=...)` is never passed (`evaluator.py:691-697`); only the env Timeout applies.
10. OPENPI fork naming: `OPENPI/src/openpi/configs/robots/b1k.py:7-26` uses `name="robot"` and keys like `robot::robot:zed_link:Camera:0::rgb`, `robot::proprio` (`eval_b1k_wrapper.py:72`). HEAD's `r1pro.yaml:2` names the robot `robot_r1`, so live keys are `robot_r1::robot_r1:zed_link:Camera:0::rgb` [inferred: KeyError unless the fork or config is aligned]. The remote `behavior` branch on GitHub (raw.githubusercontent, fetched 2026-10-21) still has `name="robot"`.
11. Wall-clock rules: docs on `main` (`updates.md:15-19`) describe a per-episode `max_steps × 1 s` timeout, a 600 s per-query limit and 3 reconnects, but **none of this is in HEAD code**. It exists only on `origin/2026-eval`. Also, `origin/group-time-budget` docs say a group budget of `N × max_steps` s, while `origin/2026-eval` code uses `time_limit_seconds=self.max_steps` for a single-port group regardless of N (`origin/2026-eval:.../evaluator.py:884`).
12. The docs on `origin/2026-eval:docs/challenge/evaluation.md:113-115` describe the first-query allowance. HEAD's `evaluation.md` has no such text.

---

## 10. Implications for our planner + executor agent
- **Instruction source**: the evaluator sends only `task_id`. A single server for all 100 tasks must map `task_id → task name → instruction/BDDL`, using `B100_task_misc.csv` "Task ID" (dataset metadata) and `docs/challenge/task_data.json` instructions. Or launch one server per task like OpenPI (`serve_b1k.py --task`). BDDL use is allowed (`evaluation.md:28`).
- **Latency budget**: at HEAD the sim waits for us. The official 2026 rules, however, give ≈ `max_steps` seconds of wall clock per rollout (≈1 s/step on average, including sim step time), plus one free first query of up to 600 s (`origin/2026-eval`). Docs benchmark sim speed at 13.5 FPS for RGBD 720/480 on an RTX 4090 with random actions (`evaluation.md:366-368`), so the sim uses about 0.07 s/step and leaves about 0.9 s/step for inference on average [inferred]. A slow LLM planner must be called rarely, e.g. at subtask boundaries. Use the free first query for warm-up or initial planning, and use action chunks (`--replay-action-chunk-size`) to cut round trips. Whether the organizers enable chunk replay in the final run is unclear (it is a CLI flag).
- **Every reply ≤ 600 s.** Run planner/LLM calls in a worker thread so websocket pings and the event loop stay responsive (HEAD client `ping_timeout=300`).
- **Reset handling**: `{"reset": True}` arrives once per rollout with no ack. Clear the plan, memory and executor buffers there, and also on a new connection. The same port is kept for the whole rollout, so history-dependent state is fine (`updates.md:57`).
- **Action hygiene**: the R1Pro arms and torso use absolute joint targets, so never emit zeros as "idle". The base is robot-frame velocity normalized to [-1,1]. Grippers take -1..1 (closed..open [inferred]). Output float32 `(N,23)`.
- **Perception inputs**: three RGBA cameras and, if RGBD, depth in meters (clip to [0.01,10] to match training). `cam_rel_poses` gives camera-to-base extrinsics; the intrinsics are the constants in `eval_utils.py:80-90`, scaled for 224×224 if `DefaultWrapper` is used [inferred]. There is no global pose, so navigation must rely on visual odometry/SLAM or integrating `base_qvel` (robot frame, `proprio[0:3]`) at dt = 1/30 s.
- **Scoring**: partial credit counts only predicates that become satisfied and are not undone. A planner should order subgoals to lock in individual predicates and avoid destroying satisfied ones. Success ends the episode immediately, and being faster than humans improves the time tie-breaker (`score_utils.py:321-324`).
- **Robot start pose** varies per instance (TRO `robot_poses[0]`), and only the task rooms are loaded (`partial_scene_load=True`), which bounds exploration.
- **Compatibility**: if we reuse OpenPI's B1K wrapper as the executor, fix the robot-name key mismatch (`robot` vs `robot_r1`) before running against HEAD.

---

## 11. Open questions
- The R1Pro `RobotDefinition` YAML (`raw_controller_order`, joint limits, gripper limits, camera prims) is not in the repo. The 23-D layout comes from `ACTION_QPOS_INDICES` and OPENPI, not from the definition file.
- `task.jsonl` (human mean lengths, hence exact `max_steps`), `B100_task_misc.csv` (task_id ↔ name, rooms) and `available_tasks.yaml` are NOT FOUND locally; they live under `DATA_PATH/2026-challenge-task-instances/metadata/`.
- Will the final hidden evaluation run HEAD (`v3.9.3-post1`/`-post2`) or the `2026/eval` branch with time budgets? Single-port group budget: `max_steps` (2026-eval code) or `N × max_steps` (group-time-budget docs)?
- Will the organizers run `RGBDFullResWrapper` (docs say official) and enable `--replay-action-chunk-size` for submissions?
- The exact websockets sync-client keepalive behavior with `ping_timeout=300` in the installed version (≥15.0.1) was not verified (no install here).
- Depth: Isaac `distance_to_image_plane` for empty pixels may be inf or a clip value; this was not verified.
- Whether `NullJointController` for `camera` adds any action dims for R1Pro. The 23-D layout implies none.
