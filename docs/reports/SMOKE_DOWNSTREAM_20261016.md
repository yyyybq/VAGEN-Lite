# 单卡冒烟：EASI、StarVLA、RLinf（2026-10-16）

目的：在定旁支模型去留（D4）和主线用 3B 还是 7B 之前，用单卡 sco 任务（zoetrope 池，1 张 H800）核查三条链路现在能不能跑、各自支持什么模型尺寸。每个任务几分钟；产物在集群 `exps/_cleanup_smoke_20261016/<名称>/`（`worker.log`、`exit_code`、`launch.sh`、`run.sh`）。

## 结论

| 链路 | 结果 | 3B | 7B | 发现的问题与处理 |
| --- | --- | --- | --- | --- |
| EASI（官方 lmms-eval 后端，EmbSpatial 8 题） | **通过**（任务 `pt-xrqsa16n`） | 通过，8 题 75.0 | 通过，8 题 75.0 | ① 官方 runner 用裸 `python -m lmms_eval`，sco 镜像默认 python 是 `/opt/conda` 的 3.9（无 numpy），第一次失败（`pt-f9j3tmf1`）；把 vagen-lite 环境放到 PATH 最前即可。② 计算节点没有外网，数据集要先在登录节点下载到 `~/.cache/huggingface`，任务里 `HF_HUB_OFFLINE=1`。两个模型推理都在一张卡上跑完。 |
| StarVLA 迁移（`prepare_starvla_transfer.py` → `train_starvla.py`，NavigateKitchen，2 步，batch 1） | **3B、7B 单卡都通过** | 通过（`pt-tv4rhwuw`，可训练参数 38.0 亿，训练 26 秒，任务全程 1 分 48 秒，写出 `steps_2_pytorch_model.pt`） | 通过（`pt-6wnv9e1u`，可训练参数 84.2 亿，每步约 3 秒，任务全程 2 分 38 秒，写出 `steps_2_pytorch_model.pt`，16 GB） | ① 仓库入口 `run_starvla_transfer.sh` 固定走 DeepSpeed ZeRO-2 + accelerate；单卡时 ZeRO-2 不分片，3B 全参数微调在 optimizer step 时 OOM（已用 71.7 GB，再要 14 GB；`pt-c8ukfb85`）。② 改成 9 月 30 日 smoke 用过的纯 torch 路径（`STARVLA_DISABLE_DEEPSPEED=1`，`attn_implementation=sdpa`）后 3B 单卡通过。③ `prepare_starvla_transfer.py` 一次只能准备同一尺寸的来源（"Controlled comparisons require the same backbone family and size"），3B 和 7B 要分开准备。④ 准备器会拒绝已存在的 `runs/<run_id>`，重跑要清掉。 |
| RLinf（RoboCasa365 NavigateKitchen，2 次更新，2 个环境） | **未通过。渲染已解决，卡在 RoboCasa 的版本**（第五次 `pt-65515vh8`，10-17 跑完）。④ 接上项目内渲染库覆盖层后，渲染这一关过了；新的报错是 `robocasa.utils.dataset_registry_utils` 不存在：从 `/var/tmp` 复制来的 `third_party/robocasa` 不带 RoboCasa365 的基准数据登记，`/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/robocasa` 那份有。⑤ 渲染诊断 `pt-a3vjvtyk`：节点上有 NVIDIA 的 EGL 驱动库，但厂商登记文件是空的，也没有 `/dev/dri`，NVIDIA EGL 用不了，只能走 OSMesa。⑥ 冒烟用的策略检查点（D9）已于 10-26 由磁盘线移到 AOSS，重跑要先恢复。 | — | — | ① RLinf 的 Python 环境 `third_party/RLinf/.venv` 原是指向登录节点本地盘 `/var/tmp/rlinf-openpi` 的符号链接，计算节点上不存在，前两次都报 "python: No such file or directory"。已把整个环境（19 GB）复制到共享盘 `third_party/RLinf/venv_mnt`，改掉其中 115 个脚本开头写死的旧路径，`.venv` 改指过去；RoboCasa 源码也从 `/var/tmp` 复制到 `third_party/robocasa`（7.3 GB），并改掉可编辑安装的路径映射。② 第三次报找不到 `git`：sco 镜像里没有 git，而 `prepare_rlinf_robocasa.py check` 要用它核对 RLinf 版本。冒烟时把登录节点的 git 放进 PATH。③ 第四次 Ray 和各工作进程都起来了，环境进程初始化时卡在模拟器渲染：镜像里既没有 EGL 调度库 `libEGL.so`，也没有 CPU 渲染库 `libOSMesa.so`。仓库里已有现成办法：`setup_render_runtime.sh` 做的项目内渲染库覆盖层 `playground/Runtime/egl/`，10 月 2 日的 sco 任务 `pt-f6ctnhzt` 用它加 OSMesa 跑通过 RoboCasa 闭环。第五次已接上这个覆盖层，同时派了一个渲染诊断任务 `pt-a3vjvtyk`，看节点上 NVIDIA 的 EGL 能不能用。 |

术语：ZeRO-2 是 DeepSpeed 把 optimizer 状态和梯度按卡切分的方式，单卡时不省显存；"纯 torch 路径"指不经 DeepSpeed、直接用 PyTorch 的 AdamW。

## 对 D4 与模型尺寸的建议

- **EASI**：推理只要一张卡，3B / 7B 都行；不限制尺寸选择。
- **StarVLA 动作 SFT**：3B 和 7B 在单卡纯 torch 路径上都能完成全参数训练的 2 步（batch 1）。两次都没有记录峰值显存，所以单卡能放多大的 batch 还不知道。正式训练用 8 卡 ZeRO-2 入口（`submit_starvla_nk_sco.sh`）。按显存粗估，7B 在 8 卡上可行：optimizer 状态约为参数量 × 16 字节，84 亿 × 16 字节 ÷ 8 卡 ≈ 17 GB/卡。
- **RLinf**：还没跑到训练那一步，尺寸限制未知。冒烟用的策略是 10 月 1 日训练的 StarVLA NavigateKitchen 检查点，底座是 Qwen3-VL-4B，不是 Qwen2.5-VL。要接主线的 3B 或 7B，需要先用那个尺寸做 StarVLA 迁移训练，再重新准备 RLinf 配置。
- 因此主线尺寸的约束主要来自 StarVLA / RLinf 的显存；EASI 不构成约束。

## 过程记录

| 任务 | 名称 | 结果 | 原因 |
| --- | --- | --- | --- |
| `pt-f9j3tmf1` | easi | 两模型都 FAIL | 子进程用了镜像自带 python 3.9（无 numpy） |
| `pt-xrqsa16n` | easi2 | 通过 | PATH 修正 |
| `pt-c8ukfb85` | starvla_3b | OOM | DeepSpeed ZeRO-2 单卡 |
| `pt-13kh2el1` | starvla_7b | OOM | 同上 |
| `pt-fsqblfdn`、`pt-udcgghvj` | starvla_{3b,7b}_single | 启动即退 | 上一次留下的 `runs/base` 目录（准备器拒绝覆盖） |
| `pt-tv4rhwuw` | starvla_3b_single2 | 通过 | 纯 torch 路径 |
| `pt-6wnv9e1u` | starvla_7b_single2 | 通过 | 纯 torch 路径 |
| `pt-ignjv4xu`、`pt-nhsxtxqj` | rlinf、rlinf2 | 退出码 127 | venv 在登录节点本地盘 |
| `pt-mabdf9b4` | rlinf3 | 退出码 1 | 镜像里没有 git |
| `pt-nfx51b5p` | rlinf4 | 退出码 1 | 镜像里没有 `libEGL.so`、`libOSMesa.so` |
| `pt-a3vjvtyk` | gl_diag | 跑完 | NVIDIA EGL 不可用：厂商登记文件为空，没有 `/dev/dri` |
| `pt-65515vh8` | rlinf5 | 退出码 255 | 渲染通过；复制来的 RoboCasa 缺 RoboCasa365 的数据登记模块 |

## 复现命令（整理后会进入 EVAL.md / DOWNSTREAM.md）

- 提交器：`exps/_cleanup_smoke_20261016/<名称>/launch.sh`（由 `smoke_job.sh` 生成：1 张 H800、`sco_run_as_artifact_owner.sh` 切回本用户后执行 `run.sh`）。
- EASI：`run.sh` 见 `easi2/`；关键是 `PATH=<vagen-lite>/bin:$PATH`、`HF_HUB_OFFLINE=1`、`--limit 8 --no-accelerate`。
- StarVLA：`run.sh` 见 `starvla_3b_single2/`；`STARVLA_DISABLE_DEEPSPEED=1`，直接 `python starVLA/training/train_starvla.py --config_yaml <prepared>/base.yaml --framework.qwenvl.attn_implementation sdpa`。
- RLinf：`run.sh` 见 `rlinf2/`；`prepare_rlinf_robocasa.py prepare --steps 2 --num-envs 2 --episode-steps 64` 生成 `train.yaml`，`CHECK_ONLY=1` 能展开 Hydra 配置。
