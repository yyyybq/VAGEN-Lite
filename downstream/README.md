# downstream/ — 下游具身任务线（StarVLA / RLinf / RoboCasa365）

把 Active Spatial 训练出的视觉语言权重装进 StarVLA 做动作 SFT、用 RLinf 做 RL、在 RoboCasa365 上测评，比较不同起点。
状态文档：[docs/downstream/STATUS.md](../docs/downstream/STATUS.md)；报告：[docs/downstream/reports/](../docs/downstream/reports/)。

| 文件 | 作用 |
| --- | --- |
| `starts.json` / `starts_lock.json` | 起点权重登记（加一个起点 = 加一条）/ 锁定时的分片大小与索引、配置哈希 |
| `verify_starts.py` | 提交任务前校验起点没变（变了就停，不换路径） |
| `starvla_patch.py`、`patches/` | StarVLA 工作区全部改动的补丁、`git apply --reverse --check` 校验 |
| `sco_job.py` | 本线的 sco 任务：prepare / submit / status / delete（载荷写在 `--` 之后） |
| `demos/subsets.py` | 示教清单：target 50 任务全部 / 30% / 10%（RoboCasa 官方子集定义）、第二步的小清单 |
| `demos/prepare_data.py` | 清单 → StarVLA 训练目录（按示教建链接、可写的 meta、`split_manifest.json`） |
| `demos/target50_tasks.json` | target 50 任务的组别与时限（RoboCasa 1.0.1 注册表） |
| `train/recipe.yaml` | 所有实验共用的 StarVLA 配方 |
| `train/make_experiment.py` | 建一个实验目录（最终配置 + 起点、数据、代码、StarVLA 补丁等全部记录） |
| `train/train_entry.py` | 训练入口：bf16 加载、建模前设种子、种子传给数据采样、不重复存最后一步；`--dry-run` 只出种子证据 |
| `eval/closed_loop.py` | 闭环评测：并行仿真进程、种子传进环境、按任务时限 |
| `eval/seed_check.py` | 同种子重置得到相同初始状态的证据 |
| `step2/run_step2.sh` | 第二步的 8 卡任务脚本 |
| `load_test/`、`inventory/` | 第一步的加载测试与盘点脚本 |
| `tests/` | CPU 单元测试：`PYTHONPATH=<StarVLA> python -m pytest -q -p no:cacheprovider downstream/tests` |

约定：StarVLA、RLinf、渲染库覆盖层只按主检出 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/{third_party,playground}` 读，不改；
worktree 里的 `third_party`、`playground` 是指向它们的链接（两者都被 .gitignore 忽略），让现有脚本原样可用。
权重只按路径读；大文件写到 `/mnt/umm/users/yinbaiqiao/downstream_runs/`。
