# downstream/ — 下游具身任务线（StarVLA / RLinf / RoboCasa365）

把 Active Spatial 训练出的视觉语言权重装进 StarVLA 做动作 SFT、用 RLinf 做 RL、在 RoboCasa365 上测评，比较不同起点。
状态文档：[docs/downstream/STATUS.md](../docs/downstream/STATUS.md)；报告：[docs/downstream/reports/](../docs/downstream/reports/)。

| 文件 | 作用 |
| --- | --- |
| `starts.json` | 起点权重登记（名字 → 路径、用途）。加一个起点 = 加一条 |
| `sco_job.py` | 本线的 sco 任务：prepare / submit / status / delete；任务目录 `RUN/job/` |
| `load_test/check_hf_export.py` | CPU：权重目录是不是完整 HF 格式（只读文件头）、精度、键名、"🔍"是否单 token |
| `load_test/load_in_starvla.py` | GPU：一个起点装进 StarVLA QwenOFT，核对加载、逐张量比对、假批次前向与推理 |
| `load_test/render_check.py` | RoboCasa365 厨房离屏渲染检查（一个进程一种后端） |
| `load_test/run_load_test.sh` | 第一步 1 卡任务的执行脚本 |
| `inventory/` | 第一步盘点用的小脚本（环境版本、target-50 任务表） |
| `tests/` | CPU 单元测试：`python -m pytest -q -p no:cacheprovider downstream/tests` |

约定：StarVLA、RLinf、渲染库覆盖层只按主检出 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite/{third_party,playground}` 的路径读，不改；
权重只按路径读；大文件写到 `/mnt/umm/users/yinbaiqiao/downstream_runs/`。
