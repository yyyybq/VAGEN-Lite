# InteriorGS 空间问答：进度

本文件是这项工作的交接记录。每完成一个部分就更新，并提交、推送到 GitHub main。
任务说明见用户 2026-10-14 的提示词（构造式出题、单图/多图/视频一次写完、试验 F/G 与颜色试做、样题审核网页）。

## 1. 做完了什么

- 2026-10-14 20:30 UTC，接手核对：dfd4bf54 之后的 7 个提交都是 v10 训练第 6、7 轮，与本任务无关；
  `data_gen/interiorgs_qa/` 不存在；/tmp 下 `interiorgs_qa_*` 都是调研中间产物；没有本任务的 GPU 任务或后台进程。
- 渲染器能输出深度图（`gs_render_local.render_depth_from_cam_param`，gsplat "RGB+ED"）。GPU 冒烟测试
  （SCO pt-rfo2h32y）：512×512 每帧约 7 ms；渲染深度与三维框推算的深度中位差 5–14 cm。
- 原题格式样本已抓到服务器 `exps/interiorgs_qa_refs/`（BLINK、3DSRBench、ViewSpatial、VSI、MMSI、MindCube 题面与官方评分代码），
  不进 Git。
- 试验 F 第 1 步（CPU，`trial_f_axes.py` / `trial_f_walls.py`）：三维框角点顺序带着物体自身坐标系（p0 只有 33% 是最小角点，
  同类物体局部轴四个方向都有）；靠墙物体的正面几乎总是 −e1 = p0−p1（床 190/197、电视 136/140、马桶 108/109），衣柜、椅子、书桌不一致。
- 代码 `data_gen/interiorgs_qa/`：geom（带旋转的三维框）、scene_ctx（导入 v10 的场景、合法位置、face_visible_v3、v10_r5 指代）、
  visible（深度核验的可见部分）、views（构造式摆相机）、render、qa（措辞族、token 计数）、single（S1–S7）、multi（M1–M11）、
  video（V1–V6）、pipeline、trials（F/G/颜色）、scoring（各 benchmark 官方评分）、site（审核网页）、sco（GPU 提交）。
  CPU 代理渲染和 GPU 小测试（pt-4aw9hv8s，2 场景 43 题）都跑通。

## 2. 正在运行什么（2026-10-14 22:00 UTC 前后提交）

| 任务 | 编号 | 输出（日志在 `exps/interiorgs_qa_jobs/<名字>/logs/`） |
| --- | --- | --- |
| 样题：单图 S1–S7（已完成，70 题） | SCO pt-6gz9gsyc | `exps/interiorgs_qa_sample_20261014/single/` |
| 样题：多图 M1–M11（已完成，57 题） | SCO pt-8g4stofb | `exps/interiorgs_qa_sample_20261014/multi/` |
| 样题：视频 V1–V6（已完成，48 题） | SCO pt-bksss0if | `exps/interiorgs_qa_sample_20261014/video/` |
| 补跑多图 M3、M6、M8、M10、M11（150 个 train 场景） | SCO pt-otbvb8v8 | `.../topup_multi_a/`，任务名 topup_multi_a |
| 补跑多图 M9 | SCO pt-n9w1ph48 | `.../topup_multi_b/`，任务名 topup_multi_b |
| 补跑视频 V5、V6（130 个额外 train 场景） | SCO pt-nhftw7pi | `.../topup_video/`，任务名 topup_video |
| 统计扩量运行（20 场景，每场景每子任务最多 3 题） | SCO pt-rvwa1d37 / pt-2pgfqkzy / pt-4yhjqh4y | `exps/interiorgs_qa_stats_20261014/{single,multi,video}/`，任务名 stats_* |
| 试验 F（重跑，10 类各 5 例，已完成） | SCO pt-0tmvbqzs | `exps/interiorgs_qa_trials/run2_f/` |
| 试验 G、颜色试做（已完成） | SCO pt-lfd9gh26 | `exps/interiorgs_qa_trials/run1/`（其中试验 F 已被 run2_f 取代） |
| CPU 普查（按场景×组拆成 480 个子任务，40 进程） | fujian 后台进程 | `exps/interiorgs_qa_census_20261014/census.json`（结束时才写） |
| 指代普查（已完成） | — | `exps/interiorgs_qa_census_20261014/naming_census.json` |

查看 GPU 任务状态：`PYTHONPATH=. python -m data_gen.interiorgs_qa.sco status --job exps/interiorgs_qa_jobs/<名字>`。

## 3. 下一步

1. 收结果：凑不够配额的组用 `configs/extra_train_scenes_30.txt` 补跑；看图抽查。
2. 生成审核网页（`site.py`），rsync 到本机。
3. 改写 COVERAGE.md 的可行性判断和白话摘要；写报告（组合数、通过率、全量估算、不看图对照、预算、标注时间）。
4. 提交、推送，停下等用户标注。

## 4. 约定

- 代码：`data_gen/interiorgs_qa/`（进 Git）。文档：`docs/interiorgs_qa/`（进 Git）。
- 运行产物（图片、深度、网页）：`exps/interiorgs_qa_*`（git 忽略）。
- 同一目录里另一条工作线（v10 训练）也在提交；本任务只用 `git commit -- <自己的路径>` 提交自己的文件，不碰别人的未提交改动。
