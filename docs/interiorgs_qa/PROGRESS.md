# InteriorGS 空间问答：进度

本文件是这项工作的交接记录。每完成一个部分就更新，并提交、推送到 GitHub main。
任务说明见用户 2026-10-14 的提示词（构造式出题、单图/多图/视频一次写完、试验 F/G 与颜色试做、样题审核网页）。

## 1. 做完了什么

- 2026-10-14 20:30 UTC，接手核对：dfd4bf54 之后的 7 个提交都是 v10 训练第 6、7 轮，与本任务无关；
  `data_gen/interiorgs_qa/` 不存在；/tmp 下 `interiorgs_qa_*` 都是调研中间产物；没有本任务的 GPU 任务或后台进程。
- 渲染器能输出深度图（`gs_render_local.render_depth_from_cam_param`，gsplat "RGB+ED"）。GPU 冒烟测试
  （SCO pt-rfo2h32y）：512×512 每帧约 7 ms；渲染深度与三维框推算的深度中位差 5–14 cm。
- 原题格式样本已抓到服务器 `exps/interiorgs_qa_refs/`（不进 Git）。
- 试验 F 第 1 步（CPU）：三维框角点顺序带着物体自身坐标系；靠墙物体的正面几乎总是 −e1 = p0−p1。
- 出题代码 `data_gen/interiorgs_qa/`（单图 S1–S7、多图 M1–M11、视频 V1–V6、试验、打分、审核网页、GPU 提交）。
- 样题：单图 70、多图 110、视频 60 题（20 个 train 场景；稀少的组另用最多 300 个 train 场景补到每组 10 题）。
- CPU 构造式普查（20 场景×24 组）和指代普查完成：`exps/interiorgs_qa_census_20261014/`。
- 统计扩量运行完成（20 场景，每场景每子任务最多 3 题）：`exps/interiorgs_qa_stats_20261014/`。
- 试验 F（10 类×5 例）、试验 G（20 个物体）、颜色试做（10 题）渲染完成：`exps/interiorgs_qa_trials/merged/`。
- COVERAGE.md 第二版（逐项重判 + 白话摘要 + 4.0 节逐项改动表；92 个子任务编号全部落到组、试验或不做清单，无遗漏）。
- 盲答基线发现 5 处由出题顺序造成的答案偏差，已改代码并重跑：M6 多步题"右前"占 10/11（连拍总向右转、角色固定）；
  S1 的 BLINK"左边吗"全答"是"；S2 上下题、S4"碰到吗"多答"否"；M1 的 BLINK 多视角题"向右"占 85–89%（转动方向与起始方位按固定顺序尝试）。
- M6、M11 重跑完成（各 10 题）；是非题与 BLINK 多视角题的答案平衡重跑完成。
- 报告 [`REPORT_20261014.md`](REPORT_20261014.md) 完成（产量、盲答基线、预算方案、试验说明、需要拍板的 10 件事）；审核网页同步到本机。

## 2. 正在运行什么

没有。最后一批 GPU 任务（2026-10-14 23:35 UTC 前完成）：

| 任务 | 编号 | 输出 |
| --- | --- | --- |
| 重跑 M6、M11（修正转身方向与角色） | SCO pt-mo70i9gp | `exps/interiorgs_qa_sample_20261014/topup_pan2/` |
| 重跑 S1/S2/S4 的是非题、M1 的 BLINK 多视角题样题（答案平衡最终版） | SCO pt-ymlxym1a | `exps/interiorgs_qa_sample_20261014/rebalance4/` |
| 同上，统计扩量（4 个任务各 5 个场景） | SCO pt-wo8h9pgr / pt-8zwxbpb4 / pt-nhjfta7a / pt-i5ejp6mc | `exps/interiorgs_qa_stats_20261014/rebalance4_{a,b,c,d}/` |

中间版本 rebalance、rebalance2、rebalance3_* 不再使用。被取代的旧题清单在 `docs/interiorgs_qa/audit/report_numbers_20261014.sh` 和 `site_20261014.sh` 的 EXCL 变量里。

## 3. 下一步

1. **等用户标注。** 审核网页在本机 `C:\Users\user\Downloads\_ICLR27_Probing_Spatial_World_Representation\interiorgs_qa_review_site\index.html`（服务器 `exps/interiorgs_qa_site_20261014/`）。用户交回导出的 JSON 后，拷到服务器 `exps/interiorgs_qa_review_labels_20261014/`。
2. 按标注结果和报告第九节的决定：校准 Around 遮挡阈值，定试验 F/G/颜色出不出题，清理出错的类别，然后做全量。

## 4. 约定

- 代码：`data_gen/interiorgs_qa/`（进 Git）。文档：`docs/interiorgs_qa/`（进 Git）。
- 运行产物（图片、深度、网页）：`exps/interiorgs_qa_*`（git 忽略）。
- 同一目录里另一条工作线（v10 训练）也在提交；本任务只用 `git commit -- <自己的路径>` 提交自己的文件，不碰别人的未提交改动。
