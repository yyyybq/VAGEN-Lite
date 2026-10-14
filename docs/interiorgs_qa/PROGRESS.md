# InteriorGS 空间问答：进度

本文件是这项工作的交接记录。每完成一个部分就更新，并提交、推送到 GitHub main。
任务说明见用户 2026-10-14 的提示词（构造式出题、单图/多图/视频一次写完、试验 F/G 与颜色试做、样题审核网页）。

## 1. 做完了什么

- 2026-10-14 20:30 UTC，接手核对：
  - dfd4bf54（调研提交）之后的 7 个提交都是 v10 训练第 6、7 轮，与本任务无关；
  - `data_gen/interiorgs_qa/` 尚不存在；/tmp 下 `interiorgs_qa_*` 都是调研的中间产物（早于 dfd4bf54）；
  - 没有本任务的 GPU 任务或后台进程。之前的会话只完成了调研。
- 渲染器能输出深度图：`vagen/envs/active_spatial/render/gs_render_local.py` 的 `render_depth_from_cam_param`
  （gsplat "RGB+ED"：RGB、沿视轴的米制深度、累计不透明度），v10 的 `depth_check.py` 已用过。

## 2. 正在运行什么

- 无。

## 3. 下一步

1. 抓取各 benchmark 的原题格式样本（只取题目文字，不下图），定模板。
2. 试验 F 第 1 步（CPU）：三维框角点顺序是否带着物体自身坐标系。
3. 写共用部件：场景与三维框、相机、深度核验的可见部分、指代、模板、打分、token 计数；GPU 冒烟测试。
4. 各组出题代码（单图 S1–S7、多图 M1–M11、视频 V1–V6），在 20 个 train 场景上出样题、渲染、核验。
5. 试验 F、G、颜色试做；审核网页；改写 COVERAGE.md；报告。

## 4. 约定

- 代码：`data_gen/interiorgs_qa/`（进 Git）。文档：`docs/interiorgs_qa/`（进 Git）。
- 运行产物（图片、深度、网页）：`exps/interiorgs_qa_*`（git 忽略）。
- 同一目录里另一条工作线（v10 训练）也在提交；本任务只用 `git commit -- <自己的路径>` 提交自己的文件，不碰别人的未提交改动。
