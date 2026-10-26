# 交接：在 fujian 集群上用真实 InteriorGS 跑 v10 生成器

给本机 Claude Code 用（它能直接 `ssh fujian`）。用法：在本仓库根目录启动 Claude Code，说
“按 data_gen/interiorgs_v10/CLUSTER_RUN_HANDOFF.md 执行，每步把结果写进 cluster_runs/ 下的报告”。

## 背景
- 目标：1000 个 InteriorGS 场景中 ≥ 90% 能产出任务，总任务数 ≥ 10,000。v1 只有 90 个场景 / 100 条。
- v10 代码在 `data_gen/interiorgs_v10/`，设计见 `SPEC.md`、`README.md`，合成数据结果见 `REPORT_SYNTHETIC.md`。
- 集群：`ssh fujian`（Windows 本机需加 `-o ControlMaster=no -o ControlPath=none`，config 里的 ControlMaster
  在 Windows 上会报 "Failed to connect to new control master"）；仓库 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite`；
  数据 `/mnt/umm/users/yinbaiqiao/InteriorGS_meta`（2026-10-11 从 AOSS `fj:s3://baiqiao/InteriorGS` 拉的 1000 场景元数据；
  旧的 `/mnt/umm/users/yinbaiqiao/InteriorGS` 只有 100 个场景且属主 root、本用户不可读）；
  Python `/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python`；sco `/mnt/umm/users/yinbaiqiao/.sco/bin/sco`。

## 约束
- 只读 labels.json / structure.json（/ occupancy），不碰 PLY；不改 v1 产物和 InteriorGS 原始数据。
- 先看集群仓库 `git status`，不要覆盖别人的未提交改动；`interiorgs_v10/` 是新目录，直接拷进去即可。
- 登录节点只跑测试、审计和 ≤ 30 个场景的试跑，并限制进程数（`--workers 8`）。全量用 sco 提交 CPU 任务，提交前先问用户。

## 步骤
设 `R=/mnt/umm/users/yinbaiqiao/VAGEN-Lite`，`PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python`，
`D=/mnt/umm/users/yinbaiqiao/InteriorGS_meta`，`O=$R/exps/vagen_active_spatial/interiorgs_v10_real_$(date +%Y%m%d)`。

0. 同步代码（在本机执行）：
   `scp -r data_gen/interiorgs_v10 fujian:/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/`
   然后 `ssh fujian "find $R/data_gen/interiorgs_v10 -name __pycache__ -prune -exec rm -rf {} +"`（可选）。
1. 测试：`cd $R && $PY -m pytest tests`（2026-10 整理后测试都在 `tests/` 下；当时生成器的测试是 92 个）。
   集群是 Python 3.10–3.12，若有语法/numpy 版本问题先修。
2. 看一眼真实数据格式：`ls $D | head; ls $D | wc -l`；任取一个场景，打印 labels.json 前 2 个对象和 structure.json 的顶层键，
   确认与 `scene.py` 的假设一致（labels: ins_id/label/bounding_box 8×{x,y,z}，Z 向上；structure: rooms[].profile, holes[] type DOOR）。
3. 审计（不渲染，登录节点可跑，必要时 nohup）：
   `cd $R && $PY -m data_gen.interiorgs_v10.coordinates --scenes-root $D --out $O/calibration_report.json`
   `cd $R && $PY -m data_gen.interiorgs_v10.objects --scenes-root $D --out $O/object_stats.json`
   报告：坐标约定 frozen / v2-resolved / unusable 各多少场景（v1 是 117 个 ambiguous）；v10_relaxed 下 ≥2 个候选物体、≥1 个物体对的场景数。
4. 试跑 30 个场景：
   `cd $R && nohup $PY -m data_gen.interiorgs_v10.run_all --scenes-root $D --out $O/pilot --workers 8 --limit 30 --expected-scenes 30 > $O/pilot.log 2>&1 &`
   完成后 `$PY -m data_gen.interiorgs_v10.run_all --out $O/pilot --aggregate`，看 `summary.json`、`per_scene.csv`：
   每场景任务数、各任务族、d* 分布、每场景耗时、失败原因漏斗。若某个闸门在真实数据上异常（如遮挡、墙距、房间名），定位并修复，重跑试跑。
5. 全量：用户同意后
   `cd $R && $PY data_gen/interiorgs_v10/sco_interiorgs_v10.py prepare --tag <tag>`，再 `submit --tag <tag>`，`status --tag <tag>` 查看。
   完成后 aggregate，对照 ≥ 900 场景 / ≥ 10,000 条核对。

## 回报
把每步的关键数字和问题写到本机 `data_gen/interiorgs_v10/cluster_runs/REPORT_REAL_<日期>.md`，
包括：测试结果、真实数据格式差异、坐标约定统计、物体统计、试跑漏斗与耗时、所做修复（附 diff 摘要）、全量结果。
