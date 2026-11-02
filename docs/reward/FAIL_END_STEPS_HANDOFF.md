# 失败局终点的真实剩余步数：交给主线的说明（2026-11-02）

按用户 11-02 的安排，这个工具由主线集成进 matrix-v3，作为评测的标准指标。本线只交文件和说明，不改主线代码。

## 它算什么

- 对一份评测结果里每个失败的局：在评测用的环境里不渲染地重放，一个记录的回答走一次 env.step。
- 然后用这道题的距离表，查最后位姿离"做成"最少还差几个动作，记作终点 d。
- d 越小，说明失败的局停得越近。只看成功率时，这部分差别是看不出来的。

每个失败局的终点归为四类：

| 类别 | 含义 |
| --- | --- |
| d | 位姿的格子在表里，并且能走到成功；d 就是剩余最少步数 |
| 相邻格 | 格子不在表里（表按 1 cm / 1° 合并格子），26 个相邻格里有能到成功的，取其中最小的 d |
| 不可达 | 格子在表里，但在表的图里走不到成功 |
| 查不到 | 格子和相邻格都不在表里 |

- 从起点算的深度加上 d 不超过表的深度（12）时，d 是精确值；超过时是上界。输出里的 exact 列记了这一点。
- 每局还会逐回合把重放和记录对一遍：这一回合后是否做成、被挡的动作数。对上的计入"重放一致"。

## 要拿的三个文件（分支 reward-design）

| 文件 | 作用 |
| --- | --- |
| `reward/__init__.py` | 空文件，让 `reward` 成为一个包 |
| `reward/fail_end_steps.py` | 工具本身 |
| `data_gen/interiorgs_v10/distance_table_near.py` | 相邻格兜底（只新增，不改 distance_table.py） |

- 工具还依赖 `data_gen/interiorgs_v10/distance_table.py`、`difficulty.py` 和 `vagen/envs/active_spatial/`。这三处 qwen3-main 上都有，接口一样。
- 不需要本线的其他代码，也不需要环境里的开关。

## 用法

```bash
python -m reward.fail_end_steps --result DIR [--result DIR ...] --tables TABLES --out OUT.json [--workers 4]
```

- DIR 是一份评测结果的文件夹，里面要有 results.jsonl、model.json、done.json。
  - 题目文件默认取 DIR/../tasks.jsonl，也可以用 --tasks 指定。
  - 环境配置用 rl_env_v10_S0.yaml 做模板，再套上 model.json 里的提示格式和环境开关。
- 输出两个文件：
  - OUT.json：每份结果的汇总，加上每个失败局的明细；
  - OUT.md：一张表，列是失败局数、重放一致数、终点 d 的平均和中位、这些题的 d\* 平均、d ≤ 1 和 d ≤ 2 的局数、相邻格、不可达、查不到。
- 有失败的题没有表时，工具会停下，列出缺哪些题。加 --allow-missing 可以跳过这些题、只算有表的。
- 在登录节点上最多用 4 个进程（用户 10-28 的规则）。

## 已有的表

| 评测集 | 表的目录 | 题数 |
| --- | --- | --- |
| eval_val_v2 + eval_test_v2 | `reward_runs/tables/eval705` | 705 |
| eval_test_v2_screened（P1 用的筛选后测试集） | `reward_runs/tables/eval_test_v2_screened` | 292 |

- eval705 是只读的。AOSS 上的副本在 `fj:s3://baiqiao/vagen_lite_archive/reward_runs/tables/eval705`。
- eval_test_v2_screened 这个目录是一个视图：
  - 233 道和 eval_test_v2 的题逐字段完全相同，直接链接到 eval705 的表；
  - 另外 59 道是新题，表是 11-02 用一个单卡 zoetrope 任务新建的，放在 `reward_runs/tables/screened59`。
  - screened59 的 AOSS 副本在 `fj:s3://baiqiao/vagen_lite_archive/reward_runs/tables/screened59`（62 个文件，大小逐个核过；清单 `aoss_archive_manifests/manifest_20261102T220934Z.jsonl`）。
- 上面路径的根目录都是 `/mnt/umm/users/yinbaiqiao/`。

## 新的评测集怎么建表

```bash
python -m reward.distance12 build --rows SET/tasks.jsonl --task-ids IDS.txt --out TABLES --gs-root GS \
    --depth 12 --prefilter strict --workers N
```

- 平均每道题约 0.38 CPU 小时。705 道共约 270 CPU 小时：一个 8 卡任务约 2.5 小时，单卡约 22 小时。
  - 筛选集的 59 道在一个单卡 zoetrope 任务上实测 12.7 CPU 小时、1.6 小时，每道约 0.22 CPU 小时。
- 这是造数据，要用 sco 提交，不在登录节点上跑。本线的单卡提交工具是 `reward/sco_job.py`，用法写在文件开头。
- `reward/distance12.py` 不在上面三个文件里。主线要自己建表，就再拿这一个文件，或者让本线来建。

## 核对过的

- 在 B 的 rw_s0 第 150 次的 379 个失败局上，加 --no-near 后和 B 的 fail_end_d 逐局完全一样（改成按题号定位之后又核了一次）。
- 在筛选集上，用两份已有的结果端到端跑过（`VAGEN-Lite-data/exps/matrix_inputs_20261029/eval_test_v2_screened/` 下的 c0q3_s1s0_150 和 c0q3_sft_e8），4 个进程共用 22 秒：

  | 结果 | 失败局 | 重放一致 | 出错 | 终点 d 平均（中位） | 这些题的 d\* 平均 | 相邻格给出 | 不可达 / 查不到 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | c0q3_s1s0_150 | 168 | 168 | 0 | 5.69（4.5） | 5.51 | 1 | 0 / 0 |
  | c0q3_sft_e8 | 161 | 161 | 0 | 6.39（6.0） | 5.34 | 4 | 0 / 0 |

- 这两份结果是从两次评测拼起来的，其中 59 道新题的 index 指向另一份题目文件。所以工具改成按 task_id 在题目文件里找行号来 reset，不再用 index（commit `7f4407b6`）。没改之前这 59 道题会 reset 到别的题上。
- 这两个文件夹没有 done.json，工具会拒绝没有 done.json 的结果。核对时我在临时目录里拷了一份，补了一个空的 done.json，原文件没动。主线自己的评测结果都有 done.json。
