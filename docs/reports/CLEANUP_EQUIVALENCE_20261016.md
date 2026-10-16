# 整理前后行为对照：第一批归档之后（2026-10-16）

**结论：第一批归档之后，留下来的代码行为和整理前一致。** 两套测试都没有新增失败。四组固定输出共 17 个文件，14 个逐字节相同，另外 3 个只差"这次运行花了多少秒"这类耗时字段。整理后少掉的 34 条测试，全部来自 9 个测试文件，它们随被测代码一起移进了 `archive/tests/`。

第一批归档指提交 `52eb7944`：807 个没有被任何保留代码引用的文件移进 `archive/`，只移动、不改逻辑。

## 怎么比的

两边代码只差整理改动：

- **整理前**：main `8134f102`（第十一轮最新提交）的干净检出，用 `git worktree` 放在集群 `/tmp/psv/base_wt`。
- **整理后**：分支 `cleanup/20261016` 合并 main 之后的 `8fe3d0ee`。

两边用同一个 Python 环境（vagen-lite）、只用 CPU、环境变量相同。`verl`、`third_party/`、`playground/` 三处在两边都指向主检出的同一份：`verl` 是子模块，另外两个不在 git 里。

上午的第一次基准作废了。那次是在主检出里跑的，当时另一个会话在那里有未提交的第十一轮改动，评测表因此多出一列"至少一条成功（采样）"，两边代码并不相同。这一次改用上面的干净检出重做。

术语：

- **套件 A**：仓库的全部测试，即 `pytest.ini` 里列出的测试目录。
- **套件 B**：`scripts/check_active_spatial_cpu.sh` 列出的环境核心测试。
- **固定输出**：G1–G4 四步小规模真实运行的产物，由 `scripts/cluster/golden_outputs.sh` 生成。G1 用第十轮生成配置生成 2 个场景的题；G2 在环境里每个题族回放 5 道题；G3 做场景划分和 SFT 帧列表；G4 在第八轮评测结果的副本上重算评测表。

## 结果

| 检查 | 整理前 | 整理后 | 差别从哪来 |
| --- | --- | --- | --- |
| 套件 A | 400 通过，3 跳过，共 403 | 367 通过，2 跳过，共 369 | 少的 34 条都在 9 个归档测试文件里：33 条原本通过，1 条原本跳过 |
| 套件 B | 76 通过，1 跳过，共 77 | 72 通过，共 72 | 少的 5 条是清单里去掉的 2 个归档测试文件：4 条通过，1 条跳过 |
| 固定输出 G1–G4 | 17 个文件 | 17 个文件 | 14 个逐字节相同，3 个只差耗时字段 |

两边都跳过的 2 条测试，原因都在运行环境，和代码无关：

- `test_task_context.py` 里调用真实 Qwen 处理器的一条，需要环境变量 `R1_TEST_MODEL_PATH`。
- `test_starvla_transfer.py` 里的一条，需要 `playground/` 下的 RoboCasa 元数据。把 `playground` 链进两个检出后，这个文件的 10 条在两边都通过。

只差耗时的 3 个文件：

- `gen/per_scene.csv` 的 `seconds` 列，即每个场景的生成耗时。
- `gen/scenes/0001_839920.json` 和 `gen/scenes/0002_839955.json` 里的 `timings` 和 `summary.seconds`。

去掉这些字段后，3 个文件逐字段相同。判断由 `scripts/cluster/compare_golden.py` 完成。

G4 重算的表和第八轮存档的 `table.md` 不一样，整理前后都是如此。原因是存档表由旧代码写成，第十一轮给表加了列。这里只比较整理前后两次新算出的表，二者逐字节相同。

## 归档的 9 个测试文件

都在 `archive/tests/`。它们测的代码在第一批里已经移进 `archive/`，留在主测试里只会因为找不到被测文件而失败。

| 文件 | 条数 |
| --- | --- |
| `test_dense_score_preflight.py` | 2 |
| `test_dense_score_validation_analysis.py` | 3 |
| `test_gate_aligned_shaping.py` | 4 |
| `test_r1_legacy_configs.py` | 7 |
| `test_r5_production_backward.py` | 4 |
| `test_renderer_microbenchmark.py` | 3 |
| `test_task_context_pairing.py` | 6 |
| `test_r1_clean_projective_v0.py` | 2，其中 1 条需要控制机数据，原本就跳过 |
| `test_r1_projective_projection_frontier.py` | 3 |

## 中间那次复跑的 1 条失败

第一批刚完成时在分支检出里复跑过一次，`test_starvla_transfer.py` 有 1 条失败，报错是找不到 `third_party/starVLA`。原因是分支检出里没有 `third_party/`，它不在 git 里。链上之后通过，不是代码问题。

## 复现

```bash
ROOT=/tmp/psv/base_wt GOUT=/tmp/psv/base_wt/exps/_cleanup_golden_base bash scripts/cluster/golden_outputs.sh
ROOT=$PWD GOUT=$PWD/exps/_cleanup_golden_after2 bash scripts/cluster/golden_outputs.sh
python scripts/cluster/compare_golden.py /tmp/psv/base_wt/exps/_cleanup_golden_base exps/_cleanup_golden_after2
```

测试日志和两边的固定输出存在集群 `exps/_cleanup_equivalence_20261016/`（主检出下）。后面每一批代码搬动之后都按同样方法再比一次。
