# 冻结期代码搬动：合并 main、统一配置、移走旁支模型代码、重组测试（2026-10-26）

**结论：你要的四步都做完了。每一步之后，两套测试和金标准都和 main 等价。** 只有一处没按原计划做完：移走旁支模型代码这一步只移了 SenseNova-U1。Cambrian-S 和 Qwen3-VL 的代码没动，因为 10 月 24 日起的模型线正在用它们，这要你重新定。整理分支 `cleanup/20261016` 的代码停在 `9af4796c`（之后只有报告和 STATUS 的提交），已推送，还没有合并回 main。

术语：

- **套件 A**：仓库的全部测试。
- **套件 B**：`scripts/check_active_spatial_cpu.sh` 列出的环境核心测试。
- **金标准**：四步小规模真实运行的 17 个产物文件，见 `tests/regression/README.md`。
- **等价**：没有一条测试的结果改变；少掉的测试都来自随代码归档的测试文件；多出的测试都是这一步新加的、并且通过；金标准逐字节相同，或者只差耗时字段。

## 每一步做了什么

| 步骤 | 提交 | 内容 |
| --- | --- | --- |
| 合并 main | `cace74ff` | 合进 main 的 `00363333`，也就是第十二轮结束时的版本，没有冲突 |
| 第一批：统一配置 | `ba949450` | 新建 `configs/` 和 `vagen/config.py`，四个提交 sco 任务的模块改为读配置；登记上处理了磁盘线交来的三件事 |
| 第二批：移走 SenseNova-U1 | `acd957a1` | 5 个适配文件移入 `archive/side_models/sensenova_u1/`，删掉核心训练循环里只对它生效的分支 |
| 第三批：重组测试 | `b1ce072f`、`9af4796c` | 65 个测试文件分到 `tests/<领域>/`，核对工具移到 `tests/regression/` |

### 第一批：统一配置

- **新的配置文件：** `configs/paths.yaml` 存集群和本机两套路径，用 `VAGEN_PROFILE` 选；`configs/cluster.yaml` 存 sco 设置。代码通过 `vagen.config` 读取，shell 用 `python -m vagen.config get …`。
- **改为读配置的代码：** `data_gen/interiorgs_v10/` 下的 `job_sco.py`、`pilot_sco.py`、`shard_sco.py`、`sco_interiorgs_v10.py`。常量名不变，另一会话的 `interiorgs_qa/sco.py` 照常导入。在 main 和分支上分别导出这四个模块的 33 个提交常量，取值完全相同。
- **留着原文的地方：** 在任何 Python 之前运行的作业脚本，计算节点上从打包源码运行的模块的默认值，EASI 注册表的根目录。这些副本由 `tests/test_config.py` 核对，保证和配置一致。
- **登记表：** `configs/datasets.yaml`、`models.yaml`、`results.yaml` 由清单自动汇总；`configs/removed_entries.yaml` 是手写的撤下记录。
- **磁盘线交来的三件事：**
  - `models/` 下 8 个断链已撤下，即第六轮 b 的 SFT 和 7 个 RL 检查点。它们 10 月 25 日在主线清理时被删，删前没有上传 AOSS，撤下的原因记在 `removed_entries.yaml`。
  - `archive/cluster/` 的六个子目录已移到 AOSS，归档索引已注明。
  - D7b、D9 已由磁盘线做完，决定记录已注明。

### 第二批：移走 SenseNova-U1

- **移走的代码：** 删掉的都是只有 SenseNova-U1 检查点才会走到的分支：
  - `agent_loop_no_concat.py` 里 `neo_chat` 的处理器选择、多模态后处理分支和它专用的展开函数；
  - `gym_agent_loop_no_concat.py` 里的紧凑占位符分词，原来 `else` 里的两行原样保留；
  - Cambrian 插件里顺带注册的两个模型类；
  - `setup.py` 里的插件入口。
- **留下的空壳：** `vagen/models/sensenova_u1_plugin.py` 只剩一个什么都不注册的同名函数。原因是 vagen-lite 环境按旧 `setup.py` 安装，vLLM 在每个进程里都会导入这个插件。重装环境后可以删掉这个空壳。
- **没有动的代码：** Cambrian-S 和 Qwen3-VL 的适配代码，以及 `.venv-qwen3vl`，原因见文末第一个待决事项。

### 第三批：重组测试

| 目录 | 文件数 | 内容 |
| --- | --- | --- |
| `tests/envs/` | 11 | 环境、成功判定、可见性、奖励、提示词和解析 |
| `tests/datagen/` | 30 | 出题生成器、数据流水线和血缘 |
| `tests/training/` | 17 | SFT、RL、训练链、sco 提交 |
| `tests/evaluation/` | 4 | 评测、浏览页、失败分析、环境回放 |
| `tests/downstream/` | 2 | StarVLA 迁移、RoboCasa 动作 |
| `tests/regression/` | 1 个测试，加核对工具 | R1 回归、金标准比对 |

- **找路径的写法：** 原来按层数找仓库根目录的写法，改成向上找 `setup.py`。
- **不加 `__init__.py`：** `tests/` 下一个都不放。包名 `tests` 会和 verl 子模块里的 `tests` 包撞名，`tests/evaluation` 也可能盖过仓库自己的 `evaluation` 包。
- **测试之间的导入：** 几个互相导入辅助函数的测试放在同一个目录，按文件名导入。
- **布局测试：** 新加的 `tests/test_layout.py` 要求测试文件都在 `tests/<领域>/` 下、文件名不重复。别的分支以后加在旧位置的测试会在这里报错，而不会因为 `pytest.ini` 只收集 `tests/` 而被悄悄漏掉。
- **原位留下的测试：** 另一会话的 `data_gen/active_spatial_qa/` 测试，按决定 D15 不动；旧 SFT 生成器自带的 `test_path_finder.py`，以后随它一起归档。

## 核对结果

基准是 main `00363333` 的干净检出：套件 A 共 490 条，488 通过、2 跳过；套件 B 共 78 条，77 通过、1 跳过。

| 核对 | 分支提交 | 套件 A | 套件 B | 金标准 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 合并 main 后 | `cace74ff` | 456 条：455 通过、1 跳过 | 73 条全部通过 | 14 个相同，3 个只差耗时 | 等价 |
| 第一批之后 | `ba949450` | 472 条：471 通过、1 跳过 | 73 条全部通过 | 同上 | 等价 |
| 第二批之后 | `acd957a1` | 472 条：471 通过、1 跳过 | 73 条全部通过 | 同上 | 等价 |
| 第三批之后 | `b1ce072f` | 479 条：478 通过、1 跳过 | 73 条全部通过 | 同上 | 等价 |

数字怎么读：

- **比基准少的 34 条和 5 条：** 都来自 10 月 16 日随代码归档的 9 个测试文件。
- **第一批多出的 16 条：** 新加的配置测试，全部通过。
- **第三批多出的 7 条：** 4 条布局测试、3 条比对工具自己的测试，全部通过。
- **第三批挪动的测试：** 按"文件名、类名、用例名"逐条对齐，原有的 456 条全部对上，没有一条结果改变。

每一轮的测试记录、对比输出和金标准校验和存在集群 `exps/_cleanup_equivalence_20261026/`。

## 对其他在开分支的影响

我用一个临时检出真正试合并了 `models` 分支，其余分支做了试算合并。

| 分支 | 合并回来时会遇到什么 |
| --- | --- |
| `models`（模型线） | `job_sco.py` 有 2 处冲突，就是第一批改成读配置的常量和提交命令，改起来很直接。它对两个已改名测试的修改会自动并进 `tests/training/`。它新增的 3 个测试会落在已取消的 `tests/active_spatial/`，要挪进 `tests/<领域>/`，否则布局测试报错。 |
| `dense-batch2` | 没有冲突；同样有那 3 个新增测试要挪 |
| `transition-qa` | `training_prep.py` 有 1 处冲突，它和 main 之间本来就有，与这次整理无关；新增的 `data_gen/interiorgs_v10/tests/test_output_format.py` 要挪进 `tests/` |
| `capability-qa/20261016` | 没有冲突；新增的 7 个 `tests/test_capability_*.py` 要挪进某个领域目录 |
| `downstream`、`eval-external`、`disk` | 没有冲突 |

## 距离合并回 main 还差什么

1. **你的决定：Cambrian-S 和 Qwen3-VL 的代码留不留。** D4 选 a 时这两条是旁支，但模型线从 10 月 24 日起正在训练 Qwen3-VL-8B 和 Cambrian-S-7B，用的就是 `vagen/models/` 里的 Cambrian 适配代码和 Qwen3-VL 运行环境。我的建议是留下，把它们从"旁支"改为"模型线支持的骨干"，等模型线有结论后再定。
2. **合并本身。** 整理分支已包含 main 的全部提交，所以合并就是让 main 快进到整理分支。推送 main 会影响所有在开的线，要你点头。
3. **共享检出的更新。** 共享检出 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite` 停在本地 main `f8e43d40`，本身就比远端 main 落后 18 个提交。整理把很多文件换了位置，更新之前要先确认没有进程在从这个检出运行代码。10 月 26 日查过：
   - 模型线和密集奖励第二批的训练链，分别从 `VAGEN-Lite-models` 和 `VAGEN-Lite-r12dense` 两个副本运行，不受影响。
   - 从共享检出加载代码的，只有第十二轮的守护进程，和它每 5 分钟重启一次的看板 `batch_board`。更新共享检出最好在看板循环停下时做。
4. **合并之后，各线把分支合回来时：** 按上表处理冲突，把新加的测试挪进 `tests/<领域>/`。
5. **重装环境之后：** 重装 vagen-lite 环境（`pip install -e .`）以后，删掉 SenseNova-U1 的空壳。
6. **第一阶段方案里还没做的搬动，可以合并后另开一个窗口做：**
   - 把 `data_gen/interiorgs_v10` 的训练和评测代码拆到 `training/`、`evaluation/`；
   - D3 的旧评测栈和旧 EASI 分片脚本归档，同时撤掉 17 条兼容链接；
   - 旧 SFT 生成器和 `sapave` 归档。
7. **git 历史重写（D1）：** 放在最后做。
