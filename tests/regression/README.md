# tests/regression/：回归与金标准比对

这里放两类东西：

- **R1 回归测试** `test_r1_regression.py`：冻结的 R1 任务在现在的生成器里仍按原样出题。
- **金标准比对工具**：整理期间每一批改动之后，用它确认行为和改动前一样。

## 金标准比对

"金标准"指四步小规模的真实运行产物，在登录节点用 CPU 跑，约 15 分钟：

| 步骤 | 内容 |
| --- | --- |
| G1 | 用第十轮生成配置给 2 个场景出题 |
| G2 | 在环境里每个题族回放 5 道题 |
| G3 | 场景划分和 SFT 帧列表 |
| G4 | 在第八轮评测结果的副本上重算评测表 |

比对规则：17 个产物文件逐字节比较；不同的文件去掉耗时字段后再比，即 JSON 的 `timings`、`seconds` 和 CSV 的 `seconds` 列。

| 文件 | 用途 |
| --- | --- |
| `golden_outputs.sh` | 生成一份金标准：`ROOT=<检出> GOUT=<新目录> bash tests/regression/golden_outputs.sh` |
| `compare_golden.py` | 比较两份金标准；只差耗时的算通过 |
| `check_round.sh` | 跑一轮完整核对：两套测试，以及金标准。结果写进 `OUT/` |
| `compare_round.py` | 比较两轮：按"测试文件名、类名、用例名"逐条对齐，所以只挪了目录的测试仍能对上；再比较金标准 |
| `archived_tests.txt` | 随代码归档、所以整理后不再收集的测试文件名，供 `compare_round.py --moved-ok` 使用 |
| `test_compare_golden.py` | `compare_golden.py` 自己的单元测试 |

一轮核对的做法：

```bash
git -C /mnt/umm/users/yinbaiqiao/VAGEN-Lite worktree add --detach /tmp/psv/base origin/main
ROOT=/tmp/psv/base OUT=/tmp/psv/rounds/base bash tests/regression/check_round.sh
ROOT=$PWD OUT=/tmp/psv/rounds/after bash tests/regression/check_round.sh
python tests/regression/compare_round.py /tmp/psv/rounds/base /tmp/psv/rounds/after --moved-ok tests/regression/archived_tests.txt
```

基准要用 main 的干净检出，不能用共享检出：别的会话常在那里留有未提交的改动。
