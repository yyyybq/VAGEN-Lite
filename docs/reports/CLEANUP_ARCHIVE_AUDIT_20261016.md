# 归档目录的引用核查与两次事故（2026-10-16）

**结论：整理时移进 `archive/cluster/` 的目录里，有两个仍被在用的代码读取，我移动之前没有查出来。** 一个已由第十一轮的会话自己修好；另一个我已原样还原到原位置。随后做了全面核查，又给保留代码会读到的旧路径补了 17 条兼容链接。现在保留代码要读的已归档路径都能打开。

术语：

- **搬移清单**：`archive/cluster/MANIFEST_moves_20261016.tsv`，记录每个移走条目的原路径和新路径，共 508 条。
- **兼容链接**：在原路径放一个符号链接，指向它在 `archive/cluster/` 里的新位置。读旧路径的代码照常工作，数据仍放在归档里。
- **保留代码**：整理分支上留在原处、没有归档的代码。

## 事故 1：第十一轮训练脚本里的 gsplat 编译缓存

- `examples/train/active_spatial/sco_interiorgs_v10_train.sh` 里的 `VALIDATED_GSPLAT_CACHE` 指向 `exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/torch_extensions/gsplat_cuda`。这行从 10 月 13 日的 `71fc1062` 起就在。每个渲染服务启动时都会复制这份编译产物。
- 10 月 16 日我把这个目录移到了 `archive/cluster/dense_score_pilot/`。
- 第十一轮的会话发现后，把编译产物复制到仓库外的 `/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007`，并在 main 的 `d362ed50` 里改了脚本、加了测试。按它的提交说明，这个问题本会让所有 RL 和评测任务在启动时失败。第十一轮的日志里没有找到因此失败的任务。

## 事故 2：自有 QA 流水线读取第六轮生成目录

- `data_gen/interiorgs_qa/pipeline.py` 的模块常量 `SPLIT_MANIFEST` 指向 `exps/vagen_active_spatial/interiorgs_v10_gen_20261014/split_manifest.json`。
- 我先把这个目录移到 `archive/cluster/data_intermediate/`，后来又把它压成了 tar.gz。QA 流水线下次读这个文件就会失败。
- 处理：这是另一个会话负责的代码，按约定不改它，而是把目录原样解压回原位置。解压后是 4,432 个文件、343,812 KB，与搬移清单一致。确认 QA 模块能导入、能找到这个文件之后，才删掉多余的压缩包。

## 为什么会漏

第一阶段的引用图只查了代码文件之间的导入和文件名引用。移动集群目录之前，我没有拿每个要移的目录去搜写死在启动脚本和模块常量里的绝对路径。

## 全面核查

方法：搜索串共 869 个，包括 508 个移走条目的原路径，以及足够特别的目录名（至少 14 个字符且含数字）。在三处用定串匹配搜索：

1. main 最新提交里跟踪的文件，不含 `archive/` 和 `docs/reports/`；
2. 第十、十一轮实验目录里的文本文件，跳过检查点、渲染图和帧；
3. 主检出里其他会话未跟踪的文件。

脚本是 `scripts/cluster/audit_archived_paths.sh`，结果在登录节点 `/tmp/psv/cleanup_inv/audit/`。

| 命中类别 | 判断 | 处理 |
| --- | --- | --- |
| 在分支上已归档的旧脚本里 | 这些脚本本身已不在主流程里 | 不处理 |
| 在实验目录里 | 全部在各轮存下的代码快照里（`job/src*/`、`code/`），不是在跑的配置 | 不处理 |
| 只是写出，或只出现在注释、说明里 | 不读旧数据，例如诊断写出的文件名、`pilot_site.py` 的用法示例、恢复脚本的写入目标 | 不处理 |
| `scripts/easi_registry.yaml` 里 7 到 8 月的 6 次旧实验 | 这些实验在整理之前就没有可用的检查点：压缩包里的 `huggingface/` 目录要么是空的，要么根本没有。注册表里这几条早已失效 | 照样补了链接，行为与整理前相同 |
| 保留代码会读取的 | 见下节 | 补兼容链接 |

## 兼容链接

17 条链接，逐条记录在 `archive/cluster/COMPAT_LINKS_20261016.tsv`：

- `active_spatial_qa` 两个 sco 入口要读的 v46、v50 两次 8 月训练目录。v50 第 50、100、700 步的 HF 权重已按 D10 移到 `models/historical/`，在归档目录里原来的位置补了指过去的链接。
- 旧统一评测工具要读写的 `exps/unified_eval_runs/` 下的 `MASTER_RESULTS.md`、`easi_results`、`a1a2b5_easi_results`。
- EASI 脚本的默认输出目录 `easi_results/`，以及旧评测栈的 `evaluation/outputs/`、`evaluation/sweeps/`。
- EASI 注册表、R1 数据源表和旧评测套件引用的 6 次 7 到 8 月旧实验。

其中 11 个条目之前压缩过。先解压回归档位置，核对大小与压缩清单一致后才删压缩包。解压记录在 `archive/cluster/DECOMPRESSED_20261016.tsv`，第六轮生成目录的还原也记在这里。

撤除时机：代码搬动批次改写或归档读取这些路径的代码时一并撤除，包括 D3 的旧评测栈和旧 EASI 分片脚本。`active_spatial_qa` 的两个入口由负责 QA 的会话决定。

另外在主检出的本地排除文件 `.git/info/exclude` 里加了 4 条：`/archive/cluster/` 和三个输出目录链接。原因是主检出的 `.gitignore` 写法盖不住符号链接，而且排除 `archive/cluster/` 的那条只在整理分支上；不加的话，别的会话执行 `git add -A` 可能把归档数据提交进去。原文件备份为 `exclude.bak_20261016`。

## 以后怎么做

- 每次移动集群数据之前，先用搬移清单草稿跑一遍 `scripts/cluster/audit_archived_paths.sh`。有读取引用的条目先不移，或者先补兼容链接。核查结果和搬移清单存在一起。
- 不再压缩被保留代码读取的条目。
