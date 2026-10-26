# configs/：统一配置

整个仓库用到的集群路径和 sco 设置集中在这里。代码通过 `vagen.config` 读取，不再各自写一份。

| 文件 | 内容 | 谁写 |
| --- | --- | --- |
| `paths.yaml` | 共享盘上的路径：集群检出、conda 环境、场景数据、AOSS 配置、gsplat 编译缓存、Hugging Face 缓存。分 `cluster` 和 `local` 两套 | 手写 |
| `cluster.yaml` | sco 任务提交：命令行位置、工作空间、镜像、存储挂载、资源池、卡数对应的规格 | 手写 |
| `datasets.yaml`、`models.yaml`、`results.yaml` | 集群固定位置 `data/releases/`、`models/`、`results/` 下每一项的清单摘要：路径、大小、代码提交、说明 | `scripts/cluster/collect_manifests.py` 生成，不要手改 |
| `removed_entries.yaml` | 从固定位置撤下的登记项：数据已不在集群上，记下是什么、何时、为什么 | 手写 |

## 怎么用

```python
from vagen import config
config.path("scenes_meta")      # 当前这套路径里的场景元数据目录
config.cluster("sco.image")     # sco 任务的镜像
```

shell 脚本里：

```bash
python -m vagen.config get paths.scenes_meta
```

- **选哪套路径：** 环境变量 `VAGEN_PROFILE`，取 `cluster` 或 `local`，默认 `cluster`。
- **换一个配置目录：** 环境变量 `VAGEN_CONFIG_DIR`。
- **本机：** `local` 这套目前都是空值，读到空值会报错并说明是哪个键。要在本机跑，把用到的键填上。

## 哪些地方已经读配置，哪些还留着自己的一份

- **读配置：** 四个提交 sco 任务的模块，即 `data_gen/interiorgs_v10/` 下的 `job_sco.py`、`pilot_sco.py`、`shard_sco.py`、`sco_interiorgs_v10.py`。它们原来的常量名不变，另一会话的 `data_gen/interiorgs_qa/sco.py` 照常导入。
- **留着自己的一份，由测试核对：** 计算节点上的作业脚本 `examples/train/active_spatial/sco_interiorgs_v10_train.sh`，它在任何 Python 之前运行；`sco_run_as_artifact_owner.sh`；在计算节点上从打包源码运行的模块的命令行默认值，包括 `evaluation.py`、`depth_check.py`、`fetch_interiorgs_metadata.py` 和 `gs_io.py`；EASI 注册表 `scripts/easi_registry.yaml` 的三个根目录。`tests/test_config.py` 检查这些副本和配置一致，改配置时测试会指出还要改哪里。
- **不动：** 数据登记和实验输入，例如 `data_gen/interiorgs_v10/configs/matrix_r12.json`、PPO 模板里会被 `rl-config` 覆盖的占位路径。另一会话负责的 `data_gen/interiorgs_qa/`、`data_gen/active_spatial_qa/`（决定 D15）也不动。

## 改配置之前

`tests/test_config.py` 钉住了集群这套的当前取值，它们就是整理之前代码里写的值。改值会改变提交出去的任务，所以要同时改测试，并在提交说明里写清楚原因。
