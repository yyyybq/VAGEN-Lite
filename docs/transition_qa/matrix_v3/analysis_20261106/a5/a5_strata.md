# 分析 5：错选分层（探索性）

* Same-category objects in view: for each target, the target itself plus its same-name objects (the generator's   ref_confusers) that are identifiable from the start pose (occlusion.identifiable under the env's face rule, the   generator's start-frame rule); a task takes its largest count over its targets: 0 (none in view), 1, 2, 3+. * Target visible area: the smallest start area_ratio over the task's targets (initial_observability, written by the   generator; a target not identifiable counts as not visible): not visible / just past the threshold (< 0.01; the face   rule's minimum is 0.003) / medium (0.01-0.05) / large (>= 0.05).

成功率是各组 RL 合起来的"题 × RL"（B′ 是一个种子的两个 RL）。

## 按视野内同类物体个数

| 分层 | 题数 | C | Cr | D | D_noP0 | B' |
| --- | --- | --- | --- | --- | --- | --- |
| 0（都不在视野里） | 2 | 25.0% | 50.0% | 50.0% | 0.0% | 50.0% |
| 1 | 290 | 45.2% | 50.4% | 49.6% | 50.3% | 44.3% |

## 按场景里同类物体个数（补充：在不在视野里都算）

| 分层 | 题数 | C | Cr | D | D_noP0 | B' |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 182 | 41.6% | 47.0% | 47.3% | 48.2% | 40.4% |
| 2 | 30 | 55.8% | 50.8% | 56.7% | 52.5% | 50.0% |
| 3 以上 | 80 | 48.8% | 58.1% | 52.2% | 53.1% | 51.2% |

## 按目标可见面积

| 分层 | 题数 | C | Cr | D | D_noP0 | B' |
| --- | --- | --- | --- | --- | --- | --- |
| 看不见 | 147 | 31.1% | 36.9% | 34.2% | 33.5% | 28.9% |
| 刚过门槛（< 0.01） | 40 | 56.9% | 56.2% | 61.2% | 65.6% | 55.0% |
| 中（0.01–0.05） | 65 | 60.8% | 66.5% | 65.4% | 68.8% | 60.8% |
| 大（≥ 0.05） | 40 | 58.8% | 68.1% | 68.8% | 64.4% | 63.8% |

起点视野里同类物体几乎从不超过 1 个：生成器给"那个 X"这类指称时，要求起点看不见别的 X（`start_context.start_refs`）。所以另按场景里同类物体的个数分层，作为"有没有可能选错"的近似。

