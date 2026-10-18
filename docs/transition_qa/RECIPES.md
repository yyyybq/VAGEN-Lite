<!-- 由 python -m data_gen.sft_mixture.build --config data_gen/sft_mixture/configs/mixture_matrix_v2_20261018.yaml 生成（exps/sft_mixture_matrix_v2_20261018/recipes.md 的副本）。说明见 PROGRESS.md 第 3 节。 -->

# 配方表：matrix_v2_20261018

由 `python -m data_gen.sft_mixture.build --config ...` 生成。数字是样本条数（一条 = 一段对话）：P0、P1 一条是一道题（P0 的视频题一条含同一段视频的几道题），P2 一条是一整条动作轨迹（多回合）。

## 池子

| 池 | 可用训练条数 | 可用验证条数 | 每个配方都用的验证条数 |
| --- | --- | --- | --- |
| P0 仿 benchmark 的 QA | 177 | 20 | 20 |
| P1 转移题 | 35039 | 4073 | 300 |
| P2 SFT 动作轨迹 | 9802 | 1293 | 300 |

## 配方

| 组 | 输出格式 | P0 | P1 | P2 | 训练条数合计 | 助手回合 | 图片（含视频帧） | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | action_only | 0 | 0 | 0 | 0 | 0 | 0 | 不做 SFT |
| B | action_only | 177 | 0 | 0 | 177 | 206 | 1057 | 只练 P0 |
| C | action_only | 177 | 0 | 9802 | 9979 | 25449 | 26300 | P0 + 动作轨迹 |
| D | action_only | 177 | 9802 | 9802 | 19781 | 35251 | 37600 | P0 + 转移题 + 动作轨迹（三种样本分开，同一次训练） |
| E | action_only（RL 时 free_think） | 177 | 9802 | 9802 | 19781 | 35251 | 37600 | SFT 与 D 相同，共用模型；RL 时换 free_think 提示（探索性，排在 D 之后；RL 前 50 步解析失败率超过 20% 就放弃） |

## 注意

- P0 的验证行来自 train 场景（样题导出把每第 10 道题留作验证，都是 train 场景；全量 P0 题库有 val 场景的题后改用那些），不是 val 场景
