# 分析 5：指认题试版（30 道，探索性）

Questions (build): start frames of 30 screened validation tasks (validation scenes; frames/<index>_00.jpg of a validation result, the frame the env renders at the task's start pose, 256 x 256), 5 a family in a seeded order, X a target named in the task text that is identifiable at the start (initial_observability).  The true box is the target's AABB projected with the env's own camera (canonical_camera.build_canonical_camera + visual_bbox_metrics. _project_object, as occlusion.identifiable), clipped to the image.  Score: the first "(x, y)" (or "x, y") in the answer, read as pixels of the 256 x 256 image and, apart, as Qwen-VL's 0-1000 normalised coordinates; right when the point falls in the true box.

| 模型 | 题数 | 读得出坐标 | 按 256 像素算落在真值框内 | 按 0–1000 归一化算落在框内 | 回答样例 |
| --- | --- | --- | --- | --- | --- |
| `base_qwen3` | 30 | 25 | 0 | 18 | `There are none.` / `(256, 768)` |
| `m3ds1s0_300` | 30 | 23 | 1 | 1 | `(71, 192)` / `0,0` |
