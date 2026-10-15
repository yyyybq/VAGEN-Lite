"""Second-version feasibility judgments of every subtask ID in COVERAGE.md (2026-10-14; updated 2026-10-15:
the user decided not to run trials F, G and the colour try-out, so their subtasks are "不做（用户决定）").

Principle (user prompt, section 3): constructive question building -- choose the objects the question needs,
compute where the camera must stand, render, and check the condition on the rendered depth -- instead of
scattering cameras at random and filtering what they happened to see.  ``python coverage_v2_judgments.py
COVERAGE.md`` rewrites the judgment cell of every subtask row and prints the revision table (section 4.0).
"""

import re
import sys

SAME = "能做，与原定义一致"
DIFF = "能做，定义有差异（逐题标注）"
DROP = "不做（用户决定）"
NO = "不能做（不出题）"
OUT = "不纳入（不属于空间推理）"
BY_GROUPS = "由各组覆盖，不单独出题"
TAG = "能做（作为标签，随所在题出现）"

# id: (new level, where it lands, what changed, why)
J = {
    "VSI-count": (SAME, "V1", "增加：每个实例都要在某帧里清楚可见（深度核验的可见部分 ≥0.4% 画面且露出 ≥50%），房间外同类实例不得出现在视频里；按官方规则排除答案 1。",
                  "原判已是能做；现在有了逐帧可见像素，能证明视频里确有计数依据。"),
    "VSI-distance-relative": (SAME, "V2", "距离改用带旋转三维框的精确最近距离（框内采样取最小点对的极限）；候选间隔按官方 30 cm / 15 cm 过滤。", "原判不变，补全实现细节。"),
    "VSI-direction-easy": (SAME, "V4", "补上角度边界：离左右分界 <30° 不出题。", "原判不变；边界阈值官方未公开，自定并标注。"),
    "VSI-direction-medium": (SAME, "V4", "'后方' = 需转身 ≥135°（官方定义）；离 135°/225° <10°、离正前 <20° 不出题。", "原判不变。"),
    "VSI-direction-hard": (SAME, "V4", "四象限，离坐标轴 <10° 不出题。", "原判不变。"),
    "VSI-route": (DIFF, "V5", "只能近似 → 能做：在可通行区域上按地标生成路线（相邻落脚点直线可走、转角明确），地标都是视频里清楚出现过的物体，挖空转向。",
                  "原判认为需要人工设计路线；构造式出题可直接由几何生成可走、可辨认的路线。差异：路线不是人设计的。"),
    "VSI-size-object": (SAME, "V3", "用原始 8 角点恢复带旋转的框取最长边（不再用轴对齐框）。", "原判不变。"),
    "VSI-size-room": (DIFF, "V3", "答案 = 房间轮廓多边形面积；视频只在该房间内走。", "原判'只能近似'的理由（轮廓 ≠ 点云 alpha-shape）仍成立，改为'能做但定义不同'并逐题标注。"),
    "VSI-distance-absolute": (SAME, "V2", "同相对距离。", "原判不变。"),
    "VSI-order": (SAME, "V6", "只能近似 → 能做：首次出现 = 物体可见像素第一次超过阈值的那一帧（深度核验），相邻首次出现至少隔 3 帧。",
                  "原判认为没有像素级可见性；渲染深度给出逐帧可见像素，与 VSI 的'像素数过阈值'定义一致。"),
    "MMSI-position-camera-camera": (SAME, "M1", "第二台相机由第一台按已知平移和转向生成，两图至少共同拍到 2 个能认出的物体。", "原判不变；共享物体改为渲染后核验。"),
    "MMSI-position-camera-object": (SAME, "M2", "关键物体只在一张图里能认出、另一张图里完全看不到；两图至少共享 2 个能认出的物体。", "原判不变；'单图答不出'由构造保证。"),
    "MMSI-position-camera-region": (DIFF, "M3", "只能近似 → 能做：区域 = 由独有物体称呼的房间，或由一组物体（如餐桌和餐椅）地面投影的外包凸多边形外扩 0.4 m 的功能区；区域面积至少 85% 落在同一个方向扇区才出题；两图至少共享 1 个能认出的物体。",
                                    "原判认为功能区没有标注；用物体组合构造出边界明确的功能区。差异：MMSI 的区域由人划定，逐题标明。"),
    "MMSI-position-object-object": (SAME, "M2", "参照方向由题目'P 在 Q 的某方向'给出，R 只在另一张图里。", "原判不变。"),
    "MMSI-position-object-region": (DIFF, "M3", "同相机–区域。", "同相机–区域。"),
    "MMSI-position-region-region": (DIFF, "M3", "同相机–区域。", "同相机–区域。"),
    "MMSI-attribute-measurement": (SAME, "M2", "高度 / 长度比较，两物体各只在一张图里，相差 ≥30%。", "原判不变。"),
    "MMSI-attribute-appearance": (DIFF, "M5（拍到哪一面）；颜色不做（用户决定）；材质、部件数不能做",
                                  "拆开：'图 2 拍到物体哪一面'能做（以图 1 拍到的一面为正面，由两台相机相对物体的方位算出）；颜色不做（用户 2026-10-15 决定）；材质、部件数量不出题。",
                                  "原判把整类记为近似；按用户决定拆成三部分。"),
    "MMSI-motion-camera": (SAME, "M1", "平移、原地左右转、俯仰都由已知位姿生成。", "原判不变。"),
    "MMSI-motion-object": (DROP, "试验 G 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 G（物体运动），MMSI 物体运动类不出题。"),
    "MMSI-multistep": (DIFF, "M6", "只能近似 → 能做：三张图从同一位置转身连拍，X、Y、Z 各只在一张图里，问'站在 X 面向 Y，Z 在哪'，少任何一张都答不出。",
                       "多步依赖由构造保证；差异：MMSI 多步题由人设计、可组合更多能力。"),
    "MC-setting-rotation": (TAG, "M7", "原地每次右转 90°，每张图正中有一个能认出的物体。", "原判不变。"),
    "MC-setting-around": (SAME, "M8", "只能近似 → 能做：按 A、B 连线摆正视图和左右视图，'A 挡住 B'用深度核验（远处物体可见部分低于阈值且被前景覆盖）。",
                          "原判的理由是'遮挡无法只靠框上几条射线证明'；渲染深度可以直接核验遮挡。阈值待审图校准。"),
    "MC-setting-among": (SAME, "M9", "只能近似 → 能做：中心物体四周近直角（偏差 ≤35°）的 4 个物体，相机在各周围物体一侧（可偏开 ≤35°）朝向中心；核验中心能认出、身后物体完全看不到、对面物体在中心后方露出，并按论文附录 B 限制严重遮挡的张数。构型稀少：150 个 train 场景里找到 18 组。",
                         "原判认为真实可见性无法验证；深度核验后可验证。"),
    "MC-pattern-linear": (TAG, "M8", "随 Around 题出现。", "原判不变。"),
    "MC-pattern-perpendicular": (TAG, "M9", "随 Among 题出现。", "原判不变。"),
    "MC-dynamics-translation": (TAG, "M10", "随平移题出现。", "原判不变。"),
    "MC-dynamics-rotation": (TAG, "M7", "随转身题出现。", "原判不变。"),
    "MC-dynamics-simultaneous": (TAG, "M8、M9", "随 Around / Among 的 q1（移动方向）出现。", "原判不变。"),
    "MC-dynamics-sequential": (TAG, "M9（q2-sequence）、M8（q3）", "随'先转身再前进'的题出现。", "原判不变。"),
    "MC-relation-agent-agent": (TAG, "M8、M9", "随 q1 出现。", "原判不变。"),
    "MC-relation-agent-object": (TAG, "M7、M9、M8", "随转身题、Among q2、Around q3 出现。", "原判不变。"),
    "MC-relation-object-object": (TAG, "M8（q2）、M9（q5）", "随相应题出现。", "原判不变。"),
    "MC-perspective-self": (TAG, "M7、M8、M9", "随相应题出现。", "原判不变。"),
    "MC-perspective-other-level1": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "MC-perspective-other-level2": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "MC-rotation-current": (SAME, "M7", "按 MindCube 原题模板。", "原判不变。"),
    "MC-rotation-mental-turn": (SAME, "M7", "按 MindCube 原题模板（左转、右转、转身）。", "原判不变。"),
    "MC-around-q1": (SAME, "M8", "移动方向由两台相机的位姿算出。", "原判不变。"),
    "MC-around-q2": (SAME, "M8", "只能近似 → 能做：侧视图里谁在前谁在后由深度核验的遮挡保证。", "同 Around 设定。"),
    "MC-around-q3": (SAME, "M8", "'转身后直走'按 1 m 模拟距离变化。", "原判不变；原题未规定步长，自定 1 m 并标注。"),
    "MC-around-q4": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "MC-among-q1": (SAME, "M9", "移动方向由位姿算出。", "原判不变。"),
    "MC-among-q2-current": (SAME, "M9", "相机身后的周围物体。", "原判不变。"),
    "MC-among-q2-sequence": (SAME, "M9", "转身后前进 1 m 是否更近。", "原判不变；步长自定。"),
    "MC-among-q3": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "MC-among-q4": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "MC-among-q5": (SAME, "M9", "从第 k 张图看中心物体左（右）侧的周围物体。", "原判不变。"),
    "MC-among-q6": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "MC-translation-transitive": (SAME, "M10", "两图朝向相同只平移，A、C 各只在一张图里，B 共享。", "原判不变；平移设定未随官方数据发布，按其辅助代码的题型构造。"),
    "VS-01": (DIFF, "S1", "水平 8 扇区照搬 ViewSpatial 代码；'前 = 沿视线更远'按答案分布推断；上下分档自定。",
              "原判'能按原定义算'过于乐观：相机视角题的生成代码未公开，上下与前后的约定只能推断，故标为定义有差异。"),
    "VS-02": (NO, "不能做", "不变。", "需要人物头部和身体朝向，场景里没有人（用户决定）。"),
    "VS-03": (NO, "不能做", "不变。", "同上。"),
    "VS-04": (NO, "不能做", "不变。", "同上。"),
    "VS-05": (SAME, "M11", "A、B、C 分布在转身连拍的不同图里，答案用 ViewSpatial 场景模拟题的同一算法。", "原判不变；跨图必要性由构造保证。"),
    "ES-01": (DIFF, "S1", "只能近似 → 能做：按两个物体可见部分（深度核验）的二维框左右分离判断。", "可见部分二维框就是人在画面上框出的物体；与官方'标注框投影'的差别逐题标注。"),
    "ES-02": (DIFF, "S1", "同 ES-01。", "同 ES-01。"),
    "ES-03": (DIFF, "S2", "同 ES-01（上下分离）。", "同 ES-01。"),
    "ES-04": (DIFF, "S2", "同 ES-03。", "同 ES-01。"),
    "ES-05": (SAME, "S3", "只能近似 → 能做：可见部分的平均深度，最近者领先 ≥ max(0.4 m, 15%)。", "原判缺像素深度；渲染深度与官方'框内平均深度'定义一致。"),
    "ES-06": (SAME, "S3", "同 ES-05（最远）。", "同 ES-05。"),
    "3D-01": (DIFF, "S2", "只能近似 → 能做：比较框中心高度，且顶面、底面高低与中心一致、中心高差 ≥0.25 m 才出题。", "官方由人判断；本项目先定几何量并只出明确的题。"),
    "3D-02": (DIFF, "S2", "只出明确正例（上方物体底面不低于下方物体顶面 −5 cm，且水平投影至少一半重叠）和明确反例。", "同上。"),
    "3D-03": (DIFF, "S3", "可见部分平均深度，差距 ≥ max(0.4 m, 15%)。", "同上。"),
    "3D-04": (DIFF, "S4", "三维框最近距离 ≤0.3 m 为挨着、≥1.5 m 为远离，中间不出题。", "同上。"),
    "3D-05": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "3D-06": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "3D-07": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "3D-08": (DIFF, "S4", "三维框最近距离，两候选差距 ≥ max(0.5 m, 30%)。", "官方由人判断。"),
    "3D-09": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "3D-10": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "3D-11": (DROP, "S7 不做", "能做 → 不做（用户在标注中决定）；第二版曾实现为 S7：只用明显细长的物体比较长轴，≤10° 平行、≥80° 垂直。",
              "用户 2026-10-15 标注时决定这类题全都不要（例如没有明确朝向的物体也被问到平行还是垂直）。"),
    "3D-12": (DROP, "试验 F 不做", "第二版初稿曾列为先试再定，现改为不做。", "用户 2026-10-15 决定不做试验 F（物体正面），依赖物体正面的子任务不出题。"),
    "BL-01": (SAME, "S5", "同类实例要么清楚可见、要么完全不在画面，有'露一半'的就不出这张图。", "原判不变；用深度核验区分。"),
    "BL-02": (SAME, "S3", "只能近似 → 能做：两点取在可见表面内部（避开边缘），按渲染深度比较。", "原判缺表面深度；渲染深度直接给出。"),
    "BL-03": (DIFF, "S1（左右）、S2（上下、on top）、S4（near / far / touching）；朝向类关系不做（用户决定）",
              "只能近似 → 能做（指定的几类关系）：左右按可见部分二维框，上下与接触按三维框，远近按框间最近距离；只出明确正反例。",
              "VSR 关系由人标注；本项目按几何定义并标注差异。VSR 的朝向类关系（facing 等）用户 2026-10-15 决定不做；其他关系（contains、inside、part of、surrounding 等）不在本次范围，见报告。"),
    "BL-04": (SAME, "M1", "相机绕同一物体左右移动，两图都对着它。", "原判不变。"),
    "BL-05": (DIFF, "S6", "只能近似 → 能做：正确框 = 可见部分二维框，干扰框按 BLINK 的高斯扰动生成（IoU 0.25–0.6）。", "差异：LVIS 是完整物体框，这里是可见部分框。"),
    "BL-06": (DIFF, "M4", "只能近似 → 能做：图 1 的表面点按深度和位姿投到图 2，再用图 2 的深度确认没被挡住。", "差异：BLINK 原题是平面单应，这里是一般三维场景。"),
    "BL-07": (OUT, "不纳入", "可选相邻能力 → 不纳入。", "拼图不属于空间推理（用户决定）。"),
    "BL-09": (NO, "不能做", "不变。", "需要跨物体的部件关键点（用户决定）。"),
    "BL-10": (NO, "不能做", "不变。", "需要功能部件标注（用户决定）。"),
    "SITE-C01": (BY_GROUPS, "S5、V1", "不单独出题。", "SITE 的题来自其他数据集，由对应组覆盖（用户决定）。"),
    "SITE-C02": (BY_GROUPS, "S1、S2、S4、V4", "不单独出题。", "同上。"),
    "SITE-C03": (BY_GROUPS, "S6", "不单独出题。", "同上。"),
    "SITE-C04": (BY_GROUPS, "S3、V2、V3", "不单独出题。", "同上。"),
    "SITE-C05": (BY_GROUPS, "M1–M11", "不单独出题。", "同上。"),
    "SITE-C06": (BY_GROUPS, "V5（只覆盖静态导航）", "只覆盖静态导航部分；物体运动轨迹、意图不覆盖。", "同上。"),
    "SITE-N01": (NO, "不能做", "只能近似 → 不能做。", "需要人物活动的第一、第三人称同步画面（用户决定）。"),
    "SITE-N02": (NO, "不能做", "只能近似 → 不能做。", "同上。"),
    "SITE-N03": (NO, "不能做", "不变。", "同上。"),
    "SITE-N04": (NO, "不能做", "不变。", "同上。"),
}
EXTRA_ROWS = [
    ("BL-08 IQ_Test", OUT, "不纳入", "原文已排除，现正式列为不纳入。", "用户决定。"),
    ("BLINK Art_Style、Forensic_Detection、Relative_Reflectance、Visual_Similarity", OUT, "不纳入", "不变。", "不属于空间推理（用户决定）。"),
    ("MMSI 外观属性：材质、部件数量", NO, "不能做", "从 MMSI-attribute-appearance 中拆出。", "没有标注，也不用 VLM 补（用户决定）。"),
    ("MMSI 外观属性：颜色", DROP, "不做", "从 MMSI-attribute-appearance 中拆出；曾试做 10 道。", "用户 2026-10-15 决定不做。"),
]


def old_level(cell: str) -> str:
    m = re.match(r"\*\*(.+?)\*\*", cell.strip())
    return m.group(1) if m else ""


def rewrite(md: str):
    lines = md.splitlines()
    table = []
    for i, l in enumerate(lines):
        if not l.startswith("|"):
            continue
        cells = l.strip().strip("|").split("|")
        sid = cells[0].strip().split()[0].strip("`")
        if sid not in J:
            continue
        k = next((n for n, c in enumerate(cells) if c.strip().startswith("**")), None)
        if k is None:
            continue
        old = old_level(cells[k])
        if old == "人工构建和审核":   # VSI-route: the bold text is in the GT column; the judgment is the next bold cell
            k2 = next((n for n, c in enumerate(cells) if n > k and c.strip().startswith("**")), None)
            if k2 is not None:
                k = k2
                old = old_level(cells[k])
        new, where, what, why = J[sid]
        cells[k] = f" **{new}**（{where}）。{what} "
        lines[i] = "|" + "|".join(cells) + "|"
        table.append((sid, old, new, where, what, why))
    return "\n".join(lines) + "\n", table


if __name__ == "__main__":
    md = open(sys.argv[1], encoding="utf-8").read()
    new_md, table = rewrite(md)
    missing = sorted(set(J) - {t[0] for t in table})
    out = ["| 子任务 | 第一版判断 | 第二版判断 | 落到哪里 | 改了什么 | 为什么 |", "|---|---|---|---|---|---|"]
    for sid, old, new, where, what, why in table:
        out.append(f"| `{sid}` | {old} | **{new}** | {where} | {what} | {why} |")
    for name, new, where, what, why in EXTRA_ROWS:
        out.append(f"| {name} | — | **{new}** | {where} | {what} | {why} |")
    open(sys.argv[2], "w", encoding="utf-8").write(new_md)
    open(sys.argv[3], "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(len(table), "rows rewritten; missing:", missing)
