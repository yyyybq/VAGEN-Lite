"""Matrix v3, Cr (pre-registration note 9, exploratory): Cr's RL against C and D by the method of 3.1 (m3_main31.py:
the same pairs, sign test, bootstrap draws and verdict rule) at each RL's pre-registered checkpoint (update 300, or the
last checkpoint of a run stopped early): m3crs1s0_250 (stopped at 270), m3crs1d_250 (stopped at 286), m3crs2s0_300,
m3crs2d_200 (stopped at 223).  Exploratory: it changes no H1-H3 verdict and does not replace C's result.  Read only;
writes OUT.

    python m3_cr31.py OUT_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_main31 as m  # noqa: E402

M3 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3")
m.MODELS = {"Cr": {"1s0": M3 / "m3_cr_s20261019" / "eval_test_v2_screened" / "m3crs1s0_250",
                   "1d": M3 / "m3_cr_s20261019" / "eval_test_v2_screened" / "m3crs1d_250",
                   "2s0": M3 / "m3_cr_s20261021" / "eval_test_v2_screened" / "m3crs2s0_300",
                   "2d": M3 / "m3_cr_s20261021" / "eval_test_v2_screened" / "m3crs2d_200"},
            "C": m.MODELS["C"], "D": m.MODELS["D"]}
m.COMPARE = [("Cr", "C", "exploratory"), ("Cr", "D", "exploratory")]
m.__doc__ = __doc__

if __name__ == "__main__":
    rc = m.main()
    md = Path(sys.argv[1]) / "main_31.md"
    md.write_text(md.read_text(encoding="utf-8").replace("# 矩阵第三版：主指标（预登记 3.1）",
                                                         "# 矩阵第三版：Cr 和 C、D 比（按 3.1 的方法，探索性）", 1),
                  encoding="utf-8")
    sys.exit(rc)
