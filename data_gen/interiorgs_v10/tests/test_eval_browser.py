"""eval_browser.py: the static trajectory page (data.js, index.html, frames stored once)."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_gen.interiorgs_v10 import eval_browser as B


def _frame(path, shade):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (shade, 0, 0)).save(path, format="JPEG")


class BrowserTest(unittest.TestCase):
    def test_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp) / "eval"
            ev.mkdir()
            tasks = [{"task_id": "t0", "scene_id": "s0", "task_type": "fov_inclusion", "difficulty": {"d_star": 2},
                      "task_description": "Fit both."}]
            (ev / "tasks.jsonl").write_text(json.dumps(tasks[0]) + "\n")
            for model, shade, ok in (("v10_rl_050", 90, True), ("v10_sft_e3", 120, False)):
                _frame(ev / model / "frames" / "000_00.jpg", 10)          # the same start frame for every model
                _frame(ev / model / "frames" / "000_01.jpg", shade)
                turn = {"frame": "frames/000_00.jpg", "user": "Current camera pose: [tx=1]\nEnvironment Feedback: Action executed.\nTask",
                        "answer": "<action>turn_left|</action>", "actions": ["turn_left"], "valid": True, "collisions": 0,
                        "reward": 4.99 if ok else -0.01, "met": ok, "failed_gates": [] if ok else ["both_in_view"],
                        "progress": 1.0 if ok else 0.2}
                row = {"index": 0, "task_id": "t0", "task_type": "fov_inclusion", "d_star": 2, "success": ok,
                       "actions_to_success": 1 if ok else None, "turns": 1, "actions": 1, "collisions": 0,
                       "invalid_turns": 0, "trajectory": [turn], "final_frame": "frames/000_01.jpg",
                       "end_reason": "success_auto" if ok else "max_turns", "start": {"failed_gates": ["x"], "progress": 0.1}}
                (ev / model / "results.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({"index": 1, "error": "renderer"}) + "\n")
            out = Path(tmp) / "page"
            res = B.build(ev, out)
            self.assertEqual(res, {"models": ["v10_sft_e3", "v10_rl_050"], "tasks": 1, "images": 3})  # start frame once
            self.assertTrue((out / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>"))
            js = (out / "data.js").read_text(encoding="utf-8")
            data = json.loads(js[len("window.EVAL_DATA = "):].rstrip().rstrip(";"))
            self.assertEqual([m["label"] for m in data["models"]], ["SFT 第 3 轮", "RL 第 50 次更新"])
            e = data["episodes"]["v10_rl_050"]["0"]
            self.assertEqual((e["s"], e["end"], e["t"][0]["fb"], e["t"][0]["pose"], e["t"][0]["g"]),
                             (True, "success_auto", "Action executed.", "[tx=1]", []))
            self.assertTrue((out / "img" / (e["t"][0]["f"] + ".jpg")).is_file())
            self.assertEqual(data["episodes"]["v10_sft_e3"]["1"], {"error": "renderer"})
            self.assertEqual(B.build(ev, Path(tmp) / "one", ["v10_rl_050"])["models"], ["v10_rl_050"])

            # round 9: a second folder with the same tasks (diagnostic runs), labels and the two first columns
            diag = Path(tmp) / "diag"
            (diag / "D1").mkdir(parents=True)
            (diag / "tasks.jsonl").write_text((ev / "tasks.jsonl").read_text())
            _frame(diag / "D1" / "frames" / "000_00.jpg", 10)
            (diag / "D1" / "results.jsonl").write_text((ev / "v10_rl_050" / "results.jsonl").read_text())
            both = B.build([ev, diag], Path(tmp) / "both", labels={"D1": "D1 每个动作检查成功"},
                           defaults={"left": "v10_sft_e3", "right": "D1"})
            self.assertEqual(both["models"], ["v10_sft_e3", "v10_rl_050", "D1"])
            js = (Path(tmp) / "both" / "data.js").read_text(encoding="utf-8")
            data = json.loads(js[len("window.EVAL_DATA = "):].rstrip().rstrip(";"))
            self.assertEqual(data["models"][-1], {"name": "D1", "label": "D1 每个动作检查成功"})
            self.assertEqual(data["defaults"], {"left": "v10_sft_e3", "right": "D1"})
            (diag / "tasks.jsonl").write_text(json.dumps(dict(tasks[0], task_id="other")) + "\n")
            with self.assertRaises(ValueError):  # other tasks: refuse to put them side by side
                B.build([ev, diag], Path(tmp) / "bad")


if __name__ == "__main__":
    unittest.main()
