"""model_matrix/sft/newformat.py: the new-format export's folder, its comparison with the old conversations, the
smoke subset (the model line, step 2 of 2026-10-24)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path


def _policies():
    from data_gen.interiorgs_v10.admission import load_policy

    new = load_policy()
    old = copy.deepcopy(new)
    old["sft"]["export"]["format_example"] = "placeholder"
    return old, new


def _record(rid, policy, frames, task_type="fov_inclusion", split="train"):
    from data_gen.interiorgs_v10.release import _format_block, protocol_system_prompt

    conv = [{"role": "system", "content": protocol_system_prompt(task_type, policy)}]
    for k, f in enumerate(frames):
        conv.append({"role": "user", "content": f"[Observation]:\n<image>\nTask: x.\n" + _format_block(policy, k == 0),
                     "image_path": str(f)})
        conv.append({"role": "assistant", "content": "<action>move_forward|</action>"})
    return {"id": rid, "task_type": task_type, "split": split, "trajectory_steps": len(frames), "conversations": conv,
            "conversation_image_paths": [str(f) for f in frames]}


class NewFormatTest(unittest.TestCase):
    def test_link_compare_subset(self):
        from model_matrix.sft import newformat as NF

        old_p, new_p = _policies()
        self.assertNotEqual(_record("a", old_p, ["/f"]), _record("a", new_p, ["/f"]))
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src", Path(tmp) / "new"
            src.mkdir()
            (src / "frames" / "plain").mkdir(parents=True)
            for name in ("rows.jsonl", "frames.jsonl", "split_manifest.json", "task_registry.jsonl"):
                (src / name).write_text("{}\n")
            NF.cmd_link(src, out)
            self.assertTrue(all((out / n).is_symlink() for n in NF.INPUTS))
            f_old = src / "frames" / "plain" / "a.png"
            f_old.write_bytes(b"x")
            f_new = out / "frames" / "plain" / "a.png"  # the same file through the link
            olds = [_record(f"r{i}", old_p, [f_old], split="train" if i < 8 else "val") for i in range(10)]
            news = [_record(f"r{i}", new_p, [f_new], split="train" if i < 8 else "val") for i in range(10)]
            (src / "sft_trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in olds))
            (out / "sft_trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in news))
            NF.cmd_compare(src, out)
            report = json.loads((out / "newformat_compare.json").read_text())
            self.assertTrue(report["ok"] and report["same_after_rewrite"] == 10)
            news[3]["conversations"][2]["content"] = "<action>turn_left|</action>"  # a real difference is caught
            (out / "sft_trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in news))
            NF.cmd_compare(src, out)
            report = json.loads((out / "newformat_compare.json").read_text())
            self.assertFalse(report["ok"])
            self.assertEqual((report["same_after_rewrite"], report["examples"]), (9, ["r3"]))
            NF.cmd_subset(out, 4, 1, "smoke.jsonl")
            picked = [json.loads(l) for l in (out / "smoke.jsonl").read_text().splitlines()]
            self.assertEqual([r["split"] for r in picked].count("train"), 4)
            self.assertEqual([r["split"] for r in picked].count("val"), 1)
            NF.cmd_subset(out, 4, 1, "smoke2.jsonl")
            self.assertEqual((out / "smoke.jsonl").read_text(), (out / "smoke2.jsonl").read_text())  # seeded


if __name__ == "__main__":
    unittest.main()
