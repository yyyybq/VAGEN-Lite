import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import accept_summary as A


def turn(answer, valid=True, entropy=0.05, tokens=10):
    return {"answer": answer, "valid": valid, "entropy": entropy, "tokens": tokens, "progress": 0.5,
            "failed_gates": []}


class AcceptSummaryTest(unittest.TestCase):
    """The sampled SFT-start acceptance (the user's decision 3 of 2026-11-03): entropy <= 0.06 and unreadable <= 1%."""

    def folder(self, root: Path, name: str, turns_by_episode, samples=4, temperature=0.8) -> Path:
        d = root / name
        d.mkdir()
        (d / "done.json").write_text("{}")
        (d / "model.json").write_text(json.dumps({"model": "M", "samples": samples, "temperature": temperature,
                                                  "top_p": 0.92,
                                                  "env_switches": {"prompt_format_example": "action_names"}}))
        rows = [{"task_id": f"t{i // 2}", "success": i % 2 == 0, "actions": len(t), "collisions": 0, "trajectory": t}
                for i, t in enumerate(turns_by_episode)]
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return d

    def test_lines(self):
        ok = "<action>move_forward|turn_left</action>"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = self.folder(root, "good", [[turn(ok)] * 50 for _ in range(4)])
            s = A.summarise(good)
            self.assertTrue(s["pass"])
            self.assertTrue(s["pass_strict"])
            self.assertEqual((s["episodes"], s["tasks"], s["success"], s["tasks_any_success"]), (4, 2, 0.5, 1.0))
            # 2 unreadable turns of 200 (1%) pass the starts' line, not the strict one; a made-up name is counted
            bad = [[turn(ok)] * 50 for _ in range(4)]
            bad[0][0] = turn("<action>turn_backward</action>", valid=False)
            bad[1][0] = turn("<action>move forward</action>", valid=False)
            s = A.summarise(self.folder(root, "edge", bad))
            self.assertEqual(s["unreadable_turns"], 2)
            self.assertTrue(s["pass"])
            self.assertFalse(s["pass_strict"])
            self.assertEqual(s["turn_backward"], 1)
            self.assertEqual(s["forms"], {"made_up_name": 1, "space_instead_of_underscore": 1})
            # entropy above 0.06 fails; so does a greedy run (no samples), whatever its numbers
            hot = self.folder(root, "hot", [[turn(ok, entropy=0.07)] * 50 for _ in range(4)])
            self.assertFalse(A.summarise(hot)["pass"])
            greedy = self.folder(root, "greedy", [[turn(ok)] * 50 for _ in range(4)], samples=None, temperature=None)
            self.assertFalse(A.summarise(greedy)["pass"])
            self.assertIn("note", A.summarise(greedy))
            self.assertTrue(A.summarise(root / "absent")["missing"])


if __name__ == "__main__":
    unittest.main()
