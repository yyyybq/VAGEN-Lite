"""data_gen/interiorgs_v10/self_marker.py and the self mode of evaluation.run_episode (the user's item 6c of 2026-11-07)."""
import unittest

from PIL import Image

from data_gen.interiorgs_v10 import self_marker as SM

ITEM = {"target_object": {"objects": [{"id": "a", "ref": "the lamp", "label": "lamp"},
                                      {"id": "b", "ref": "the sofa", "label": "sofa"}]}}


class SelfMarkerTest(unittest.TestCase):
    def test_questions_are_pg_box_questions(self):
        qs = SM.questions(ITEM)
        self.assertEqual([l for l, _ in qs], ["A", "B"])
        self.assertTrue(qs[0][1].startswith("Draw a box around the lamp. Answer with the coordinates (x1,y1,x2,y2)"))
        self.assertIn("0 to 1000", qs[1][1])

    def test_read_box(self):
        self.assertEqual(SM.read_box("(100,200,300,400)"), [25.6, 51.2, 76.8, 102.4])
        self.assertEqual(SM.read_box("(300,400,100,200)"), [25.6, 51.2, 76.8, 102.4])  # corners ordered
        self.assertEqual(SM.read_box("(0,0,1200,500)"), [0.0, 0.0, 256.0, 128.0])  # clipped to the frame
        self.assertIsNone(SM.read_box("none"))
        self.assertIsNone(SM.read_box("(10,10,10,50)"))  # empty

    def test_mark_turn(self):
        img = Image.new("RGB", (256, 256), (128, 128, 128))
        answers = {"lamp": "(100,200,300,400)", "sofa": "I cannot see it"}
        ask = lambda q, im: next(v for k, v in answers.items() if k in q)  # noqa: E731
        truth = [{"letter": "A", "in_view": True, "box": [25.6, 51.2, 76.8, 102.4]},
                 {"letter": "B", "in_view": False, "box": None}]
        shown, rec = SM.mark_turn(ask, img, ITEM, truth)
        self.assertEqual([r["iou"] for r in rec], [1.0, None])
        self.assertEqual(shown.getpixel((26, 80)), (230, 30, 40))  # A's box drawn
        self.assertEqual(img.getpixel((26, 80)), (128, 128, 128))
        s = SM.summary([rec])
        self.assertEqual((s["questions"], s["boxes"], s["visible"], s["iou_ge_05"], s["boxes_without_truth"]), (2, 1, 1, 1, 0))

    def test_run_episode_shows_the_marked_frame(self):
        from data_gen.interiorgs_v10 import evaluation as EV

        frame = Image.new("RGB", (256, 256), (0, 0, 0))

        class Env:
            current_item = {"task_id": "t", "task_type": "x", "difficulty": {"d_star": 1}}
            _current_step = 1

            def reset(self, seed):
                return {"obs_str": "<image>", "multi_modal_data": {"<image>": [frame]}}, {}

            def system_prompt(self):
                return "s"

            def step(self, text):
                return {"obs_str": "<image>", "multi_modal_data": {"<image>": [frame]}}, 0.0, True, {"actions": ["move_forward"]}

            def _calculate_canonical_metric(self):
                return {"success": False, "gates": {"distance": False}}

        seen = []

        class Policy:
            def generate(self, system, user, image, **kw):
                seen.append(image.getpixel((0, 0)))
                return "<action>move_forward|</action>"

        marked = Image.new("RGB", (256, 256), (9, 9, 9))
        row = EV.run_episode(Env(), Policy(), 0, 3, self_marker=lambda f, item, env: (marked, [
            {"letter": "A", "answer": "x", "box": None, "truth_box": None, "iou": None}]))
        self.assertEqual(seen, [(9, 9, 9)])
        self.assertEqual(row["self_marker"]["questions"], 1)


if __name__ == "__main__":
    unittest.main()
