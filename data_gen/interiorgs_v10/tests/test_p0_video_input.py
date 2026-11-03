"""data_gen.interiorgs_qa.evaluate.video_input: a Qwen3-VL video carries the metadata LLaMA-Factory gives it in training."""
import unittest

from data_gen.interiorgs_qa.evaluate import video_input


class VideoInputTest(unittest.TestCase):
    def test_qwen3_vl_gets_the_training_metadata(self):
        frames = list(range(32))
        got, meta = video_input(frames, "qwen3_vl", 2.0)
        self.assertIs(got, frames)
        self.assertEqual(meta, {"fps": 2.0, "frames_indices": list(range(32)), "total_num_frames": 32,
                                "duration": 16.0, "do_sample_frames": False})

    def test_other_models_get_the_frames_only(self):
        frames = list(range(32))
        self.assertIs(video_input(frames, "qwen2_5_vl", 2.0), frames)


if __name__ == "__main__":
    unittest.main()
