import unittest
import argparse
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from .image_contract_v2 import image_stats, pose_from_forward
from .qa_eval import parse_answer


class DiagnosisRegressionTests(unittest.TestCase):
    def test_camera_down_convention(self):
        pose = pose_from_forward([1, 0, 1.5], [-1, 0, -0.1])
        self.assertLess(pose[2, 1], 0)
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1)
        # An above-axis point must move UP on an image with positive fy.
        center = pose[:3, 3] + pose[:3, 2] * 10
        above = center + np.array([0, 0, 0.2])
        camera = pose[:3, :3].T @ (above - pose[:3, 3])
        self.assertLess(camera[1] / camera[2], 0)

    def test_vertical_forward_is_valid(self):
        for z in (1, -1):
            pose = pose_from_forward([0, 0, 0], [0, 0, z])
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
        with self.assertRaises(ValueError):
            pose_from_forward([0, 0, 0], [0, 0, 0])

    def test_solid_colour_is_blank_despite_channel_difference(self):
        result = image_stats(Image.new('RGB', (512, 512), (128, 160, 245)))
        self.assertGreater(result['std'], 5)
        self.assertFalse(result['nonblank'])
        self.assertEqual(result['spatial_channel_std_max'], 0)
        self.assertFalse(result['target_visibility_verified'])

    def test_spatial_detail_is_not_target_visibility_proof(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8); a[::2] = 255
        result = image_stats(Image.fromarray(a))
        self.assertTrue(result['nonblank'])
        self.assertFalse(result['target_visibility_verified'])

    def test_missing_and_invalid_answers_are_not_no(self):
        self.assertIsNone(parse_answer(''))
        self.assertIsNone(parse_answer('Cannot determine.'))
        self.assertEqual(parse_answer('No.'), 'No')

    def test_legacy_bank_cannot_silently_resume_new_pose_convention(self):
        from .generate_paired_qa import generate
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out/'manifest.jsonl').write_text(json.dumps({'sample_id': 'old'})+'\n')
            args = argparse.Namespace(output_dir=td)
            with self.assertRaisesRegex(ValueError, 'separate versioned'):
                generate(args)

    def test_freeze_does_not_trust_legacy_nonblank_flag(self):
        from .finalize_act2qa import freeze
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            image_path = directory/'flat.png'
            Image.new('RGB', (512, 512), (128, 160, 245)).save(image_path)
            pose = np.eye(4).tolist()
            row = {'sample_id': 'fixture', 'private_answer': 'Yes', 'state_pose_c2w': pose,
                   'observability_validity': 'valid', 'question': 'Answer Yes or No.',
                   'public_observation': {'image_path': str(image_path)},
                   'render': {'image_path': str(image_path), 'pose_c2w': pose, 'rgb_stats': {'nonblank': True}}}
            bank = directory/'fixture.jsonl'; bank.write_text(json.dumps(row)+'\n')
            summary = freeze(argparse.Namespace(output_dir=td, rendered_bank=str(bank), errors=''))
            self.assertEqual(summary['eligible'], 0)
            self.assertEqual(summary['decode_valid'], 1)
            self.assertEqual(summary['pose_matched'], 1)


if __name__ == '__main__':
    unittest.main()
