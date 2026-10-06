"""CPU checks for transfer identity, episode isolation and execution contract."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_starvla_transfer import partition_episodes, prepare_data
from eval_starvla_mobile import rollout
from export_starvla_backbone import extract_backbone
from vagen.envs.robocasa.starvla_contract import CAMERAS, PolicyContract, gym_action, verify_server
from vagen.envs.robocasa.utils.actions import GYM_SLICES, STARVLA_STATE_KEY_DIMS


class TransferTest(unittest.TestCase):
    def test_export_preserves_backbone_and_excludes_action_head(self):
        weight = object()
        state = {"qwen_vl_interface.model.model.embed_tokens.weight": weight, "action_model.net.weight": object()}
        self.assertEqual(extract_backbone(state), {"model.embed_tokens.weight": weight})
        with self.assertRaises(ValueError):
            extract_backbone({"action_model.net.weight": weight})

    def contract(self, state=False):
        return PolicyContract(CAMERAS, (224, 224), state, 16)

    def observation(self):
        obs = {k: np.full((256, 256, 3), i * 70, dtype=np.uint8) for i, k in enumerate(CAMERAS)}
        obs["annotation.human.task_description"] = "Navigate to the sink."
        obs.update({k: np.arange(n, dtype=np.float32) / 3 for k, n in STARVLA_STATE_KEY_DIMS.items()})
        return obs

    def test_split_is_episode_disjoint_deterministic_and_nonempty(self):
        episodes = [{"episode_index": i, "length": 20} for i in range(503)]
        train, val = partition_episodes(episodes, .1, 42)
        self.assertEqual((len(train), len(val)), (453, 50))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(range(503)))
        self.assertEqual((train, val), partition_episodes(list(reversed(episodes)), .1, 42))

    def test_observations_match_camera_order_and_omit_state(self):
        example = self.contract().example(self.observation())
        self.assertNotIn("state", example)
        self.assertEqual([int(x[0, 0, 0]) for x in example["image"]], [0, 70, 140])
        self.assertTrue(all(x.shape == (224, 224, 3) for x in example["image"]))

    def test_optional_state_matches_per_key_training_transform(self):
        obs = self.observation()
        expected = np.concatenate([np.concatenate([np.sin(obs[k]), np.cos(obs[k])]) for k in STARVLA_STATE_KEY_DIMS])
        np.testing.assert_allclose(self.contract(True).example(obs)["state"][0], expected)

    def test_invalid_action_shape_and_nan_fail_closed(self):
        for arr in [np.zeros((1, 8, 12)), np.zeros((1, 16, 7)), np.full((1, 16, 12), np.nan)]:
            with self.assertRaises(ValueError):
                self.contract().actions(arr)

    def test_action_mapping_preserves_base_and_mode_and_clips(self):
        space = {k: SimpleNamespace(low=-np.ones(e-s), high=np.ones(e-s)) for k, (s, e) in GYM_SLICES.items()}
        arr = np.arange(12, dtype=np.float32) / 10
        out = gym_action(arr, space)
        np.testing.assert_allclose(out["action.base_motion"], [.7, .8, .9, 1.])
        np.testing.assert_allclose(out["action.control_mode"], [1.])

    def test_wrong_server_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_server({"ckpt_path": "/another.pt"}, "/expected.pt", self.contract())

    def test_rollout_stops_mid_chunk_on_success_and_uses_env_language(self):
        outer = self
        class Env:
            action_space = {k: SimpleNamespace(low=-np.ones(e-s), high=np.ones(e-s)) for k, (s, e) in GYM_SLICES.items()}
            count = 0
            def reset(self, seed):
                return outer.observation(), {}
            def step(self, actions):
                self.count += 1
                return outer.observation(), 1., False, False, {"success": self.count == 3}
        class Policy:
            def predict_action(self, query):
                outer.assertEqual(query["examples"][0]["lang"], "Navigate to the sink.")
                outer.assertNotIn("state", query["examples"][0])
                return {"data": {"actions": np.zeros((1, 16, 12))}}
        result = rollout(Env(), Policy(), self.contract(), 42, 100, 8)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["policy_calls"], 1)

    def test_split_overlay_never_copies_full_data_statistics(self):
        real = ROOT / "playground/Datasets/robocasa365/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot"
        if not real.is_dir():
            self.skipTest("Local RoboCasa metadata unavailable")
        with tempfile.TemporaryDirectory() as temp:
            src, out = Path(temp) / "source", Path(temp) / "out"
            (src / "meta").mkdir(parents=True)
            for name in ("info.json", "modality.json", "tasks.jsonl"):
                (src / "meta" / name).write_bytes((real / "meta" / name).read_bytes())
            info = json.loads((src / "meta/info.json").read_text())
            episodes = [{"episode_index": i, "length": 20} for i in range(4)]
            (src / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
            (src / "meta/stats_gr00t.json").write_text('{"should_not_leak": true}')
            for ep in range(4):
                path = src / info["data_path"].format(episode_chunk=0, episode_index=ep)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test fixture")
                for key in info["features"]:
                    if key.startswith("observation.images."):
                        video = src / info["video_path"].format(episode_chunk=0, episode_index=ep, video_key=key)
                        video.parent.mkdir(parents=True, exist_ok=True)
                        video.touch()
            result = prepare_data([("test", str(src))], out, .25, 42)
            train = out / "data/train/test/lerobot"
            self.assertFalse((train / "meta/stats_gr00t.json").exists())
            self.assertEqual(len(list(train.glob("data/*/*.parquet"))), 3)
            self.assertFalse(set(result["datasets"][0]["train_ids"]) & set(result["datasets"][0]["val_ids"]))
            self.assertTrue((src / "meta/stats_gr00t.json").is_file())

    def test_local_hf_model_type_does_not_depend_on_directory_name(self):
        spec = importlib.util.spec_from_file_location("vlm_selection", ROOT / "third_party/starVLA/starVLA/model/modules/vlm/__init__.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actor/huggingface"
            path.mkdir(parents=True)
            (path / "config.json").write_text('{"model_type": "qwen2_5_vl"}')
            self.assertEqual(module.local_model_type(str(path)), "qwen2_5_vl")


if __name__ == "__main__":
    unittest.main()
