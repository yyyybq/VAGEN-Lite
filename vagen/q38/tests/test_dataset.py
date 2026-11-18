"""T10 (DESIGN_ASYNC 5.1): AgenticDatasetV1 rows through verl's collate_fn and the V1 trainer's fetch path.

Pass line: the rows have raw_prompt (its length after collate_fn is the row count, so 1 a fetch) and index; no dummy
tensors; uids without underscores; every row field stays a NonTensorStack (the async trainers persist exactly those
for re-issuing after a resume).  The fetch path is copied from verl/verl/trainer/ppo/v1/trainer_base.py:1533-1586 and
the worker's per-row extraction from agent_loop_tq.py:89-101.
"""

import os
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from vagen.gym_agent_dataset import AgenticDataset
from vagen.q38.dataset import DUMMY_TENSOR_KEYS, AgenticDatasetV1, single_data_file

R8 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_train_q3s3_20261027")
V4_TRAIN_ENV = R8 / "rl_qw3v4s3" / "train_env.yaml"  # the 8B main line V4 run, read only

ENV_YAML = """\
envs:
- name: ActiveSpatial
  data_source: active_spatial
  max_turns: 12
  response_length_per_turn: 384
  config:
    client_url: ${oc.env:R1_RENDER_URL}
    jsonl_path: /nonexistent/train.jsonl
    nested: {a: 1, b: [1, 2]}
  n_envs: 30
  seed: [0, 30]
  seed_list: [%s]
- name: ActiveSpatial
  data_source: active_spatial_b
  max_turns: 12
  response_length_per_turn: 384
  config:
    client_url: ${oc.env:R1_RENDER_URL}
    jsonl_path: /nonexistent/other.jsonl
  n_envs: 6
  seed: [100]
""" % ", ".join(str(i) for i in range(29, -1, -1))

ROW_KEYS = {"env_name", "seed", "config", "max_turns", "response_length_per_turn", "data_source", "agent_name",
            "raw_prompt", "index"}


def fetch_one_gen_batch(dataloader_it):
    """trainer_base.py:1533-1537 (PPOTrainer._fetch_one_gen_batch)."""
    from verl.utils import tensordict_utils as tu

    batch_dict = next(dataloader_it)
    batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
    return batch_dict, tu.get_tensordict(batch_dict)


def next_train_batch(dataloader_it, num_prompts, global_steps):
    """trainer_base.py:1549-1564 with gen_batch_size 1 (forced for the async modes, trainer_base.py:754-766)."""
    from verl.utils import tensordict_utils as tu

    chunks = [fetch_one_gen_batch(dataloader_it)[1] for _ in range(num_prompts)]
    batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
    tu.assign_non_tensor_data(batch, "global_steps", global_steps)
    return batch


def worker_prompts(batch):
    """agent_loop_tq.py:89-101: the per-row kwargs a worker hands to one prompt's agent loops."""
    from tensordict import NonTensorData, NonTensorStack

    prompts = []
    for i in range(len(batch)):
        prompt = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prompt[k] = v[i]
            elif isinstance(v, NonTensorStack):
                prompt[k] = v[i].data
            elif isinstance(v, NonTensorData):
                prompt[k] = v.data
            else:
                raise TypeError(f"unsupported {type(v)} for {k}")
        prompts.append(prompt)
    return prompts


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env_yaml = Path(cls.tmp.name) / "train_env.yaml"
        cls.env_yaml.write_text(ENV_YAML)
        os.environ.setdefault("R1_RENDER_URL", "http://127.0.0.1:1/render")
        cls.data_config = OmegaConf.create({"seed": 20261022, "shuffle": True})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def dataset(self, **kw):
        return AgenticDatasetV1(data_files=str(self.env_yaml), config=self.data_config, **kw)

    def test_rows(self):
        ds, parent = self.dataset(), AgenticDataset(data_files=str(self.env_yaml), config=self.data_config)
        self.assertEqual(len(ds), 36)
        self.assertEqual(len(ds), len(parent))
        for idx in range(len(ds)):
            row = ds[idx]
            self.assertEqual(set(row), ROW_KEYS)
            for key in DUMMY_TENSOR_KEYS:
                self.assertNotIn(key, row)
            self.assertFalse(any(isinstance(v, torch.Tensor) for v in row.values()))
            self.assertEqual(row["raw_prompt"], [])
            self.assertEqual(row["index"], idx)
            self.assertIsInstance(row["index"], int)
            expected = {k: v for k, v in parent[idx].items() if k not in DUMMY_TENSOR_KEYS}
            self.assertEqual({k: row[k] for k in expected}, expected)
        self.assertEqual([ds[i]["seed"] for i in range(30)], list(range(29, -1, -1)))  # seed_list wins
        self.assertEqual([ds[i]["seed"] for i in range(30, 36)], list(range(100, 106)))  # active spatial rows
        self.assertEqual(ds[0]["agent_name"], "gym_agent")
        self.assertEqual(ds[0]["config"]["client_url"], os.environ["R1_RENDER_URL"])  # resolved at load

    def test_rows_do_not_share_config(self):
        ds = self.dataset()
        row = ds[0]
        row["config"]["nested"]["a"] = 99
        self.assertEqual(ds[1]["config"]["nested"]["a"], 1)
        self.assertEqual(ds[0]["config"]["nested"]["a"], 1)

    def test_data_files_and_max_samples(self):
        self.assertEqual(single_data_file(OmegaConf.create([str(self.env_yaml)])), str(self.env_yaml))
        self.assertEqual(len(AgenticDatasetV1(data_files=OmegaConf.create([str(self.env_yaml)]),
                                              config=self.data_config)), 36)
        with self.assertRaises(ValueError):
            single_data_file([str(self.env_yaml), str(self.env_yaml)])
        with self.assertRaises(TypeError):
            single_data_file(3)
        with self.assertRaises(ValueError):
            self.dataset(max_samples=5)
        self.assertEqual(len(self.dataset(max_samples=-1, tokenizer=None, processor=None)), 36)

    def test_verl_creates_it_from_the_config(self):
        """verl/verl/trainer/ppo/utils.py:110-137 with data.custom_cls as in the q38 configs (pkg:// path)."""
        from verl.trainer.ppo.utils import create_rl_dataset

        data_config = OmegaConf.create({"custom_cls": {"path": "pkg://vagen.q38.dataset", "name": "AgenticDatasetV1"},
                                        "seed": 20261022, "shuffle": True})
        ds = create_rl_dataset(str(self.env_yaml), data_config, None, None, is_train=True, max_samples=-1)
        self.assertIsInstance(ds, AgenticDatasetV1)
        self.assertEqual(len(ds), 36)

    def test_collate_fetch_and_dispatch(self):
        """collate_fn -> _fetch_one_gen_batch -> _next_train_batch -> _submit_batch_to_rollout -> the worker."""
        from tensordict import NonTensorData, NonTensorStack
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl.trainer.ppo.utils import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        ds = self.dataset()
        # trainer_base.py:770-777 (num_workers 0 here: no forked loader processes on the login node)
        loader = StatefulDataLoader(dataset=ds, batch_size=1, num_workers=0, drop_last=True, collate_fn=collate_fn,
                                    sampler=create_rl_sampler(self.data_config, ds))
        it = iter(loader)

        batch_dict, chunk = fetch_one_gen_batch(it)
        self.assertEqual(len(batch_dict["raw_prompt"]), 1)  # raw_prompt is an object array of one empty list
        self.assertEqual(list(batch_dict["raw_prompt"][0]), [])
        self.assertEqual(len(chunk), 1)
        for key in DUMMY_TENSOR_KEYS:
            self.assertNotIn(key, chunk.keys())

        batch = next_train_batch(it, 24, global_steps=3)
        self.assertEqual(len(batch), 24)
        keys = set(batch.keys())
        self.assertEqual(keys, ROW_KEYS | {"uid", "global_steps"})
        for key in ROW_KEYS | {"uid"}:
            self.assertIsInstance(batch.get(key), NonTensorStack, key)
        self.assertIsInstance(batch.get("global_steps"), NonTensorData)

        # trainer_base.py:1569-1577: what the async trainers persist with the prompt key
        persisted = batch.select(*[k for k in batch.keys() if not isinstance(batch.get(k), NonTensorData)])
        self.assertEqual(set(persisted.keys()), ROW_KEYS | {"uid"})

        prompts = worker_prompts(batch)
        uids = [p["uid"] for p in prompts]
        self.assertEqual(len(set(uids)), 24)
        for p in prompts:
            # trajectory keys are {uid}_{session}_{i} and resume splits them at the first "_"
            # (agent_loop_tq.py:206; trainer_base.py:956,978)
            self.assertNotIn("_", p["uid"])
            self.assertEqual(p["raw_prompt"], [])
            self.assertEqual(p["global_steps"], 3)
            self.assertEqual(p["agent_name"], "gym_agent")
            self.assertEqual(ds[p["index"]]["seed"], p["seed"])
            self.assertEqual(ds[p["index"]]["config"], p["config"])
        indices = [p["index"] for p in prompts]
        self.assertEqual(len(set(indices + [int(batch_dict["index"][0])])), 25)  # shuffled, no repeats
        self.assertNotEqual(sorted(indices), indices)  # data.shuffle true: a random order
        # agent_loop.py:1221-1226 (get_trajectory_info) reads batch["index"][i] as a plain int
        self.assertEqual([batch["index"][i] for i in range(24)], indices)

        # resume: the dataloader state restores the position (trainer_base.py:924-928)
        state = loader.state_dict()
        after = [int(next(it)["index"][0]) for _ in range(3)]
        restored = StatefulDataLoader(dataset=ds, batch_size=1, num_workers=0, drop_last=True,
                                      collate_fn=collate_fn, sampler=create_rl_sampler(self.data_config, ds))
        restored.load_state_dict(state)
        rit = iter(restored)
        self.assertEqual([int(next(rit)["index"][0]) for _ in range(3)], after)

    @unittest.skipUnless(V4_TRAIN_ENV.is_file(), "8B V4 run not on this machine")
    def test_8b_v4_env_yaml(self):
        """The 8B main line V4 env yaml (read only) loads as 10,798 V1 rows of 12 turns x 384 tokens."""
        ds = AgenticDatasetV1(data_files=str(V4_TRAIN_ENV), config=self.data_config)
        self.assertEqual(len(ds), 10798)
        for idx in (0, 5000, 10797):
            row = ds[idx]
            self.assertEqual(set(row), ROW_KEYS)
            self.assertEqual((row["max_turns"], row["response_length_per_turn"]), (12, 384))
            self.assertEqual(row["index"], idx)
            self.assertEqual(row["seed"], idx)  # the seed_list is 0..10797: the seed is the task row
            self.assertEqual(row["env_name"], "ActiveSpatial")


def stop_marked_processes(marker: str) -> list:
    """SIGTERM this user's processes whose environment holds ``Q38_T10_RAY_MARKER=<marker>`` (not this one)."""
    import signal

    needle = f"Q38_T10_RAY_MARKER={marker}".encode()
    stopped = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as f:
                if needle in f.read().split(b"\0"):
                    os.kill(int(entry), signal.SIGTERM)
                    stopped.append(int(entry))
        except (OSError, ProcessLookupError):
            continue
    return stopped


# Ray's session directory: short (AF_UNIX socket paths are at most 107 bytes) and outside /tmp, removed afterwards
RAY_TMP = os.environ.get("Q38_RAY_TMP_B1", "/mnt/umm/users/yinbaiqiao/q38/rt_b1")
RAY_START_TIMEOUT_S = 300


class TransferQueueRoundTripTest(unittest.TestCase):
    """The prompt fields the async trainers persist with the prompt key come back intact for the resume re-issue.

    Put as trainer_base.py:1566-1577, read back and dispatched again as trainer_base.py:971-987; a real
    TransferQueue on a local Ray with num_cpus=2.

    Skipped when Ray cannot start, as the other q38 TransferQueue tests (test_sampler.py real_tq, test_agent_loop.py
    tq_ray): on the fujian login node (2026-11-18) every local start failed because the raylet waits only 15 s for
    the dashboard agent's port files ("Timed out waiting for file .../metrics_agent_port") and the agent's Python
    start-up from /mnt/umm took 22-34 s there, so the raylet never registered.  The test then needs a node where Ray
    starts (a one-card job).
    """

    @classmethod
    def setUpClass(cls):
        import shutil
        import signal

        import ray
        import transfer_queue as tq

        os.environ.setdefault("R1_RENDER_URL", "http://127.0.0.1:1/render")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env_yaml = Path(cls.tmp.name) / "train_env.yaml"
        cls.env_yaml.write_text(ENV_YAML)
        try:
            os.makedirs(RAY_TMP, exist_ok=True)
        except OSError as e:
            cls.tmp.cleanup()
            raise unittest.SkipTest(f"no Ray temp directory {RAY_TMP}: {e!r}")

        def _timeout(*_):
            raise TimeoutError(f"Ray / TransferQueue start-up took more than {RAY_START_TIMEOUT_S} s")

        # address="local": a cluster of its own, never one another process on the login node started.  Every process
        # this Ray starts inherits the marker: a failed start leaves its dashboard agent behind (parent 1), which is
        # then found by the marker and stopped.
        marker = f"q38-t10-{uuid.uuid4().hex}"
        os.environ["Q38_T10_RAY_MARKER"] = marker
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(RAY_START_TIMEOUT_S)
        try:
            ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False,
                     object_store_memory=200 * 1024 * 1024, _temp_dir=RAY_TMP)
            tq.init(OmegaConf.create({"backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {
                "num_data_storage_units": 1, "total_storage_size": 1000}}}))
        except Exception as e:  # any start-up failure means: no local Ray on this node
            signal.alarm(0)
            try:
                ray.shutdown()
            finally:
                stop_marked_processes(marker)
                shutil.rmtree(RAY_TMP, ignore_errors=True)
                cls.tmp.cleanup()
            raise unittest.SkipTest(f"local Ray could not start on this node: {type(e).__name__}: {str(e)[:160]}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            os.environ.pop("Q38_T10_RAY_MARKER", None)
        cls.ray, cls.tq, cls.marker = ray, tq, marker

    @classmethod
    def tearDownClass(cls):
        import shutil

        try:
            cls.tq.close()
        finally:
            cls.ray.shutdown()
            stop_marked_processes(cls.marker)
            shutil.rmtree(RAY_TMP, ignore_errors=True)
            cls.tmp.cleanup()

    def test_persisted_prompt_fields(self):
        from tensordict import NonTensorData, NonTensorStack
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl.trainer.ppo.utils import create_rl_sampler
        from verl.utils import tensordict_utils as tu
        from verl.utils.dataset.rl_dataset import collate_fn

        data_config = OmegaConf.create({"seed": 7, "shuffle": True})
        ds = AgenticDatasetV1(data_files=str(self.env_yaml), config=data_config)
        loader = StatefulDataLoader(dataset=ds, batch_size=1, num_workers=0, drop_last=True, collate_fn=collate_fn,
                                    sampler=create_rl_sampler(data_config, ds))
        batch = next_train_batch(iter(loader), 6, global_steps=2)
        uids = list(batch["uid"])
        tags = [{"is_prompt": True, "status": "pending", "global_steps": 2} for _ in uids]
        persisted = batch.select(*[k for k in batch.keys() if not isinstance(batch.get(k), NonTensorData)])
        self.tq.kv_batch_put(keys=uids, partition_id="train", tags=tags, fields=persisted)

        listed = self.tq.kv_list("train")["train"]
        self.assertEqual({k for k, tag in listed.items() if tag.get("is_prompt")}, set(uids))

        back = self.tq.kv_batch_get(keys=uids, partition_id="train")
        self.assertEqual(set(back.keys()), ROW_KEYS | {"uid"})
        for key in ROW_KEYS | {"uid"}:
            self.assertIsInstance(back.get(key), NonTensorStack, key)
        tu.assign_non_tensor_data(back, "global_steps", 5)  # the re-issue's new dispatch step (trainer_base.py:984)
        sent, again = worker_prompts(batch), worker_prompts(back)
        self.assertEqual(len(again), 6)
        for a, b in zip(sent, again):
            self.assertEqual(b["global_steps"], 5)
            for key in ROW_KEYS | {"uid"}:
                self.assertEqual(b[key], a[key], key)
        self.tq.kv_clear(keys=uids, partition_id="train")


if __name__ == "__main__":
    unittest.main()
