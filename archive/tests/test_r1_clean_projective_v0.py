"""CPU tests for the endpoint adapter and policy/audit boundary."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class EndpointTests(unittest.TestCase):
    def test_initialized_scheduler_not_rebuilt(self):
        tree=ast.parse((ROOT/'vagen/r1_clean_projective_ppo.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FixedEndpointTrainer')
        with tempfile.TemporaryDirectory() as d:
            ck=Path(d)/'global_step_250';ck.mkdir();(ck/'COMPLETE').touch()
            optim=SimpleNamespace(total_training_steps=700)
            cfg=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(optim=optim)),
                                critic=SimpleNamespace(optim=optim),trainer=SimpleNamespace(resume_mode='disable',default_local_dir=d))
            seen={}
            class Base:
                def fit(self):
                    seen.update(stop=self.total_training_steps,scheduler=self.config.actor_rollout_ref.actor.optim.total_training_steps)
            namespace={'RayPPOTrainer':Base,'os':os,'Path':Path}
            exec(compile(ast.Module(body=[cls],type_ignores=[]),'<adapter>','exec'),namespace)
            t=namespace['FixedEndpointTrainer']();t.config=cfg;t.total_training_steps=700
            with patch.dict(os.environ,{'R1_PPO_ENDPOINT':'250'}):t.fit()
            self.assertEqual(seen,{'stop':250,'scheduler':700})

    def test_no_oracle_in_policy_rows(self):
        frozen=ROOT/'exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1'
        if not frozen.exists():self.skipTest('control-host immutable data not mounted')
        rows=[json.loads(l) for l in (frozen/'train.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),210);self.assertEqual(len({r['task_id'] for r in rows}),210)
        for r in rows:
            self.assertEqual(r['task_type'],'projective_relations')
            self.assertFalse({'actions','sample_target','target_pose','reachability_construction'} & r.keys())
            # Region metadata is used by the existing reward, not sent to the
            # policy. The live preflight compares actual text with/without it.
            self.assertIn('sample_point',r['target_region'])
            self.assertEqual(r['canonical_task_metric_version'],'canonical_spatial_task_h1_v1')

if __name__=='__main__':unittest.main()
