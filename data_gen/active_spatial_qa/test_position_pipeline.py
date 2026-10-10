import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, AsyncMock
import numpy as np
from PIL import Image
from .generate_paired_qa import generate
from .finalize_act2qa import freeze, _pose_close
from .position_render_run import validate_outputs,effective_request, execute, main
from . import position_render_run as position_worker
from .build_position_requests import request_hash
from .restore_diagnostic_scene import _sha256,_validate_scene
from .test_qa_contract import _fixture
from .image_contract_v2 import pose_from_forward


class PositionPipelineTests(unittest.TestCase):
    def test_pose_match_rejects_malformed_and_nonfinite(self):
        pose = np.eye(4).tolist()
        self.assertTrue(_pose_close(pose, pose))
        self.assertFalse(_pose_close(pose[:3], pose))
        invalid = np.eye(4); invalid[0, 0] = np.nan
        self.assertFalse(_pose_close(invalid.tolist(), invalid.tolist()))

    def test_real_geometry_generation_to_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); scene=root/'assets'/'fixture_scene'; scene.mkdir(parents=True)
            labels=[]
            for i in range(24):
                x,y=2+i%6,2+i//6
                corners=[{'x':a,'y':b,'z':c} for a,b,c in itertools.product((x,x+.1),(y,y+.1),(0,2))]
                labels.append({'label':'chair','ins_id':str(i),'bounding_box':corners})
            (scene/'labels.json').write_text(json.dumps(labels))
            (scene/'structure.json').write_text(json.dumps({'rooms':[{'profile':[[0,0],[8,0],[8,8],[0,8]]}]}))
            a=np.zeros((512,512,3),dtype=np.uint8); a[::2]=255
            image=root/'detail.png'; Image.fromarray(a).save(image)
            item=_fixture(); item['states']=[]
            for sid,pos in [('legal_negative',[1,1,1.5]),('outdoor_no_collision',[100,100,1.5]),('old_wrong_reference',[-1,-1,1.5])]:
                pose=np.eye(4); pose[:3,3]=pos
                item['states'].append({'state_id':sid,'c2w':pose.tolist(),'image_path':str(image)})
            # Reproduce the historical diagnostic's erroneous inverse of an
            # upstream c2w, with its actual center/forward (0003_839989 line 0).
            reference = pose_from_forward([4.994601935252783, 0.1891533614114399, 1.6],
                                          [0.9818461825901439, 0.17312597289625198, -0.07749497559124885])
            item['states'].append({'state_id':'historical_inverted_c2w',
                                   'c2w':np.linalg.inv(reference).tolist(), 'image_path':str(image)})
            src=root/'source.jsonl'; src.write_text(json.dumps(item)+'\n'); out=root/'generated'
            generate(argparse.Namespace(input=str(src),output_dir=str(out),split='test',split_scenes='',task_type='',limit=0,scene_root=str(root/'assets')))
            rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['scene_pose_legality']['status'],'legal_indoor')
            self.assertEqual(rows[0]['private_answer'],'No')
            for r in rows[1:3]:
                self.assertIn('outside_room_polygon',r['scene_pose_legality']['reasons'])
                self.assertEqual(r['scene_pose_legality']['collision']['type'],'none')
                self.assertIsNone(r['private_answer'])
            self.assertIsNone(rows[3]['private_answer'])
            self.assertIn('height_out_of_range', rows[3]['scene_pose_legality']['reasons'])
            # Even forged legality metadata cannot bypass freeze's reclassification.
            for r in rows:
                r['scene_pose_legality']={'status':'legal_indoor'}; r['label_validity']='valid'
                r['render']={'pose_c2w':r['state_pose_c2w'],'image_path':str(image)}
            rendered=root/'rendered.jsonl'; rendered.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            dest=root/'frozen'; dest.mkdir()
            stats=freeze(argparse.Namespace(output_dir=str(dest),rendered_bank=str(rendered),errors='',scene_root=str(root/'assets')))
            self.assertEqual(stats['eligible'],1)
            audit=json.loads((dest/'rgb_diagnostics.json').read_text())
            self.assertFalse(audit[3]['eligible'])
            self.assertIn('height_illegal', audit[3]['scene_pose_legality']['reasons'])
            stats=freeze(argparse.Namespace(output_dir=str(dest),rendered_bank=str(rendered),errors='',scene_root=''))
            self.assertEqual(stats['eligible'],0)

    def test_hash_provenance_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); image=root/'image.png'; Image.new('RGB',(8,8),'red').save(image)
            state={'ok':True,'artifacts':[{'path':str(image),'sha256':_sha256(image),'size':[8,8]}]}
            self.assertTrue(validate_outputs(state))
            self.assertFalse(validate_outputs(state, ['wrong-id']))
            state['artifacts'][0]['sample_id'] = 'one'
            self.assertTrue(validate_outputs(state, ['one']))
            self.assertFalse(validate_outputs(state, ['one', 'missing']))
            self.assertFalse(validate_outputs(dict(state,ok=False)))
            Image.new('RGB',(8,8),'blue').save(image)
            self.assertFalse(validate_outputs(state))
            a=effective_request({'pose':[[1]],'K':[[1]],'width':8,'height':8},{'ply':'one'})
            b=effective_request({'pose':[[1]],'K':[[1]],'width':8,'height':8},{'ply':'two'})
            self.assertNotEqual(a,b)
            (root/'structure.json').write_text('{"rooms":[{}]}'); (root/'labels.json').write_text('[{}]'); (root/'3dgs_compressed.ply').write_bytes(b'ply\n')
            provenance=_validate_scene(root)
            self.assertEqual(provenance['status'],'valid')
            path=root/'provenance.json'; path.write_text(json.dumps(provenance))
            self.assertEqual(json.loads(path.read_text())['files']['3dgs_compressed.ply']['sha256'],hashlib.sha256(b'ply\n').hexdigest())


class WorkerAttemptTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_records_failure_and_propagates_exception(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(position_worker, 'OUT', Path(td)), \
                 patch.object(position_worker, '_run',
                       AsyncMock(side_effect=RuntimeError('early request failure'))):
                with self.assertRaisesRegex(RuntimeError, 'early request failure'):
                    await main()
            state = json.loads(next((Path(td)/'worker_runs').glob('*.json')).read_text())
            self.assertFalse(state['ok'])
            self.assertEqual(state['phase'], 'failed')
            self.assertIn('early request failure', state['error'])

    async def test_failed_attempt_retries_without_overwrite_and_success_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pose = np.eye(4).tolist()
            request = {'route': 'upstream', 'width': 8, 'height': 8,
                       'asset_identity': {'ply': 'test'}, 'rows': [
                           {'sample_id': 'one', 'state_pose_c2w': pose,
                            'camera': {'intrinsics': np.eye(3).tolist()}}]}
            request['request_sha256'] = request_hash(request)

            class FakeRenderer:
                failure = True
                calls = 0
                def __init__(self, config): pass
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
                async def set_scene(self, scene): pass
                async def render_image(self, K, pose):
                    FakeRenderer.calls += 1
                    if self.failure:
                        raise RuntimeError('injected renderer failure')
                    return Image.new('RGB', (8, 8), 'red')

            with patch.object(position_worker, 'SceneRenderer', FakeRenderer):
                with self.assertRaisesRegex(RuntimeError, 'injected renderer failure'):
                    await execute(request, root, request['asset_identity'])
                failed_path = next(root.glob('attempt_*/status.json'))
                original = failed_path.read_bytes()
                self.assertFalse(json.loads(original)['ok'])
                corrupt = root/'attempt_corrupt'; corrupt.mkdir()
                (corrupt/'status.json').write_text('invalid JSON')
                FakeRenderer.failure = False
                success = await execute(request, root, request['asset_identity'])
                self.assertTrue(validate_outputs(success, ['one']))
                self.assertEqual(failed_path.read_bytes(), original)
                cached = await execute(request, root, request['asset_identity'])
                self.assertEqual(cached['attempt'], success['attempt'])
                self.assertEqual(FakeRenderer.calls, 2)


if __name__=='__main__': unittest.main()
