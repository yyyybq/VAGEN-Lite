"""Finite rendering-only A/B/C run with immutable attempts and verified resume."""
import argparse
import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path
import traceback
import numpy as np
from PIL import Image
from data_gen.active_spatial_pipeline.render_utils import SceneRenderer, RenderConfig
from .render_paired_bank import render
from .restore_diagnostic_scene import OUT, SCENE, _sha256
from .build_position_requests import request_hash, ROOT
from .image_contract_v2 import image_stats, pose_from_forward


def build_requests():
    ref = ROOT/'data_gen/active_spatial_qa/artifacts/position_repair_v5_20261002'
    historical = json.loads((ref/'01_A_upstream/request.json').read_text())
    png = ref/'01_A_upstream/A_upstream_init_0003.png'
    assets = json.loads((ref/'asset_provenance.json').read_text())['validation']['files']
    init = historical['rows'][0]
    common = {'scene_id':SCENE, 'width':historical['width'], 'height':historical['height'],
              'asset_identity':assets,'renderer_version':'position_diagnostic_v6',
              'reference':{'image':str(png),'sha256':_sha256(png),'job_id':'pt-1ic3guxb',
                           'request':str(ref/'01_A_upstream/request.json'),
                           'inspection':'upright pink wardrobe doors, floor and wall; directly inspected RGB',
                           'asset_provenance':str(ref/'asset_provenance.json')}}
    def save(name, route, rows):
        req=dict(common,route=route,rows=rows); req['request_sha256']=request_hash(req)
        path=OUT/'requests'/name; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and json.loads(path.read_text())!=req:
            raise RuntimeError('refusing to replace request: '+str(path))
        if not path.exists(): path.write_text(json.dumps(req,indent=2)+'\n')
    save('01_A_upstream.json','upstream',[init])
    save('02_B_qa_exact_same_pose.json','qa',[init])
    bank=ROOT/'data_gen/active_spatial_qa/artifacts/qa_v2_runtime_canary_20260914/oracle_bank/manifest.jsonl'
    rows=[json.loads(x) for x in bank.read_text().splitlines() if x.strip()]
    parent=rows[0]['parent_goal_id']; crows=[]
    selected=[r for r in rows if r['parent_goal_id']==parent and r['state_id']!='positive_region_1']
    selected += [r for r in rows if r['sample_id'] in ('255f8e714bcb656b93e6','ed9440ce5818ebcae771','269c12b109b87274fb6')]
    for row in selected:
        for axis in ('old','fixed'):
            rec=copy.deepcopy(row); p=np.asarray(row['state_pose_c2w'])
            rec.update(original_sample_id=row['sample_id'],sample_id=row['sample_id']+'_'+axis,
                       control={'factor':'camera_axis_convention','pair':row['sample_id'],'level':axis})
            if axis=='fixed': rec['state_pose_c2w']=pose_from_forward(p[:3,3],p[:3,2]).tolist()
            crows.append(rec)
    for row in [init]+[r for r in rows if r['parent_goal_id']==parent]:
        rec=copy.deepcopy(rows[0]); pose=np.asarray(init['state_pose_c2w']).copy()
        pose[:3,3]=np.asarray(row['state_pose_c2w'])[:3,3]
        rec.update(original_sample_id=row['sample_id'],sample_id='translation_'+row['sample_id'],
                   state_pose_c2w=pose.tolist(),state_id=row['state_id'],
                   control={'factor':'translation_only','rotation_reference':init['sample_id']})
        crows.append(rec)
    save('03_C_controlled.json','qa',crows)
    print(json.dumps({'C_rows':len(crows),'output':str(OUT)}))


def effective_request(request, assets):
    paths=['data_gen/active_spatial_qa/position_render_run.py','data_gen/active_spatial_qa/render_paired_bank.py',
           'data_gen/active_spatial_pipeline/render_utils.py','vagen/envs/active_spatial/render/unified_renderer.py',
           'vagen/envs/active_spatial/render/gs_render_local.py']
    return {'request':request,'asset_identity':assets,'render_code':{p:_sha256(ROOT/p) for p in paths}}


def validate_outputs(state, expected_ids=None):
    if not state.get('ok') or not state.get('artifacts'): return False
    try:
        if expected_ids is not None:
            artifact_ids = [str(entry['sample_id']) for entry in state['artifacts']]
            if artifact_ids != [str(x) for x in expected_ids]:
                return False
        for entry in state['artifacts']:
            path=Path(entry['path'])
            if _sha256(path)!=entry['sha256']: return False
            with Image.open(path) as im:
                im.load()
                if list(im.size)!=entry['size']: return False
        return True
    except (OSError, ValueError, KeyError, TypeError): return False


async def execute(request,dest,assets):
    if request.get('request_sha256')!=request_hash(request): raise ValueError('immutable request hash mismatch')
    if request['asset_identity']!=assets: raise ValueError('assets differ from pinned RGB reference identity')
    effective=effective_request(request,assets)
    key=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
    dest.mkdir(parents=True,exist_ok=True)
    expected_ids = [str(row['sample_id']) for row in request['rows']]
    for prior in sorted(dest.glob('attempt_*/status.json'),reverse=True):
        try:
            state=json.loads(prior.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if state.get('effective_sha256')==key and validate_outputs(state, expected_ids): return state
    attempt=dest/('attempt_'+str(time.time_ns())); attempt.mkdir()
    state={'ok':False,'effective_sha256':key,'attempt':str(attempt),'expected_sample_ids':expected_ids,'artifacts':[]}
    (attempt/'actual_request.json').write_text(json.dumps(effective,indent=2)+'\n')
    try:
        if request['route']=='upstream':
            cfg=RenderConfig(gs_root=str(OUT/'assets'),render_backend='local',image_width=request['width'],image_height=request['height'],gpu_device=0)
            async with SceneRenderer(cfg) as renderer:
                await renderer.set_scene(SCENE)
                for row in request['rows']:
                    K=np.asarray(row['camera']['intrinsics'],dtype=np.float32)
                    pose=np.asarray(row['state_pose_c2w'],dtype=np.float32)
                    image=await renderer.render_image(K,pose)
                    if image is None: raise RuntimeError('upstream RGB is None')
                    image.save(attempt/(row['sample_id']+'.png'))
                    (attempt/(row['sample_id']+'_renderer_call.json')).write_text(json.dumps({'K':K.tolist(),'c2w':pose.tolist(),'w2c':np.linalg.inv(pose).tolist(),'width':request['width'],'height':request['height']},indent=2))
            paths=[attempt/(r['sample_id']+'.png') for r in request['rows']]
        elif request['route']=='qa':
            bank=attempt/'input.jsonl'; bank.write_text(''.join(json.dumps(r)+'\n' for r in request['rows']))
            summary=await render(argparse.Namespace(bank=str(bank),output_dir=str(attempt),limit=0,backend='local',gs_root=str(OUT/'assets'),renderer_url='',gpu_device=0,width=request['width'],height=request['height']))
            if summary['errors']: raise RuntimeError('QA render errors; see attempt errors.jsonl')
            records=[json.loads(x) for x in (attempt/'manifest.jsonl').read_text().splitlines()]
            if [r['sample_id'] for r in records]!=[r['sample_id'] for r in request['rows']]: raise RuntimeError('output IDs mismatch')
            paths=[Path(r['render']['image_path']) for r in records]
        else: raise ValueError('unknown render route')
        if len(paths)!=len(request['rows']): raise RuntimeError('output count mismatch')
        for row,path in zip(request['rows'],paths):
            with Image.open(path) as im:
                im.load()
                if im.size!=(request['width'],request['height']): raise RuntimeError('output resolution mismatch')
                state['artifacts'].append({'sample_id':row['sample_id'],'path':str(path),'sha256':_sha256(path),'size':list(im.size),'rgb_stats':image_stats(im)})
        state['ok']=True
        if not validate_outputs(state, expected_ids): raise RuntimeError('output validation failed')
    except Exception:
        state['ok']=False; state['error']=traceback.format_exc()
        (attempt/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        raise
    (attempt/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    return state


async def _run():
    assets=json.loads((OUT/'asset_provenance.json').read_text())['validation']['files']
    (OUT/'worker_ready.json').write_text(json.dumps({'ready':True,'started':time.time()}))
    from .position_evidence import classify_candidates,make_report
    classify_candidates(OUT)
    states={}
    for batch in sorted((OUT/'requests').glob('*.json')):
        request=json.loads(batch.read_text())
        if batch.stem.startswith('03'):
            a=states['01_A_upstream']['artifacts'][0]; b=states['02_B_qa_exact_same_pose']['artifacts'][0]
            av=np.asarray(Image.open(a['path']).convert('RGB'),dtype=float); bv=np.asarray(Image.open(b['path']).convert('RGB'),dtype=float)
            ref=request['reference']; historical=np.asarray(Image.open(ref['image']).convert('RGB'),dtype=float)
            comparison={'A_B_max_abs':float(abs(av-bv).max()),'A_B_mean_abs':float(abs(av-bv).mean()),
                        'A_reference_max_abs':float(abs(av-historical).max()),'reference_sha256_matches':_sha256(Path(ref['image']))==ref['sha256']}
            (OUT/'AB_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            if comparison['A_B_max_abs']>1 or not a['rgb_stats']['nonblank'] or not comparison['reference_sha256_matches'] or comparison['A_reference_max_abs']>1:
                raise RuntimeError('A/B or trusted RGB reference replay failed; C blocked')
        states[batch.stem]=await execute(request,OUT/batch.stem,assets)
    (OUT/'render_results.json').write_text(json.dumps(states,indent=2)+'\n')
    make_report(OUT,states)
    (OUT/'worker_finished.json').write_text(json.dumps({'ok':True,'finished':time.time()}))


async def main():
    runs = OUT/'worker_runs'
    runs.mkdir(parents=True, exist_ok=True)
    state_path = runs/('run_'+str(time.time_ns())+'.json')
    state = {'ok': False, 'phase': 'running', 'started': time.time()}
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    try:
        await _run()
        state.update(ok=True, phase='succeeded')
    except BaseException:
        state.update(ok=False, phase='failed', error=traceback.format_exc())
        raise
    finally:
        state['finished'] = time.time()
        state_path.write_text(json.dumps(state, indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args=parser.parse_args()
    if args.prepare: build_requests()
    else: asyncio.run(main())
