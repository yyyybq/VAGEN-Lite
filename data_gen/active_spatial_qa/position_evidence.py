"""Scene-grounded position evidence and RGB review, without any VLM."""
import argparse
import csv
import json
import math
import itertools
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .generate_paired_qa import _load_scene_context,generate
from .image_contract_v2 import classify_scene_pose,image_stats

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'data_gen/active_spatial_qa/artifacts/qa_v2_runtime_canary_20260914/oracle_bank/manifest.jsonl'
SOURCE=ROOT/'data_gen/active_spatial_pipeline/output_100scenes/scenes/0003_839989/data.jsonl'


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def classify_candidates(out):
    context=_load_scene_context(str(out/'assets'),'0003_839989')
    if context is None: raise RuntimeError('restored scene could not load collision geometry')
    detector,layout=context
    coord={'status':detector.structure_convention_status,'structure_y_sign':detector.structure_y_sign,
           'alignment_scores':detector.structure_alignment_scores,'label_evidence_count':detector.structure_alignment_evidence_count,
           'relative_margin':detector.structure_alignment_relative_margin,'room_polygons':layout.room_polys,
           'wall_segments':layout.wall_segments,'asset_provenance':str(out/'asset_provenance.json')}
    (out/'coordinate_evidence.json').write_text(json.dumps(coord,indent=2)+'\n')
    rows=read_rows(OLD); records=[]
    for row in rows:
        pose=np.asarray(row['state_pose_c2w']); c=classify_scene_pose(pose[:3,3],detector,layout)
        state=row['state_id']
        source='original_sample_point' if state=='positive_target' else ('QA_region' if state.startswith('positive_region') else 'QA_offset' if state=='negative_position_offset' else 'QA_flip')
        record={'sample_id':row['sample_id'],'parent_goal_id':row['parent_goal_id'],'state_id':state,'source_category':source,
                'position':pose[:3,3].tolist(),'classification':c,'historical_GT':row['private_answer'],
                'qa_sft_allowed':False,'reason':'angular/pixel definition unresolved; invalid observations not training No'}
        records.append(record)
    for index,row in enumerate(read_rows(SOURCE)[:6]):
        pose=np.asarray(row['init_camera']['extrinsics'])
        records.append({'sample_id':'source_init_line_'+str(index),'source_category':'original_init','position':pose[:3,3].tolist(),
                        'classification':classify_scene_pose(pose[:3,3],detector,layout),'qa_sft_allowed':False})
    (out/'position_classifications.json').write_text(json.dumps(records,indent=2)+'\n')
    with (out/'position_classifications.csv').open('w') as f:
        fields=['sample_id','state_id','source_category','position','status','reasons','height','room_inside','wall_distance','collision']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in records:
            c=r['classification']; p=r['position']
            writer.writerow({'sample_id':r['sample_id'],'state_id':r.get('state_id','init'),'source_category':r['source_category'],
                             'position':p,'status':c['status'],'reasons':c['reasons'],'height':p[2],
                             'room_inside':c.get('room_polygon',{}).get('contains'),'wall_distance':c.get('room_polygon',{}).get('wall_distance'),
                             'collision':c.get('collision')})
    summary={}
    for category in sorted({r['source_category'] for r in records}):
        selected=[r for r in records if r['source_category']==category]
        summary[category]={'n':len(selected),'statuses':dict(Counter(r['classification']['status'] for r in selected)),
                           'reasons':dict(Counter(x for r in selected for x in r['classification']['reasons']))}
    (out/'source_failure_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Exercise production generator with restored scene, preserving all failed branches.
    # Keep every generator run versioned; this avoids confusing a pre-contract
    # artifact with the current scene-geometry contract.
    generate(argparse.Namespace(input=str(SOURCE),output_dir=str(out/'generated_v4_geometry'),split='train',split_scenes='',task_type='screen_occupancy',limit=30,scene_root=str(out/'assets')))
    draw_overhead(out,records,detector,layout,rows[0]['complete_goal']['target_object'])


def draw_overhead(out,records,detector,layout,target):
    points=[np.asarray(r['position'][:2]) for r in records]+[np.asarray(p) for poly in layout.room_polys for p in poly]
    xy=np.asarray(points); lo=xy.min(0)-1; hi=xy.max(0)+1
    scale=1000/max(hi-lo); im=Image.new('RGB',(1720,1140),'white'); d=ImageDraw.Draw(im)
    def project(p):
        q=(np.asarray(p[:2])-lo)*scale
        return (int(q[0]+40),int(1080-q[1]))
    for poly in layout.room_polys:
        d.polygon([project(p) for p in poly],fill=(234,244,233),outline=(30,70,30))
    for box in detector.object_boxes:
        a,b=project(box.min_point),project(box.max_point)
        d.rectangle((min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])),outline=(160,160,160))
    for a,b in layout.wall_segments: d.line((project(a),project(b)),fill=(20,20,20),width=3)
    tc=project(target['center']); d.ellipse((tc[0]-8,tc[1]-8,tc[0]+8,tc[1]+8),fill='purple'); d.text((tc[0]+9,tc[1]),'wardrobe id 23',fill='purple')
    for i,r in enumerate(records):
        p=project(r['position']); col='green' if r['classification']['status']=='legal_indoor' else 'red'
        if r['source_category']=='original_init': col='blue'
        d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=col)
        d.text((p[0]+5,p[1]+(i%3)*10),str(i),fill=col)
        d.text((1100,35+i*28),f"{i:02d} {r['sample_id']} {r['source_category']}",fill=col)
        d.text((1100,48+i*28),r['classification']['status'],fill=col)
    d.text((20,15),'Actual room polygons / walls / expanded collision objects / world XY cameras',fill='black')
    im.save(out/'camera_overhead.png')


def make_report(out,states):
    context=_load_scene_context(str(out/'assets'),'0003_839989')
    panels=[]; evidence=[]
    for name,state in states.items():
        request=json.loads((Path(state['attempt'])/'actual_request.json').read_text())['request']
        for row,artifact in zip(request['rows'],state['artifacts']):
            pose=np.asarray(row['state_pose_c2w']); c=classify_scene_pose(pose[:3,3],*context)
            img=Image.open(artifact['path']).convert('RGB')
            panel=Image.new('RGB',(320,370),'white'); panel.paste(img.resize((320,320)),(0,50))
            d=ImageDraw.Draw(panel); d.text((3,3),row['sample_id'],fill='black'); d.text((3,18),str(row.get('control',{}).get('factor',name)),fill='black')
            d.text((3,33),c['status'],fill='black'); panels.append(panel)
            entry={'sample_id':row['sample_id'],'original_sample_id':row.get('original_sample_id'),
                   'control':row.get('control'),'classification':c,'artifact':artifact,'historical_GT':row.get('private_answer'),
                   'rgb_task_evidence':'UNVERIFIED: manual review required; nonblank is not target visibility','qa_sft_allowed':False}
            if row.get('complete_goal'):
                params=row['complete_goal']['target_region']['params']; distance=np.linalg.norm(pose[:2,3]-np.array(params['object_center'])[:2])
                angular=2*math.atan(params['object_height']/(2*distance))/math.radians(params['fov_vertical'])
                obj=row['complete_goal']['target_object']; corners=np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
                cam=(corners-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T; front=cam[:,2]>1e-6
                pixel=0.0
                if front.any():
                    K=np.array(row['camera']['intrinsics']); uv=cam[front]@K.T; uv=uv[:,:2]/uv[:,2:3]
                    pixel=max(0,min(request['height']-1,uv[:,1].max())-max(0,uv[:,1].min()))/request['height']
                entry['occupancy']={'angular_declared_FOV_fraction':angular,'pixel_clipped_3D_bbox_height_fraction':pixel,
                                    'target_fraction':params['occupancy_ratio'],'declared_FOV_degrees':params['fov_vertical'],
                                    'definition_conflict_unresolved':True}
            evidence.append(entry)
    sheet=Image.new('RGB',(320*4,370*math.ceil(len(panels)/4)),'white')
    for i,p in enumerate(panels): sheet.paste(p,((i%4)*320,(i//4)*370))
    sheet.save(out/'ABC_contact_sheet.png')
    (out/'render_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    statistics=json.loads((out/'source_failure_statistics.json').read_text())
    lookup={r['sample_id']:r for r in json.loads((out/'position_classifications.json').read_text())}
    # Old RGB rates cover the full 30 candidates, including the 3 formerly filtered frames.
    old_rgb=ROOT/'data_gen/active_spatial_qa/artifacts/act2qa_real_canary_20260919T0618/rgb/manifest.jsonl'
    by_source=defaultdict(list)
    for row in read_rows(old_rgb):
        path=row['public_observation']['image_path']; category=lookup[row['sample_id']]['source_category']
        with Image.open(path) as img: stats=image_stats(img)
        by_source[category].append({'id':row['sample_id'],'nonblank':stats['nonblank']})
    for category,values in by_source.items():
        statistics[category]['historical_RGB']={'n':len(values),'spatial_nonblank':sum(x['nonblank'] for x in values),
                                               'target_judgability':'not equivalent to nonblank; see manual review'}
    (out/'source_failure_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
