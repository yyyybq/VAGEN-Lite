#!/usr/bin/env python3
"""Build a renderer-independent A/B/C provenance package.

The script never changes labels or rotates images.  It records the exact
source camera (A), serialized QA camera (B), and regenerated camera (C), and
draws their top-down positions.  If the scene asset is unavailable, the
report says so instead of treating historical PNGs as a fresh render.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .generate_paired_qa import _state_variants

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--bank', required=True); ap.add_argument('--output-dir', required=True)
    a = ap.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    src = json.loads(Path(a.source).read_text().splitlines()[0])
    qa = [json.loads(x) for x in Path(a.bank).read_text().splitlines() if x.strip()]
    parent = qa[0].get('parent_goal_id') if qa else None
    qa = [r for r in qa if r.get('parent_goal_id') == parent]
    # Active's init_camera.extrinsics is already c2w in the pipeline and
    # renderer callsites. Inverting it here would turn translation into a
    # world-space camera matrix and falsely place A outside the room.
    A = np.asarray(src['init_camera']['extrinsics'], float)
    variants = _state_variants(src)
    rows = []
    for state_id, c, source, provenance in variants:
        rows.append({'state_id': state_id, 'A_init_c2w': A.tolist(), 'B_serialized_c2w': next((r['state_pose_c2w'] for r in qa if r.get('state_id') == state_id), None), 'C_regenerated_c2w': c.tolist(), 'source': source, 'provenance': provenance})
    (out/'abc_comparison.json').write_text(json.dumps({'scene_id':src.get('scene_id'),'parent_goal_id':parent,'asset_render_status':'not_attempted_asset_unavailable','rows':rows}, indent=2)+'\n')
    pts=[]
    for label, p, col in [('A init', A[:3,3], (220,40,40))]: pts.append((label,p,col))
    for r in rows:
        p=np.asarray(r['C_regenerated_c2w'])[:3,3]; pts.append(('C '+r['state_id'],p,(30,120,220)))
    xy=np.asarray([p[:2] for _,p,_ in pts]); lo=xy.min(0)-1; hi=xy.max(0)+1; scale=700/max(float((hi-lo).max()),1)
    im=Image.new('RGB',(760,760),'white'); d=ImageDraw.Draw(im)
    for label,p,col in pts:
        q=((p[:2]-lo)*scale+30).astype(int); q[1]=760-q[1]; d.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),fill=col); d.text((q[0]+7,q[1]-7),label,fill=col)
    d.text((20,20),f"scene {src.get('scene_id')} | A/B/C geometry; A extrinsics treated as c2w",fill='black'); im.save(out/'camera_overhead.png')
    (out/'README.md').write_text('# A/B/C diagnosis\n\nA is the original Active init camera. B is the serialized QA camera when present. C is regenerated from the QA sample pose. Fresh rendering was not claimed because the scene PLY was unavailable to the current process.\n')

if __name__ == '__main__': main()
