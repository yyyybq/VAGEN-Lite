#!/usr/bin/env python3
"""Atomic, committed-source-only SCO package; never snapshot unrelated edits."""
import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--package-name',default='package_v2');p.add_argument('--freeze-name',default='frozen_v1')
    a=p.parse_args();root=a.root
    run=root/'exps/vagen_active_spatial/R1-clean-Projective-v0';final=run/a.package_name
    assert not final.exists(),'immutable package already published'
    commit=git(root,'rev-parse','HEAD').decode().strip()
    dependency_commit=git(root/'verl','rev-parse','HEAD').decode().strip()
    assert not git(root/'verl','diff','--name-only')
    historic=root/'exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_train_pair_universe_v1_20260928/code_bf14b881_source_only.tar.gz'
    assert sha(historic)=='20db60ccdcefa7146e99032da0c1baecfffe212fc232efb8dfbd50f9cff6a432'
    stage=Path(tempfile.mkdtemp(prefix='.package-',dir=run));members={};file_hashes={}
    def include(raw,only=None):
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:*') as src:
            for m in src:
                if not m.isfile() or (only and not only(m.name)):continue
                assert not m.name.startswith('/') and '..' not in Path(m.name).parts
                data=src.extractfile(m).read()
                if m.name in members:assert members[m.name][1]==data
                members[m.name]=(m,data)
    include(git(root,'archive','HEAD','vagen','scripts','examples','tools','tests','setup.py','ply_gaussian_loader.py'))
    include(git(root/'verl','archive','--prefix=verl/','HEAD'))
    include(historic.read_bytes(),lambda n:n.startswith('data_gen/active_spatial_pipeline/'))
    with tarfile.open(stage/'source.tar.gz','w:gz') as dst:
        for name,(m,data) in sorted(members.items()):
            m.uid=m.gid=20325;m.uname=m.gname='yinbaiqiao';m.mtime=0
            dst.addfile(m,io.BytesIO(data));file_hashes[name]=hashlib.sha256(data).hexdigest()
    (stage/'source_files_sha256.json').write_text(json.dumps(file_hashes,indent=2,sort_keys=True)+'\n')
    (stage/'provenance.json').write_text(json.dumps({'commit':commit,'verl_commit':dependency_commit,
        'archive_sha256':sha(stage/'source.tar.gz'),'pipeline_archive_sha256':sha(historic),
        'unrelated_worktree_edits_included':False,'frozen_input_sha256':sha(run/a.freeze_name/'SHA256SUMS'),
        'cluster':'h800','pool':'h800','renderer_gpus':1,'training_gpus':8,
        'artifact_owner':{'uid':20325,'gid':20325},'scheduler_horizon':700,'formal_stop_step':250},indent=2)+'\n')
    for mode in ('renderer','training'):
        text=f'''#!/usr/bin/env bash
set -euo pipefail
ROOT={root}
RUN={run}
export R1_PACKAGE_DIR={final}
export R1_FROZEN_DIR={run/a.freeze_name}
if [[ $(id -u) == 0 ]]; then
  exec bash "${{ROOT}}/examples/train/active_spatial/sco_run_as_artifact_owner.sh" bash "$0" owner
fi
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
umask 022
cd "${{R1_PACKAGE_DIR}}"
sha256sum -c SHA256SUMS
LAUNCH=$(mktemp -d /tmp/r1_clean_launch.XXXXXXXX)
tar -xzf source.tar.gz -C "${{LAUNCH}}" examples/train/active_spatial/sco_r1_clean_projective_entry.sh
exec bash "${{LAUNCH}}/examples/train/active_spatial/sco_r1_clean_projective_entry.sh" {mode}
'''
        (stage/f'launch_{mode}.sh').write_text(text)
    (stage/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(stage.iterdir())))
    os.rename(stage,final)
    print(json.dumps({'package':str(final),'commit':commit,'sha256':sha(final/'source.tar.gz'),'files':len(members)}))

if __name__=='__main__':main()
