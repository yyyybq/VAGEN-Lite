#!/usr/bin/env python3
"""Control-host gate: one H800 training job, only after live data PASS.

Uses the established SCO CLI, not a new API client. A durable submission intent
prevents duplicate jobs after a control-process crash or ambiguous CLI response.
"""
import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
RUN=Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0')
FROZEN=RUN/'frozen_v1'
def now():return datetime.now(timezone.utc).isoformat()
def save(p,r):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--renderer-job',required=True);p.add_argument('--timeout-hours',type=float,default=24)
    p.add_argument('--package-name',default='package_v4');p.add_argument('--control-name',default='control')
    p.add_argument('--job-name',default='R1-clean-Projective-v0-250');a=p.parse_args()
    package=RUN/a.package_name
    out=RUN/a.control_name;out.mkdir(exist_ok=True)
    with (out/'submit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'training_submission.json').exists():print('already submitted; no duplicate');return
        assert not (out/'submission_intent.json').exists(),'prior intent without receipt: inspect SCO before retry'
        started=time.monotonic()
        while time.monotonic()-started<a.timeout_hours*3600:
            result=subprocess.run([SCO,'acp','jobs','describe','--workspace-name=aigc','-o','json',a.renderer_job],capture_output=True,text=True)
            if result.returncode:
                save(out/'status.json',{'status':'CONTROL_QUERY_ERROR','utc':now(),'renderer_job':a.renderer_job});time.sleep(30);continue
            job=json.loads(result.stdout);assert job['resource_pool']['name']=='h800'
            save(out/'renderer_job.json',job)
            if job['state'] in ('FAILED','SUCCEEDED','STOPPED','DELETED','CANCELED','CANCELLED'):
                raise RuntimeError('renderer job ended: '+job['state'])
            gate=RUN/'runtime_preflight.json';endpoint=RUN/'renderer_endpoint.txt'
            if gate.exists() and endpoint.exists():
                report=json.loads(gate.read_text())
                if report['status']=='PASS' and report['completed']==210:break
            save(out/'status.json',{'status':'WAITING_FOR_RUNTIME_PREFLIGHT','utc':now(),'renderer_job':a.renderer_job,'job_state':job['state']})
            time.sleep(30)
        else:raise TimeoutError('bounded control wait expired; no training submitted')
        assert report['manifest_sha256']==hashlib.sha256((FROZEN/'train.jsonl').read_bytes()).hexdigest()
        subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=FROZEN,check=True)
        subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=package,check=True)
        command=[SCO,'acp','jobs','create','--workspace-name=aigc','--aec2-name=h800',
          '--job-name='+a.job_name,'--priority=HIGHEST','--quota-type=reserved',
          '--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
          '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm',
          '--training-framework=pytorch','--worker-nodes=1','--worker-spec=N4lS.Iq.I80.8',
          '--command=bash '+str(package/'launch_training.sh')]
        intent={'utc':now(),'command':command,'renderer_job':a.renderer_job,
                'runtime_preflight_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),
                'formal_requires':'PPO smoke actual update + checkpoint load PASS; fresh pretrained process',
                'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        save(out/'submission_intent.json',intent)
        result=subprocess.run(command,capture_output=True,text=True)
        (out/'training_submit.log').write_text(result.stdout+'\n'+result.stderr)
        ids=set(re.findall(r'pt-[a-z0-9]+',result.stdout))
        assert result.returncode==0 and len(ids)==1,'submission response uncertain: do not automatically resubmit'
        jobid=ids.pop();save(out/'training_submission.json',{**intent,'job_id':jobid,'state':'SUBMITTED_NOT_YET_TRAINED'})
        save(out/'status.json',{'status':'TRAINING_JOB_SUBMITTED','job_id':jobid,'utc':now()})
        print(jobid,flush=True)

if __name__=='__main__':main()
