import json, os, shutil, sys
from pathlib import Path
W = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix"
sys.path.insert(0, W)
from data_gen.interiorgs_v10 import job_sco
run = Path(W) / "exps/matrix_v3/g_prep_20261106/data_run"
(run / "job").mkdir(parents=True, exist_ok=True)
if (run / "job" / "submission.json").exists():
    raise SystemExit("submitted before")
shutil.copyfile("/tmp/yb_msgs/g/g_data_launch.sh", run / "job" / "launch.sh")
os.chmod(run / "job" / "launch.sh", 0o755)
pool = job_sco.pick_pool()
(run / "job" / "prepared.json").write_text(json.dumps({"utc": job_sco.now(), "gpus": 1, "spec": job_sco.SPECS[1],
                                                       "pool": pool, "mode_args": ["g-data"]}, indent=1))
print(pool, job_sco.submit(run))
