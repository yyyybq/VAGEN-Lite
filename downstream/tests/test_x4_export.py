"""downstream/x/run_x4_export.sh with a stand-in exporter: each E1 model exported once, a half-written folder redone."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB = """import sys
from pathlib import Path
a = sys.argv[1:]
out, ckpt = Path(a[a.index("--out") + 1]), a[a.index("--ckpt") + 1]
if out.exists():
    raise SystemExit("exists")
out.mkdir(parents=True)
(out / "model.safetensors").write_text("w")
(out / "starvla_export.json").write_text(ckpt)
"""


def test_each_model_is_exported_once_and_a_half_export_is_redone(tmp_path):
    main = tmp_path / "main"
    (main / "scripts").mkdir(parents=True)
    (main / "scripts/export_starvla_backbone.py").write_text(STUB)
    hf = tmp_path / "hf"
    (hf / "E1-B-200k").mkdir(parents=True)
    (hf / "E1-B-200k/starvla_export.json").write_text("before")  # done
    (hf / "E1-S-200k").mkdir()  # half-written: no starvla_export.json
    env = {"X4_MAIN": str(main), "X4_RUNS": "/runs", "X4_PY": sys.executable, "PATH": "/usr/bin:/bin"}
    res = subprocess.run(["bash", str(ROOT / "x/run_x4_export.sh"), str(hf)], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (hf / "E1-B-200k/starvla_export.json").read_text() == "before"
    for s in ("S", "RS", "RD"):
        assert (hf / f"E1-{s}-200k/starvla_export.json").read_text() == f"/runs/DS-{s}-100-s1/run/final_model/pytorch_model.pt"
    log = (hf / "export.log").read_text()
    assert "B: done before" in log and "S: removed a half-written export" in log
