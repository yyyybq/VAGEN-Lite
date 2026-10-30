"""downstream/x/v4_recheck.py: the model line's builder is called with this recheck's own start registry and DS-Q3RD's
protocol, each job is prepared with the model line's run_full.sh, queue entries come in order (zoetrope first), and
a second call builds nothing again; all under a temporary runs folder, with stand-ins for the builder and sco_job."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "x"))
import v4_recheck as V  # noqa: E402


def test_make_builds_prepares_and_queues(tmp_path):
    (tmp_path / "queue").mkdir()
    built, prepared = [], []

    def build(start, data, exp, steps, save_interval, log_every, seed, full_state_interval=0, eval_points=()):
        registry = json.loads(Path(fake_mc.STARTS).read_text())
        built.append((start, registry[start]["path"], steps, save_interval, log_every, seed, full_state_interval,
                      list(eval_points)))
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")

    def prepare(run, gpus, pool, name, payload):
        prepared.append((run.name, gpus, pool, name, payload))
        (run / "job").mkdir()
        (run / "job/prepared.json").write_text("{}")

    fake_mc = SimpleNamespace(STARTS=None, build=build)
    fake_sco = SimpleNamespace(prepare=prepare)
    queued = V.make(["Q3V4S3", "Q3DCKS2"], runs=tmp_path, make_config=fake_mc, sco_job=fake_sco)
    assert [b[0] for b in built] == ["Q3V4S3", "Q3DCKS2"]
    assert built[0][1].endswith("rl_run_qw3v4s3/checkpoints/global_step_250/actor/huggingface")
    assert built[1][1].endswith("rl_run_qw3dcks2/checkpoints/global_step_300/actor/huggingface")
    assert all(b[2:] == (200000, 25000, 100, 1, 25000, [200000]) for b in built)
    assert prepared[0][:4] == ("DS-Q3V4S3-100-s1", 8, "zoetrope", "v4-q3v4s3")
    assert prepared[0][4][1].endswith("model_matrix/downstream/run_full.sh")
    assert [Path(q).name for q in queued] == ["q11_v4_1_Q3V4S3.json", "q11_v4_2_Q3DCKS2.json"]
    entry = json.loads(Path(queued[1]).read_text())
    assert entry["cards"] == 8 and entry["pools"] == ["zoetrope", "h800"] and entry["run"].endswith("DS-Q3DCKS2-100-s1")
    V.make(["Q3V4S3", "Q3DCKS2"], runs=tmp_path, make_config=fake_mc, sco_job=fake_sco)
    assert len(built) == 2 and len(prepared) == 2  # nothing built or prepared twice
