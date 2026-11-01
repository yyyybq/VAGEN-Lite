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


def _runs(root: Path, succ: dict, points=V.POINTS):
    """succ: start -> task -> successes at each point, of 20 episodes (seeds 1000..1019, the first k succeed)."""
    for start, tasks in succ.items():
        exp = root / f"experiments/DS-{start}-100-s1"
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")
        for i, step in enumerate(points):
            d = exp / f"evals/steps_{step}/out/episodes"
            d.mkdir(parents=True)
            rows = [{"task": t, "seed": 1000 + j, "success": j < ks[i], "steps": 10, "max_steps": 100,
                     "status": "complete"} for t, ks in tasks.items() for j in range(20)]
            (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (d / "summary.json").write_text(json.dumps({"per_task": {}}))


def test_report_pairs_the_recheck_with_the_model_lines_controls_by_hand(tmp_path, monkeypatch):
    """Two runs roots read as one collection; every start against Q3B and against Q3RD on the three-point mean."""
    import pytest
    from downstream.report import collect

    monkeypatch.setattr(collect, "excluded", lambda: set())
    models, ours = tmp_path / "models", tmp_path / "ours"
    _runs(models, {"Q3B": {"OpenDrawer": (10, 10, 10), "CloseFridge": (5, 5, 5)},
                   "Q3RD": {"OpenDrawer": (8, 8, 8), "CloseFridge": (5, 5, 5)},
                   "B": {"OpenDrawer": (20, 20, 20)}})  # another start in a root: left out
    _runs(ours, {"Q3V4S3": {"OpenDrawer": (6, 8, 10), "CloseFridge": (5, 5, 5)},
                 "Q3DCKS2": {"OpenDrawer": (4, 4, 4), "CloseFridge": (7, 7, 7)}})
    res = V.report(tmp_path / "out", roots=(models, ours))
    assert res["starts"] == ["Q3B", "Q3RD", "Q3V4S3", "Q3DCKS2"] and "B" not in res["groups"]
    assert res["counts"]["Q3V4S3"] == [[11, 40], [13, 40], [15, 40]]
    assert res["groups"]["Q3V4S3"]["all"]["mean"] == pytest.approx(13 / 40)
    vb, vr = res["vs"]["Q3B"], res["vs"]["Q3RD"]
    assert set(vb) == {"Q3RD", "Q3V4S3", "Q3DCKS2"} and set(vr) == {"Q3B", "Q3V4S3", "Q3DCKS2"}
    # per task, the mean over the points: V4S3 OpenDrawer (-0.2 - 0.1 + 0) / 3 = -0.1, CloseFridge 0
    assert vb["Q3V4S3"]["all"]["task_mean_diff"] == pytest.approx(-0.05)
    assert vb["Q3DCKS2"]["all"]["task_mean_diff"] == pytest.approx((-0.3 + 0.1) / 2)
    assert vr["Q3V4S3"]["all"]["task_mean_diff"] == pytest.approx(0.0)
    assert vr["Q3DCKS2"]["all"]["task_mean_diff"] == pytest.approx((-0.2 + 0.1) / 2)
    assert vb["Q3V4S3"]["all"]["pooled_mean_diff"] == pytest.approx(-2 / 40)
    assert [x["step"] for x in vb["Q3V4S3"]["all"]["pooled_by_step"]] == list(V.POINTS)
    md = (tmp_path / "out/report.md").read_text()
    assert "| Q3V4S3 | V4 换种子 20261022，第 250 次 | 11/40（27.5%） | 13/40（32.5%） | 15/40（37.5%） | 32.5% |" in md
    assert "## 和 Q3B 配对" in md and "## 和 Q3RD 配对" in md and "| OpenDrawer | 50.0% | 40.0% | 40.0% | 20.0% |" in md
    assert json.loads((tmp_path / "out/report.json").read_text())["counts"]["Q3B"][0] == [15, 40]


def test_report_stops_without_every_point_and_on_a_name_in_two_roots(tmp_path, monkeypatch):
    import pytest
    from downstream.report import collect

    monkeypatch.setattr(collect, "excluded", lambda: set())
    models, ours = tmp_path / "models", tmp_path / "ours"
    _runs(models, {"Q3B": {"OpenDrawer": (1, 1, 1)}, "Q3RD": {"OpenDrawer": (1, 1, 1)}})
    _runs(ours, {"Q3V4S3": {"OpenDrawer": (1, 1, 1)}, "Q3DCKS2": {"OpenDrawer": (1, 1)}}, points=V.POINTS[:2])
    with pytest.raises(SystemExit, match="Q3DCKS2"):
        V.report(tmp_path / "out", roots=(models, ours))
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit, match="more than one runs root"):
        V.merged((models, models), ("Q3B",))
