"""R1 runs keep R1's success rule: every R1 training / evaluation config read the way its runner reads it
has r1_legacy_success on (decision of 2026-10-14)."""

import importlib.util
import re
from argparse import Namespace
from pathlib import Path

import yaml
from omegaconf import OmegaConf, open_dict

from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.envs.active_spatial.r1_legacy import env_entries, force_r1_legacy, legacy_env_yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/train/active_spatial"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _typed(raw):
    keys = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    return ActiveSpatialEnvConfig(**{k: v for k, v in raw.items() if k in keys})


def test_default_is_off():
    assert ActiveSpatialEnvConfig().r1_legacy_success is False


def test_dense_score_matrices_materialize_with_the_switch_on():
    preflight = _script("active_spatial_dense_score_preflight")
    for name in ("dense_score_ablation_r1_20261006.yaml", "dense_score_ablation_candidates.yaml"):
        matrix = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
        env = preflight.apply_environment_overrides(matrix, {"success_reward": 5.0})
        assert _typed(env).r1_legacy_success is True, name
        for spec in matrix["variants"].values():  # no variant turns it back off
            assert "r1_legacy_success" not in spec["reward_overrides"]


def test_r1_env_yaml_in_examples():
    doc = yaml.safe_load((EXAMPLES / "env_config_diagnostic_r1_real_batch.yaml").read_text(encoding="utf-8"))
    assert all(_typed(e).r1_legacy_success is True for e in env_entries(doc))


def test_runners_launch_with_legacy_env_copies(tmp_path):
    env_yaml = tmp_path / "train.yaml"
    env_yaml.write_text(yaml.safe_dump({"envs": [{"name": "ActiveSpatial", "config": {
        "client_url": "${oc.env:R1_RENDER_URL}", "success_reward": 5.0}}]}))
    val_yaml = tmp_path / "val.yaml"
    val_yaml.write_text(env_yaml.read_text())
    cfg = OmegaConf.create({"data": {"train_files": str(env_yaml), "val_files": [str(val_yaml)]}})
    with open_dict(cfg):
        written = force_r1_legacy(cfg, tmp_path / "out")
    assert len(written) == 2 and cfg.data.train_files == written[0] and list(cfg.data.val_files) == [written[1]]
    for path in written:
        doc = yaml.safe_load(Path(path).read_text())
        entry = doc["envs"][0]["config"]
        assert entry["r1_legacy_success"] is True
        assert entry["client_url"] == "${oc.env:R1_RENDER_URL}"  # interpolation kept for OmegaConf
    assert "r1_legacy_success" not in env_yaml.read_text()  # the frozen original is untouched
    # the legacy-dict layout ({env1: {env_config: ...}}) too
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump({"env1": {"env_name": "active_spatial", "env_config": {"image_width": 256}}}))
    assert yaml.safe_load(legacy_env_yaml(old, tmp_path / "o2").read_text())["env1"]["env_config"]["r1_legacy_success"]


def test_runner_sources_call_the_helper():
    for module in ("vagen/r1_clean_projective_ppo.py", "vagen/dense_score_reward_only_pilot.py"):
        text = (ROOT / module).read_text(encoding="utf-8")
        assert "force_r1_legacy(" in text, module


def test_r1_eval_and_preflight_scripts_build_legacy_configs(tmp_path):
    eval32 = _script("r1_run_canonical_dev_eval32")
    cfg = eval32.env_config(Namespace(renderer_url="http://x/render", gs_root=tmp_path), tmp_path / "rows.jsonl")
    assert cfg.r1_legacy_success is True
    consistency = _script("r1_runtime_consistency")
    args = Namespace(repaired=tmp_path / "rows.jsonl", renderer_url="http://x/render", gs_root=tmp_path, max_steps=12)
    assert consistency.config(args).r1_legacy_success is True
    text = (ROOT / "scripts/r1_clean_projective_runtime_preflight.py").read_text(encoding="utf-8")
    assert "raw.update(R1_ENV_OVERRIDES)" in text


def test_every_r1_entry_script_goes_through_a_legacy_entry_point():
    """sco_r1_* / sco_dense_score_* launchers start training only through the two runners above and evaluate
    only through the scripts above."""
    runners = {"r1_clean_projective_ppo", "dense_score_reward_only_pilot"}
    for sh in sorted(EXAMPLES.glob("sco_*.sh")):
        if not (sh.name.startswith("sco_r1_") or sh.name.startswith("sco_dense_score_")):
            continue
        text = sh.read_text(encoding="utf-8")
        for module in re.findall(r"-m\s+vagen\.([A-Za-z0-9_]+)", text):
            assert module in runners, f"{sh.name} starts vagen.{module}"
        assert "main_ppo" not in text, sh.name
