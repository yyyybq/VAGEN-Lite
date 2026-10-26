"""Unified configuration (cleanup phase 2, 2026-10): configs/paths.yaml, configs/cluster.yaml and vagen.config.

The cluster values are pinned here: they are the literals the code used before the configuration existed, so a
change to configs/ is a deliberate change of the jobs the repository submits.  Places that keep their own copy of a
value (shell job scripts that run before any Python, command-line defaults of modules that run on the workers from a
packaged source tree, the EASI registry) are checked against the configuration instead.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while not (p / "setup.py").is_file():
        p = p.parent
    return p


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from vagen import config  # noqa: E402

PINNED_PATHS = {
    "user_root": "/mnt/umm/users/yinbaiqiao",
    "repo": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite",
    "conda_env": "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite",
    "scenes_meta": "/mnt/umm/users/yinbaiqiao/InteriorGS_meta",
    "scenes_ply": "/mnt/umm/users/yinbaiqiao/InteriorGS_ply",
    "aoss_conf": "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf",
    "gsplat_cache": "/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007",
    "torch_extensions_renderer": "/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer",
    "hf_home": "/mnt/umm/users/yinbaiqiao/.cache/huggingface",
}
PINNED_SCO = {
    "sco.bin": "/mnt/umm/users/yinbaiqiao/.sco/bin/sco",
    "sco.workspace": "aigc",
    "sco.image": "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
    "sco.storage_mount": "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
    "sco.default_pool": "zoetrope",
    "sco.pools": ["zoetrope", "h800"],
    "sco.render_pools": ["zoe-cf-render"],
    "sco.worker_specs": {1: "N4lS.Iq.I80.1", 8: "N4lS.Iq.I80.8"},
}


@pytest.fixture(autouse=True)
def _default_profile(monkeypatch):
    monkeypatch.delenv("VAGEN_PROFILE", raising=False)
    monkeypatch.delenv("VAGEN_CONFIG_DIR", raising=False)


def test_cluster_values_are_the_ones_the_code_used_before():
    assert config.profile() == "cluster"
    for key, value in PINNED_PATHS.items():
        assert config.path(key) == value, key
    for key, value in PINNED_SCO.items():
        assert config.cluster(key) == value, key


def test_both_profiles_name_the_same_keys():
    profiles = yaml.safe_load((ROOT / "configs/paths.yaml").read_text())["profiles"]
    assert set(profiles) == set(config.PROFILES)
    assert set(profiles["cluster"]) == set(profiles["local"]) == set(PINNED_PATHS)


def test_local_profile_and_bad_names(monkeypatch):
    monkeypatch.setenv("VAGEN_PROFILE", "local")
    with pytest.raises(KeyError, match="scenes_meta"):
        config.path("scenes_meta")
    monkeypatch.setenv("VAGEN_PROFILE", "elsewhere")
    with pytest.raises(ValueError, match="VAGEN_PROFILE"):
        config.profile()
    monkeypatch.setenv("VAGEN_PROFILE", "cluster")
    with pytest.raises(KeyError, match="no key"):
        config.cluster("sco.nothing")


def test_config_dir_override(tmp_path, monkeypatch):
    (tmp_path / "cluster.yaml").write_text("sco:\n  workspace: other\n")
    monkeypatch.setenv("VAGEN_CONFIG_DIR", str(tmp_path))
    assert config.cluster("sco.workspace") == "other"
    with pytest.raises(FileNotFoundError, match="paths.yaml"):
        config.path("repo")


def test_command_line():
    env = {k: v for k, v in os.environ.items() if k not in ("VAGEN_PROFILE", "VAGEN_CONFIG_DIR")}
    env["PYTHONPATH"] = str(ROOT)

    def run(key):
        return subprocess.run([sys.executable, "-m", "vagen.config", "get", key], cwd=ROOT, env=env,
                              capture_output=True, text=True, check=True).stdout.strip()

    assert run("paths.scenes_meta") == PINNED_PATHS["scenes_meta"]
    assert run("cluster.sco.workspace") == "aigc"
    assert run("cluster.sco.pools") == "zoetrope h800"


def test_sco_modules_take_their_settings_from_the_configuration():
    from data_gen.interiorgs_v10 import job_sco, pilot_sco, sco_interiorgs_v10, shard_sco

    for mod in (job_sco, pilot_sco, shard_sco, sco_interiorgs_v10):
        assert mod.SCO == config.cluster("sco.bin")
        assert mod.IMAGE == config.cluster("sco.image")
        assert mod.MOUNT == config.cluster("sco.storage_mount")
        assert mod.WORKSPACE == config.cluster("sco.workspace")
        text = Path(mod.__file__).read_text()
        for literal in ('"--workspace-name=aigc"', "registry.cn-fz-01", "019ec9f9-6d12", "/.sco/bin/sco"):
            assert literal not in text, (mod.__name__, literal)
    assert job_sco.DEFAULT_WORKTREE == config.path("repo")
    assert job_sco.POOL == "zoetrope" and job_sco.POOLS == ("zoetrope", "h800")
    assert job_sco.RENDER_POOLS == ("zoe-cf-render",)
    assert job_sco.SPECS == {1: "N4lS.Iq.I80.1", 8: "N4lS.Iq.I80.8"}
    assert pilot_sco.RUNTIME_ENV == shard_sco.RUNTIME_ENV == config.path("conda_env")
    assert pilot_sco.WORKER_SPEC == shard_sco.WORKER_SPEC == sco_interiorgs_v10.WORKER_SPEC == "N4lS.Iq.I80.1"
    assert pilot_sco.TORCH_EXTENSIONS_DIR == config.path("torch_extensions_renderer")
    assert shard_sco.SCENES_ROOT == str(sco_interiorgs_v10.SCENES_ROOT) == config.path("scenes_meta")
    assert str(sco_interiorgs_v10.ENV_PY) == config.path("conda_env") + "/bin/python"


def _line_value(rel: str, pattern: str) -> str:
    found = [m.group(1) for m in (re.match(pattern, line) for line in (ROOT / rel).read_text().splitlines()) if m]
    assert len(found) == 1, (rel, pattern, found)
    return found[0]


@pytest.mark.parametrize("rel,pattern,key", [
    ("examples/train/active_spatial/sco_interiorgs_v10_train.sh", r"^WORKTREE=\$\{VAGEN_LITE_WORKTREE:-(.+)\}$", "repo"),
    ("examples/train/active_spatial/sco_interiorgs_v10_train.sh", r"^ENV_ROOT=(\S+)$", "conda_env"),
    ("examples/train/active_spatial/sco_interiorgs_v10_train.sh", r"^GS_ROOT=(\S+)$", "scenes_ply"),
    ("examples/train/active_spatial/sco_interiorgs_v10_train.sh", r"^VALIDATED_GSPLAT_CACHE=(\S+)$", "gsplat_cache"),
    ("examples/train/active_spatial/sco_run_as_artifact_owner.sh", r"^TARGET_HOME=\$\{R1_ARTIFACT_HOME:-(.+)\}$", "user_root"),
    ("data_gen/interiorgs_v10/evaluation.py", r'^\s*r\.add_argument\("--gs-root", default="([^"]+)"\)', "scenes_ply"),
    ("data_gen/interiorgs_v10/depth_check.py", r'^\s*r\.add_argument\("--meta-root", default="([^"]+)",', "scenes_meta"),
    ("data_gen/interiorgs_v10/fetch_interiorgs_metadata.py", r'^\s*ap\.add_argument\("--aoss-conf", default="([^"]+)"\)', "aoss_conf"),
    ("vagen/envs/active_spatial/gs_io.py", r'^\s*"(/mnt/umm/[^"]+petreloss\.conf)",$', "aoss_conf"),
])
def test_copies_outside_the_configuration_match_it(rel, pattern, key):
    assert _line_value(rel, pattern) == config.path(key)


def test_easi_registry_roots_match_the_configuration():
    reg = yaml.safe_load((ROOT / "scripts/easi_registry.yaml").read_text())
    repo = config.path("repo")
    assert reg["easi_dir"] == repo + "/third_party/EASI"
    assert reg["lmms_eval_dir"] == repo + "/third_party/lmms-eval"
    assert reg["base_ckpt_dir"] == repo
