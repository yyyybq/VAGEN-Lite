#!/usr/bin/env python3
"""Publish the validated R1 SFT visualization to this repo's gh-pages branch."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "exps/vagen_active_spatial/R1-gate-aligned-sft-v2-20261008"
OWNER = "yyyybq"
REPOSITORY = "VAGEN-Lite"
REMOTE = f"https://github.com/{OWNER}/{REPOSITORY}.git"
SITE_URL = f"https://{OWNER}.github.io/{REPOSITORY}/"
USER_SITE_REPOSITORY = f"{OWNER}.github.io"
ALIAS_PATH = "close_loop_spatial/index.html"
ALIAS_URL = f"https://{OWNER}.github.io/close_loop_spatial/"


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout.strip()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    output = run_dir / "output"
    site = output / "visualization"
    gate_path = output / "release_gate.json"
    complete = run_dir / "COMPLETE"
    if not complete.is_file() or not gate_path.is_file():
        raise FileNotFoundError("run is not complete or has no release gate")
    gate = json.loads(gate_path.read_text())
    expected = {
        "status": "PASS",
        "records": 210,
        "successful_trajectories": 210,
        "dashboards": 210,
        "certified_shortest_matches": 147,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(f"release gate mismatch: {key}={gate.get(key)!r}, expected {value!r}")
    if digest(site / "index.html") != gate["site_index_sha256"]:
        raise ValueError("site index hash does not match release gate")
    dashboards = sorted((site / "dashboards").glob("*.png"))
    if len(dashboards) != 210 or any(path.stat().st_size == 0 for path in dashboards):
        raise ValueError("site does not contain 210 non-empty dashboard images")
    return site, gate


def github_token() -> str:
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError("GitHub credential helper failed")
    values = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    token = values.get("password", "")
    if not token:
        raise RuntimeError("GitHub credential helper returned no password/token")
    return token


def api(token: str, method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "VAGEN-Lite-R1-SFT-publisher",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else None
    except HTTPError as error:
        payload = error.read()
        parsed = json.loads(payload) if payload else None
        return error.code, parsed


def enable_pages(token: str) -> dict[str, Any]:
    path = f"/repos/{OWNER}/{REPOSITORY}/pages"
    status, page = api(token, "GET", path)
    if status == 404:
        status, page = api(
            token,
            "POST",
            path,
            {"source": {"branch": "gh-pages", "path": "/"}},
        )
    elif status == 200:
        source = (page or {}).get("source") or {}
        if source.get("branch") != "gh-pages" or source.get("path") != "/":
            status, page = api(
                token,
                "PUT",
                path,
                {"source": {"branch": "gh-pages", "path": "/"}},
            )
    if status not in (200, 201, 204):
        raise RuntimeError(f"GitHub Pages API failed with HTTP {status}: {page}")
    status, current = api(token, "GET", path)
    if status != 200:
        raise RuntimeError(f"GitHub Pages readback failed with HTTP {status}: {current}")
    return current


def site_reachable() -> bool:
    try:
        request = Request(SITE_URL, headers={"User-Agent": "VAGEN-Lite-R1-SFT-publisher"})
        with urlopen(request, timeout=20) as response:
            return response.status == 200 and b"Active Spatial" in response.read(262144)
    except (HTTPError, URLError, TimeoutError):
        return False


def publish_alias_route(run_dir: Path, wait_seconds: int) -> dict[str, Any]:
    site, gate = validate(run_dir)
    alias_html = (site / "index.html").read_text()
    alias_html = alias_html.replace(
        "<head>",
        f"<head><base href='/{REPOSITORY}/'><link rel='canonical' href='{SITE_URL}'>",
        1,
    )
    encoded = base64.b64encode(alias_html.encode()).decode()
    token = github_token()
    try:
        path = f"/repos/{OWNER}/{USER_SITE_REPOSITORY}/contents/{ALIAS_PATH}"
        status, existing = api(token, "GET", path)
        if status == 200:
            remote_content = base64.b64decode((existing or {}).get("content", "")).decode()
            if remote_content != alias_html:
                raise FileExistsError(
                    f"refusing to overwrite an existing, different user-site route: {ALIAS_PATH}"
                )
            commit_sha = ((existing or {}).get("sha") or "already-present")
        elif status == 404:
            status, created = api(
                token,
                "PUT",
                path,
                {
                    "message": "Publish close_loop_spatial route",
                    "content": encoded,
                    "branch": "master",
                },
            )
            if status not in (200, 201):
                raise RuntimeError(f"user-site route creation failed with HTTP {status}: {created}")
            commit_sha = (((created or {}).get("commit") or {}).get("sha") or "unknown")
        else:
            raise RuntimeError(f"user-site route lookup failed with HTTP {status}: {existing}")
    finally:
        token = ""

    deadline = time.monotonic() + wait_seconds
    reachable = False
    while not reachable and time.monotonic() <= deadline:
        try:
            request = Request(ALIAS_URL, headers={"User-Agent": "VAGEN-Lite-R1-SFT-publisher"})
            with urlopen(request, timeout=20) as response:
                content = response.read(262144)
                reachable = (
                    response.status == 200
                    and b"Active Spatial" in content
                    and f"<base href='/{REPOSITORY}/'>".encode() in content
                )
        except (HTTPError, URLError, TimeoutError):
            reachable = False
        if not reachable:
            time.sleep(10)

    receipt_path = run_dir / "release/github_pages_release.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    receipt["alias"] = {
        "status": "PUBLISHED" if reachable else "DEPLOYING",
        "url": ALIAS_URL,
        "source_url": SITE_URL,
        "user_site_repository": f"{OWNER}/{USER_SITE_REPOSITORY}",
        "path": ALIAS_PATH,
        "commit": commit_sha,
        "site_index_sha256": gate["site_index_sha256"],
    }
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    print(json.dumps(receipt["alias"], indent=2, sort_keys=True))
    return receipt["alias"]


def publish(run_dir: Path, wait_seconds: int) -> dict[str, Any]:
    site, gate = validate(run_dir)
    with tempfile.TemporaryDirectory(prefix="r1-sft-pages-") as temporary:
        checkout = Path(temporary) / "repo"
        checkout.mkdir()
        run(["git", "init", "-q"], checkout)
        run(["git", "symbolic-ref", "HEAD", "refs/heads/gh-pages"], checkout)
        run(["git", "remote", "add", "origin", REMOTE], checkout)
        shutil.copytree(site, checkout, dirs_exist_ok=True)
        shutil.copy2(run_dir / "output/release_gate.json", checkout / "release_gate.json")
        (checkout / ".nojekyll").write_text("\n")
        run(["git", "config", "user.name", "R1 SFT Publisher"], checkout)
        run(["git", "config", "user.email", "r1-sft-publisher@users.noreply.github.com"], checkout)
        run(["git", "add", "--all"], checkout)
        run(
            [
                "git",
                "commit",
                "-q",
                "-m",
                f"Publish R1 score-guided trajectories ({gate['records']} records)",
            ],
            checkout,
        )
        commit = run(["git", "rev-parse", "HEAD"], checkout)
        run(["git", "push", "origin", "gh-pages:gh-pages"], checkout)

    token = github_token()
    try:
        pages = enable_pages(token)
    finally:
        token = ""
    deadline = time.monotonic() + wait_seconds
    reachable = site_reachable()
    while not reachable and time.monotonic() < deadline:
        time.sleep(10)
        reachable = site_reachable()
    result = {
        "status": "PUBLISHED" if reachable else "DEPLOYING",
        "site_url": SITE_URL,
        "branch": "gh-pages",
        "commit": commit,
        "records": gate["records"],
        "site_index_sha256": gate["site_index_sha256"],
        "pages": {
            "html_url": pages.get("html_url"),
            "status": pages.get("status"),
            "source": pages.get("source"),
        },
    }
    release_dir = run_dir / "release"
    release_dir.mkdir(exist_ok=True)
    receipt = release_dir / "github_pages_release.json"
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--alias-route-only", action="store_true")
    args = parser.parse_args()
    if args.alias_route_only:
        publish_alias_route(args.run_dir.resolve(), max(0, args.wait_seconds))
    else:
        publish(args.run_dir.resolve(), max(0, args.wait_seconds))


if __name__ == "__main__":
    main()
