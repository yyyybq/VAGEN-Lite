"""Prepare immutable A/B requests for the SCO diagnostic worker."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def request_hash(req):
    payload = dict(req); payload.pop('request_sha256', None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def main():
    """Use the immutable v6 builder; never silently rewrite a prior v3 bank."""
    from .position_render_run import build_requests
    build_requests()

if __name__ == '__main__': main()
