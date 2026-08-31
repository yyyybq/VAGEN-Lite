# Active Spatial RemoteEnv Service

This project now supports serving the full `ActiveSpatialGymEnv` through VAGEN's upstream `RemoteEnv` HTTP protocol. This is different from the legacy
WebSocket-only GS render server:

- Legacy `render_backend=client` sends only render requests to a GS render server.
- `RemoteEnv` sends normal environment calls (`connect`, `reset`, `step`, `close`)
  to a remote `GymService`; the service creates `ActiveSpatialGymEnv` locally.

Use `RemoteEnv` when the render machine has local access to InteriorGS / gsplat
and the training machine should avoid running environment dependencies locally.
The transport uses the upstream VAGEN improvements: persistent HTTP clients,
multipart binary image transfer, retries, URL pool/failover, and service-side
inflight admission control.

## Start the service

On the environment/render machine:

```bash
bash examples/train/active_spatial/start_active_spatial_env_server.sh \
  --port 8000 \
  --gpus 0 \
  --max-inflight 16
```

The service does not take a dataset path or GS root at startup. Those remain in
the experiment env config and are sent by each `RemoteEnv` client session.

## Use from training

On the training machine, set `REMOTE_ENV_URLS` before running the usual
experiment script:

```bash
export REMOTE_ENV_URLS=http://<env-service-host>:8000
export REMOTE_ENV_TIMEOUT=600
export REMOTE_ENV_RETRIES=3

bash examples/train/active_spatial/experiments/<experiment>.sh
```

When `REMOTE_ENV_URLS` is set, `run_experiment.sh` writes train/val YAML entries
with `name: RemoteEnv` and forwards the original Active Spatial config inside
the remote env config. If `REMOTE_ENV_URLS` is unset, the launcher preserves the
existing local / legacy render-server behavior.

## Relation to ViewAgent

ViewAgent's ScanNet render service uses a stronger renderer-server design:
a fixed process pool, sticky scene-to-worker routing, GPU assignment,
crash recovery, multipart image responses, and optional routed clients. This
commit only wires Active Spatial into VAGEN's generic `RemoteEnv` transport.
Porting ViewAgent's worker-pool renderer to InteriorGS is a separate step
because ViewAgent's implementation is tied to its ScanNet mesh/gsplat data
layout.
