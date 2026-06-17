# argus

On-prem **surveillance-footage analysis & understanding**: read recorded video and make
sense of it — detect objects, track them across frames, and triage which clips are worth a
closer look. Built to extend toward face-ID/re-ID, embeddings, and search (full design in
[context/face-id-design.md](context/face-id-design.md)).

`argus` is a small, model-agnostic, pluggable SDK; the marimo notebooks in `examples/`
exercise it on real surveillance footage.

## Layout

```
argus/
  core/        shared types + extension Protocols (Detector, Tracker, FaceDetector,
               Embedder, AudioClassifier, Store) + COCO taxonomy — dependency-free
  detect/      object detection; YOLO11 + open-vocabulary YOLO-World backends
  track/       tracking-by-detection; ByteTrack backend
  face/ embed/ face detection + alignment + ArcFace embedding
  store/       SQLite + sqlite-vec sighting store
  audio/       audio classification (AST / zero-shot CLAP) backends
  pipeline/    orchestration: track_video, peek_videos, ingest_video, analyze_audio
  identity/    face search / re-ID clustering / compliance over the store
  mcp/         MCP server exposing the facade as HTTP tools
examples/      marimo notebooks
tests/         fast unit suite (no GPU/weights/data needed)
context/       architecture.md + face-id-design.md + mcp-server.md
```

The extension contracts all live in `argus/core` — implement a Protocol and drop the file in
the matching `*/backends/` folder. See [context/architecture.md](context/architecture.md).

## Setup

```bash
uv sync
```

Installs torch/ultralytics/opencv directly (no SSH/private repo needed). torch defaults to
the **CPU** wheel — reliable everywhere and what the test suite wants; opt into GPU per the
Docker section below. `yolo11s.pt` auto-downloads on first detector use. Optional extras:
`face`/`face-gpu` (face-ID), `store` (sighting DB), `cluster` (re-ID), `audio` (sound
classification), `open-vocab` (YOLO-World; needs `git`), `mcp` (server), `datasets` (sample
data, needs SSH), `notebooks` (marimo).

## Use the SDK

```python
from argus import track_video, peek_videos

result = track_video("clip.mp4", targets=("person", "vehicle"), device="cuda")
result.metrics()                       # per-track polars DataFrame
result.to_parquet("tracks.parquet", what="tracks")
result.render("annotated.mp4")         # annotated H.264 video

peek_videos(clips, targets=("vehicle",), device="cuda")   # fast folder triage
```

Both axes are pluggable: any object satisfying the `Detector` (`.detect(frame)`) or
`Tracker` (`.update(dets, frame)` / `.reset()`) protocol drops into `VideoTracker`.

## Sample scripts

Runnable CLI examples in `examples/` (each takes `--help`):

```bash
uv run python examples/track_clip.py    clip.mp4 --targets person vehicle --render
uv run python examples/peek_folder.py   /footage --targets vehicle --workers 8
uv run python examples/custom_backend.py clip.mp4   # a custom Detector via the protocol
```

## CLI

Installing the package exposes an `argus` command (a thin wrapper over the facade):

```bash
uv run argus peek  /footage --targets vehicle --workers 8 --json
uv run argus track clip.mp4 --targets person vehicle --render out.mp4 --json
```

## Run a notebook

```bash
uv run marimo edit examples/01_dvr_person_tracking.py
```

Run from the **repo root** so `import argus` resolves.

## Docker

A single multi-stage `Dockerfile` builds both a **CPU** image (multi-arch amd64+arm64) and a
**CUDA** image (amd64); the `Makefile` wraps the buildx/QEMU commands and the CPU-only smoke
test. The image entrypoint is the `argus` CLI; mount your footage and run:

```bash
make build-amd64                                  # or: make build-cpu (multi-arch)
docker run --rm -v /footage:/data argus:amd64 peek /data --json
make smoke-amd64 test-amd64                       # CPU-only smoke + in-container pytest
make build-cuda                                   # CUDA image (cu126); run with --gpus all
```

### Develop in a container

`docker-compose.yml` builds a **dev** image with the full environment and bind-mounts the
working directory over `/app`, so edits on the host reflect immediately inside (argus is
installed editable — no rebuild for code changes):

```bash
# Point ARGUS_DATA at a host footage folder; it's mounted read-only at /data.
ARGUS_DATA=/home/pepe/data docker compose up -d dev   # build + start (first run installs torch etc.)
docker compose exec dev bash                          # shell in; then run argus / pytest / python
docker compose exec dev pytest -q
docker compose exec dev argus track /data/shoplifting_dataset/normal/normal-10.mp4 --render --json
docker compose down                                   # stop
```

`--render` with no path writes `out/<name>_tracked.mp4` in the repo root (bind-mounted, so it
appears on the host).

CPU by default — `dev` runs anywhere. For local GPU work (needs the NVIDIA Container Toolkit)
use the **`dev-gpu`** service (same image + a GPU reservation), built with the CUDA torch index —
the bind mount alone doesn't switch torch, so the image must be rebuilt:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu126 docker compose build dev-gpu
docker compose up -d dev-gpu
docker compose exec dev-gpu python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

With GPU torch present, `device=None` (the CLI default) auto-selects `cuda:0`.

## MCP server

Expose the triage/track workflow as tools an LLM agent can call (`list_clips`, `peek_folder`,
`peek_clip`, `track_clip`, plus `search_face` for face-ID re-identification and `classify_audio`
for sound labeling) over HTTP:

```bash
uv run argus-mcp --port 8000                 # local (localhost only)
docker compose up -d mcp                     # or containerized (mcp-gpu for GPU)
```

To reach the server **from another machine on the LAN**, allow-list this host's IP — the MCP SDK
rejects requests whose `Host` header isn't allow-listed (a DNS-rebinding guard), so binding to
`0.0.0.0` alone returns HTTP 421:

```bash
uv run argus-mcp --host 0.0.0.0 --port 8000 --allowed-hosts 192.168.1.14   # then http://192.168.1.14:8000/mcp
ARGUS_MCP_ALLOWED_HOSTS=192.168.1.14 docker compose up -d mcp              # same, containerized
```

A bare IP allows any port; pass `--insecure-disable-host-check` (env `ARGUS_MCP_INSECURE=1`) to
turn the guard off entirely on a trusted LAN. Step-by-step tutorial (local + Docker, a test
client, and connecting Claude Code): [context/mcp-server.md](context/mcp-server.md).

## Tutorial (Docker, end-to-end)

Smoke-test the CLI + MCP tools on real footage. **CPU by default — runs anywhere, incl. macOS /
Apple Silicon.** Build the images once, then mount one folder at a time at `/data`.

Build (CPU):
```bash
TORCH_INDEX=https://download.pytorch.org/whl/cpu \
  EXTRAS='[face,store,audio,open-vocab]' docker compose build dev mcp
```

CLI — tracking + audio:
```bash
ARGUS_DATA=/path/to/clips docker compose up -d dev
docker compose exec dev argus peek  /data --json                                    # triage a folder
docker compose exec dev argus track /data/CLIP.mp4 --render --json                  # detect + track
docker compose exec dev argus track /data/CLIP.mp4 --prompt person backpack --json  # open-vocab classes
docker compose exec dev argus audio /data/CLIP.mp4 --json                           # AST sound labels
docker compose exec dev argus audio /data/CLIP.mp4 --model laion/clap-htsat-unfused \
    --labels "gunshot" "glass breaking" "scream" "speech" --json                    # zero-shot CLAP
```

MCP — same mount, tools over HTTP:
```bash
ARGUS_DATA=/path/to/clips docker compose up -d mcp
uv run python examples/mcp_client_demo.py --url http://127.0.0.1:8000/mcp --dir /data
```

**GPU (NVIDIA)** — build the `*-gpu` images with the CUDA torch index and use them in place of
`dev`/`mcp` (they add the GPU reservation):
```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
  EXTRAS='[face-gpu,store,audio,open-vocab]' docker compose build dev-gpu mcp-gpu
ARGUS_DATA=/path/to/clips docker compose up -d dev-gpu    # then: docker compose exec dev-gpu argus …
```

Face-ID (people clips) is SDK ingest then MCP `search_face` — see
[context/mcp-server.md](context/mcp-server.md).

## Notes

- Displayed videos are H.264 and downscaled to 480p so they play inline; **detection
  always runs at full resolution** (`TrackingResult.render` handles the downscale +
  H.264 transcode — this OpenCV wheel has no H.264 encoder).
- Tests: `uv run pytest` (synthetic fakes; no model/GPU/data needed).
