# argus

On-prem **surveillance-footage analysis** — detect, track, and triage recorded video, then
**identify and search faces** across it. A model-agnostic Python SDK plus an **MCP server** that
exposes the whole workflow as tools an LLM agent can call. Runs entirely on-box (a single
SQLite + sqlite-vec store, no external services); CPU-portable, GPU-accelerated when available.

## Features

- **Detect & track** — YOLO11, plus open-vocabulary YOLO-World for free-text classes, with
  ByteTrack multi-object tracking.
- **Peek** — fast folder triage: which clips contain people/vehicles worth a closer look.
- **Face-ID** — detect → align → ArcFace embed → persist sightings; then cosine-KNN **search by
  face**, "more-like-this" re-ID, HDBSCAN clustering into provisional identities, and a known-person
  watchlist.
- **Audio** — per-segment sound labels (AST, or zero-shot CLAP against your own labels).
- **MCP server** — 13 HTTP tools over the same facade, gated by OAuth 2.1 scopes, with a
  compliance audit trail.
- **Three interfaces, one facade** — the Python SDK, an `argus` CLI, and the MCP server are all
  thin layers over the same functions.

Deeper docs: [context/architecture.md](context/architecture.md) (module map + how to add a
backend), [context/face-id-design.md](context/face-id-design.md) (face-ID design),
[context/mcp-server.md](context/mcp-server.md) (MCP tutorial + tool reference).

## Install & deploy (Docker)

The whole stack ships as one multi-stage image driven by `docker compose`. Configure the machine
once, then build and run the MCP server:

```bash
cp .env.example .env     # then edit (see below)
docker compose build mcp # CPU image  (GPU: see below)
docker compose up -d mcp # serves http://<host>:8000/mcp
```

Key `.env` settings:

| Var | Purpose |
|-----|---------|
| `ARGUS_DATA` | Host footage folder, bind-mounted read-only at `/data`. |
| `TORCH_INDEX` | `…/whl/cpu` (default, runs anywhere) or `…/whl/cu126` (NVIDIA GPU). |
| `EXTRAS` | Features baked in, e.g. `[face-gpu,store,cluster,audio,open-vocab]` (CPU: `face` instead of `face-gpu`). |
| `ARGUS_DB` | Sighting store path. Default `/app/out/argus.db` (under the bind-mounted `./out`), so ingested faces **persist across restarts**. |
| `ARGUS_MCP_ALLOWED_HOSTS` | Hosts allowed to reach the server (DNS-rebinding guard) when not on localhost. |
| `ARGUS_MCP_AUTH` | `off` (default) or `on` to require OAuth 2.1 bearer tokens. |

**GPU:** set `TORCH_INDEX=…/whl/cu126` + `EXTRAS=[face-gpu,…]` and use the `mcp-gpu` service
(same config + a GPU reservation): `docker compose build mcp-gpu && docker compose up -d mcp-gpu`.

**Reach it from another machine:** the MCP SDK rejects requests whose `Host` isn't allow-listed
(returns HTTP 421 from `0.0.0.0`), so set `ARGUS_MCP_ALLOWED_HOSTS=<ip>` (a bare IP allows any
port). `ARGUS_MCP_AUTH=on` adds OAuth — a Keycloak IdP service is included for testing. Full
walkthrough (auth, a test client, connecting Claude Code): [context/mcp-server.md](context/mcp-server.md).

**Without Docker:** `uv sync` (CPU torch by default), adding extras as needed:

```bash
uv sync --extra face --extra store --extra cluster --extra mcp
uv run argus-mcp --port 8000          # or: uv run argus <subcommand> …
```

## Usage

### MCP tools

13 tools over `POST /mcp`. All inputs are **server-side paths** (e.g. under `/data`):

| Group | Tools |
|-------|-------|
| **Triage / track** | `list_clips`, `peek_folder`, `peek_clip`, `track_clip` |
| **Face-ID** | `ingest_clip`, `list_sightings`, `list_identities`, `search_face` (probe image), `search_similar` (by sighting), `enroll_identity`, `cluster_sightings`, `audit_log` |
| **Audio** | `classify_audio` |

Typical agent flow: `peek_folder` → `ingest_clip` the interesting clips → `list_sightings` →
`search_face` / `search_similar` to re-identify a person → `enroll_identity` / `cluster_sightings`
to curate identities. Every search/enroll/cluster is recorded in `audit_log`. Full inputs,
outputs, and per-tool scopes: [context/mcp-server.md](context/mcp-server.md).

Smoke-test with the bundled client, or drive it from any MCP client:

```bash
uv run python examples/mcp_client_demo.py --url http://127.0.0.1:8000/mcp --dir /data
```

```python
import base64
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
    async with ClientSession(read, write) as s:
        await s.initialize()
        await s.call_tool("ingest_clip", {"path": "/data/clip.mp4", "camera_id": "cam-1"})
        # Remote clients upload the probe image bytes (server-side paths only work on the host):
        probe = base64.b64encode(open("probe.jpg", "rb").read()).decode()
        hits = await s.call_tool("search_face", {"image_base64": probe, "top_k": 10})
        print(hits.structuredContent["hits"])
```

### CLI

The `argus` command mirrors the tools (handy for scripting and smoke tests; add `--json` to any).
Run it in the container with `docker compose exec mcp argus …`:

```bash
argus peek   /data                                            # triage a folder
argus ingest /data/clip.mp4 --camera cam-1 --db /app/out/argus.db
argus ls sightings --db /app/out/argus.db
argus search --image /data/probe.jpg --db /app/out/argus.db   # or: --sighting <id>
argus track  clip.mp4 --render                                # stateless; runs locally too
```

### SDK

```python
from argus import track_video, ingest_video, search_by_image, SqliteStore

track_video("clip.mp4", targets=("person", "vehicle")).render("annotated.mp4")

store = SqliteStore("argus.db")
ingest_video("clip.mp4", "cam-1", store=store)          # detect → track → embed → persist
search_by_image("probe.jpg", store=store, top_k=10)     # ranked SearchHits
```

Any object satisfying a `core` Protocol (`Detector`, `Tracker`, `FaceDetector`, `Embedder`,
`Store`) drops in as a backend — see [context/architecture.md](context/architecture.md).

## Notes

- Detection always runs at full resolution; rendered clips are downscaled to 480p H.264.
- Tests: `uv run pytest` (synthetic fakes — no GPU/weights/data needed).
- Notebooks: `uv run marimo edit examples/01_dvr_person_tracking.py` (run from the repo root).
