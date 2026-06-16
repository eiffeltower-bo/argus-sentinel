# argus MCP server — guide & step-by-step tutorial

The MCP server exposes argus' triage/track workflow as tools an LLM agent can call. It's a thin
HTTP server (`argus/mcp/`) over the same facade the `argus` CLI wraps — no extra logic, no SDK
changes. Transport is **streamable HTTP**; the endpoint is `/mcp`.

## Tools

| Tool | What it does | Cost |
|------|--------------|------|
| `list_clips(directory, glob="*.mp4")` | Enumerate video files in a server-side folder (path + size). Discovery, no analysis. | trivial |
| `peek_folder(directory, glob, targets, n_samples, min_hits, device)` | Fast-triage a whole folder: which clips contain people/vehicles worth tracking. | cheap |
| `peek_clip(path, targets, n_samples, min_hits, device)` | Fast-triage one clip → verdict + per-category counts + `interesting` bool. | cheap |
| `track_clip(path, targets, max_frames, stride, render, device)` | Detect + track through a clip → per-track metrics; `render=true` also writes an annotated H.264 clip. | heavy |

Intended agent flow: `list_clips` → `peek_folder`/`peek_clip` (cheap) → `track_clip` only the
interesting ones. **All paths are server-side** (in Docker, footage is mounted at `/data`).
`device=None` (default) auto-selects the GPU when GPU torch is installed, else CPU.

---

## Tutorial A — run locally (fastest to test)

**1. Install (the `mcp` package ships in the dev group):**
```bash
uv sync
```

**2. Start the server** (leave it running in this terminal):
```bash
uv run argus-mcp --host 127.0.0.1 --port 8000
# -> "Uvicorn running on http://127.0.0.1:8000"
```

**3. In a second terminal, exercise it** with the bundled demo client. Point `--dir` at any
folder with `.mp4` files:
```bash
uv run python examples/mcp_client_demo.py --url http://127.0.0.1:8000/mcp \
    --dir /home/pepe/data --glob '**/*.mp4'
```
Expected output: the 4 tool names, a clip count from `list_clips`, and a `peek_clip` verdict
(JSON) for the first clip. That confirms the transport + tools end-to-end.

**4. Stop the server:** `Ctrl-C` in the first terminal.

---

## Tutorial B — run in Docker

The container ships ffmpeg and auto-downloads the YOLO weights on first use (to `/app`, needs
network once); footage is mounted read-only at `/data`. Config is read from a local `.env`
(copy `.env.example`): `ARGUS_DATA` (footage folder), `TORCH_INDEX` (CPU/CUDA build),
`ARGUS_MCP_PORT`/`ARGUS_MCP_HOST` (server bind/publish).

**CPU:**
```bash
docker compose up -d mcp                 # builds argus:mcp, serves on :8000
docker compose logs -f mcp               # watch startup (Ctrl-C to stop watching)
# from the host, same demo client against the published port:
uv run python examples/mcp_client_demo.py --url http://127.0.0.1:8000/mcp --dir /data
docker compose down                      # stop
```
`/data` is whatever `ARGUS_DATA` points at (set it in `.env` or inline). Renders land in `./out`
on the host.

**GPU** (needs the NVIDIA Container Toolkit) — must build with the CUDA torch index, then run the
GPU service:
```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu126 docker compose build mcp-gpu
docker compose up -d mcp-gpu
```
Verify the GPU is actually in use:
```bash
docker compose exec mcp-gpu python -c "import torch; print(torch.cuda.is_available())"   # -> True
```
With GPU torch present, `device=None` auto-selects `cuda:0`, so every tool runs on the GPU.

---

## Connect a real MCP client (Claude Code)

Point an MCP-capable client at the HTTP endpoint. For Claude Code:
```bash
claude mcp add --transport http argus http://localhost:8000/mcp
```
Then ask it things like *"use argus to list the clips in /data, peek the folder for vehicles,
and track the interesting ones."* Any MCP client that speaks streamable HTTP works the same way —
connect to `http://<host>:8000/mcp`.

---

## Writing your own client (snippet)

`examples/mcp_client_demo.py` is the reference; the core is:
```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("peek_clip", {"path": "/data/clip.mp4"})
            print(res.structuredContent)   # JSON-able dict; also mirrored in res.content as text

asyncio.run(main())
```
Tool results arrive both as `structuredContent` (a dict) and as text in `content`.

---

## Tool inputs & outputs

- **`list_clips`** → `{directory, glob, n_clips, clips:[{path, size_bytes}]}`.
- **`peek_folder`** → `{directory, n_clips, n_interesting, n_unreadable, clips:[verdict…], unreadable:[path…]}`.
- **`peek_clip`** (one verdict) → `{video_path, fps, width, height, total_frames, n_sampled,
  frames_with_hits, counts:{category:int}, min_hits, elapsed_s, interesting, summary}`.
- **`track_clip`** → `{video_path, fps, width, height, n_frames, n_tracks, tracks:[…per-track
  metrics…], rendered}`. Each track row: `id, category, type, first_s, last_s, duration_s,
  n_frames, continuity, avg_*, entry_edge, exit_edge`. `rendered` is the server-side path of the
  annotated clip when `render=true`, else `null`.

---

## Notes & troubleshooting

- **Paths are server-side.** In Docker, use `/data/...` (the mount), not host paths.
- **GPU not used?** The image must be *built* with the CUDA torch index (`TORCH_INDEX=…/cu126`);
  a running CPU image won't switch. Check with the `torch.cuda.is_available()` command above.
- **`render=true` permission error in Docker.** The container runs as uid 10001; if writes to the
  mounted `./out` fail, `chmod 777 out` once (or run the server as your uid).
- **Long clips.** Tools are synchronous — `track_clip` on a long clip can run for minutes and is
  bounded only by the client's request timeout. Run `peek_*` first and pass `max_frames`/`stride`.
- **No auth.** v1 has none — intended for on-prem/LAN behind a firewall. Don't expose `:8000`
  publicly without adding an auth layer.
- **Not yet exposed:** face-ID ingest and face search (those need SDK additions) — see
  [face-id-design.md](face-id-design.md).
