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
| `search_face(image, top_k, cameras, since, min_quality, device, actor)` | Re-identify a probe face across **already-ingested** footage → ranked hits (cosine `score` + evidence `chip_path`). Candidates for human review, never an automated match. | medium |
| `ingest_clip(path, camera_id, device, conf, stride, face_stride, max_frames, min_face_px, min_blur_var, max_yaw_ratio, min_det_score)` | Populate the sighting store: detect→track→embed the best face per track → persist. The footage `search_face`/`search_similar` query. Heavy; bound with `max_frames`/`stride`. Needs `face`+`store`. | heavy |
| `search_similar(sighting_id, top_k, cameras, since, min_quality, actor)` | "More like this": find more sightings of the person in an existing sighting (uses its stored embedding — no probe image). | medium |
| `list_sightings(cameras, min_quality, limit)` | List stored sightings (metadata + evidence `chip_path`, no vectors). Discover sighting ids for `search_similar`. | cheap |
| `list_identities(type)` | List identities — `known` (enrolled) + `provisional` (clusters); filter by `type`. | cheap |
| `enroll_identity(label, images, source, device, actor)` | Enroll a known person from face photos into the watchlist gallery → new identity id. Needs `face`+`store`. | medium |
| `cluster_sightings(space_id, min_cluster_size, min_samples, include_assigned, actor)` | Group unlabeled sightings into provisional identities (HDBSCAN). Needs the `cluster` extra. | medium |
| `audit_log(actor, since)` | Read the compliance audit trail (every search/enroll/cluster/assignment is logged). | trivial |
| `classify_audio(path, model, overlap_seconds, segment_seconds, top_k, candidate_labels, device)` | Classify a clip's **audio** track into per-segment sound labels (AST/ESC-50, or zero-shot CLAP via `candidate_labels`). Needs the `audio` extra. | medium |

Intended agent flow: `list_clips` → `peek_folder`/`peek_clip` (cheap) → `track_clip` only the
interesting ones. The face-ID path runs over the sighting DB (`ARGUS_DB`): `ingest_clip` populates
it, then `list_sightings`/`list_identities` browse it, `search_face` (probe image) and
`search_similar` (existing sighting) re-identify faces, `enroll_identity`/`cluster_sightings` build
the identity gallery, and `audit_log` reads the compliance trail — so footage must be ingested
before search returns anything. `classify_audio` is the audio path: it extracts and labels the
sound track of a clip directly (no ingest needed).
**All paths are server-side** (in Docker, footage is mounted at `/data`). `device=None` (default)
auto-selects the GPU when GPU torch is installed, else CPU.

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
Expected output: the 6 tool names, a clip count from `list_clips`, and a `peek_clip` verdict
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

## Authentication (OAuth 2.1)

Auth is **off by default**. When enabled, the server acts as an OAuth 2.1 **Resource Server (RS)**:
it validates the bearer token an MCP client attaches to every request and gates each tool by scope.
Tokens are *issued* by a separate **Authorization Server (IdP)** — here **Keycloak** — which runs
the user login/consent/PKCE dance. The MCP SDK does the discovery plumbing for us: it serves the
RFC 9728 Protected Resource Metadata document and the `401 WWW-Authenticate` challenge from
`AuthSettings`; argus only implements the token verifier (`argus/mcp/auth.py`,
`JwtTokenVerifier` — RS256 over the IdP's JWKS, checking `iss` + `aud` + `exp`).

**Toggle & config** (env, see `.env.example`):

| Var | Meaning |
|-----|---------|
| `ARGUS_MCP_AUTH` | `off` (default) or `on`. On → every request needs a valid token. |
| `ARGUS_OAUTH_ISSUER` | IdP base URL (token issuer), e.g. `http://localhost:8080/realms/argus`. |
| `ARGUS_OAUTH_RESOURCE` | This server's canonical URI; tokens must be audience-bound to it (RFC 8707), e.g. `http://localhost:8000/mcp`. |
| `ARGUS_OAUTH_SCOPES` | Optional *blanket* scope(s) the SDK requires on every request (blank = none). |

**Scopes are per-tool.** Beyond a valid token, each tool needs its own scope:
`argus:peek` (`list_clips`/`peek_folder`/`peek_clip`), `argus:track` (`track_clip`),
`argus:search` (`search_face`/`search_similar`), `argus:audio` (`classify_audio`),
`argus:read` (`list_sightings`/`list_identities`), `argus:audit` (`audit_log` — note: distinct
from `argus:audio`), and the write scopes `argus:ingest` (`ingest_clip`), `argus:enroll`
(`enroll_identity`), `argus:cluster` (`cluster_sightings`). A valid token missing a tool's
scope yields an **MCP tool error** ("insufficient scope") — not an HTTP `403` (all tools share the
one `POST /mcp` route). HTTP `403 insufficient_scope` is reserved for the SDK's blanket
`ARGUS_OAUTH_SCOPES`.

**Test loop (Keycloak + MCPJam).** Bring up the IdP, run the RS, then drive the flow:
```bash
docker compose up -d keycloak          # imports the `argus` realm; console http://localhost:8080 (admin/admin)
# Run the RS on the host (simplest — see the hostname note) with auth on:
ARGUS_MCP_AUTH=on \
  ARGUS_OAUTH_ISSUER=http://localhost:8080/realms/argus \
  ARGUS_OAUTH_RESOURCE=http://localhost:8000/mcp \
  uv run argus-mcp --host 127.0.0.1 --port 8000
# In another terminal, the required conformance tool — the MCPJam OAuth Debugger:
npx @mcpjam/inspector@latest           # open the printed localhost URL, point it at http://localhost:8000/mcp
```
In the inspector's **OAuth Debugger**, run the guided flow — it walks PRM **discovery (RFC 9728)** →
AS metadata → client registration → **authorization redirect + PKCE** → **token exchange** →
**authenticated MCP request**, with conformance checks per spec version. Log in as the realm's test
user **`analyst` / `analyst`**. Green across steps = pass.

The realm (`deploy/keycloak/realm-argus.json`) ships an `argus-mcp` public PKCE client, the four
`argus:*` scopes (granted to the test user by default), an **audience mapper** binding tokens to
`http://localhost:8000/mcp`, and the `analyst` user.

**Quick checks without a browser:**
```bash
# PRM document points at the IdP:
curl -s http://localhost:8000/.well-known/oauth-protected-resource/mcp
# No token -> 401 + WWW-Authenticate with resource_metadata:
curl -i -X POST http://localhost:8000/mcp -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
# Mint a token directly (Keycloak direct-access grant) and call with it:
TOKEN=$(curl -s -X POST http://localhost:8080/realms/argus/protocol/openid-connect/token \
  -d grant_type=password -d client_id=argus-mcp -d username=analyst -d password=analyst -d scope=openid \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -i -X POST http://localhost:8000/mcp -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/json, text/event-stream' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```
Expected: PRM lists `authorization_servers: [http://localhost:8080/realms/argus]`; no/garbage/expired/
wrong-audience token → `401`; valid token → `200` + an `mcp-session-id` header.

**Hostname note (the one fiddly bit).** The issuer URL must resolve *identically* from MCPJam (on
the host) and from the RS — otherwise JWKS fetch or issuer validation fails. The simplest loop runs
Keycloak in compose (`localhost:8080`) and the **RS on the host** (above), so both see
`localhost:8080`. Running the RS in the `mcp` container instead needs the issuer reachable under the
same name inside the container (a shared network alias / `extra_hosts`); use the host-RS loop unless
you specifically need the containerized RS.

**Production.** Use short token lifetimes, terminate **TLS** at a reverse proxy (Caddy/nginx) for
any non-loopback deployment (the spec requires HTTPS for AS endpoints and non-localhost redirects),
and point `ARGUS_OAUTH_ISSUER` at your real IdP (a managed IdP — Auth0/Okta/WorkOS — is a drop-in
alternative to Keycloak as long as it issues JWT access tokens with the right audience).

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
- **`search_face`** / **`search_similar`** → `{query, n_hits, hits:[…]}`. Each hit: `sighting_id,
  score` (cosine, 0–1), `distance, camera_id, ts, video_id, track_id, frame_idx, bbox, quality,
  chip_path, identity_id, cluster_id`. `chip_path` is the server-side aligned-face image for operator
  review. (`search_similar`'s `query` is `"sighting:<id>"`.)
- **`ingest_clip`** → `{video_id, video_path, n_frames, n_tracks, n_faces_detected, n_gated_out,
  n_sightings, avg_quality, summary}`.
- **`list_sightings`** → `{n, sightings:[{id, video_id, camera_id, track_id, frame_idx, ts, x1, y1,
  x2, y2, quality, chip_path, embedding_space_id, identity_id, cluster_id, created_at}, …]}` (no
  embedding vector).
- **`list_identities`** → `{n, identities:[{id, type, label, created_by, created_at, notes}, …]}`.
- **`enroll_identity`** → `{identity_id, label, n_images}`.
- **`cluster_sightings`** → `{n_sightings, n_clusters, n_noise, run_id, identity_ids:[…], summary}`.
- **`audit_log`** → `{n, rows:[{id, actor, action, target_type, target_id, query_ref, details, ts}, …]}`.
- **`classify_audio`** → `{input_file, audio_path, input_duration_seconds, model_name,
  overlap_seconds, segment_seconds, segments:[…]}`. Each segment: `{segment_index, start_time,
  end_time, predictions:[{class, confidence}, …]}` (top-`top_k`, ranked best-first).

---

## Notes & troubleshooting

- **Paths are server-side.** In Docker, use `/data/...` (the mount), not host paths.
- **GPU not used?** The image must be *built* with the CUDA torch index (`TORCH_INDEX=…/cu126`);
  a running CPU image won't switch. Check with the `torch.cuda.is_available()` command above.
- **`render=true` permission error in Docker.** The container runs as uid 10001; if writes to the
  mounted `./out` fail, `chmod 777 out` once (or run the server as your uid).
- **Long clips.** Tools are synchronous — `track_clip` on a long clip can run for minutes and is
  bounded only by the client's request timeout. Run `peek_*` first and pass `max_frames`/`stride`.
- **HTTP 421 from another machine.** The MCP SDK rejects requests whose `Host` header isn't
  allow-listed (a DNS-rebinding guard seeded with localhost only), so binding `--host 0.0.0.0`
  isn't enough to reach the server over the LAN. Allow-list this host's IP:
  `argus-mcp --host 0.0.0.0 --allowed-hosts 192.168.1.14` (a bare IP allows any port), or
  `ARGUS_MCP_ALLOWED_HOSTS=192.168.1.14 docker compose up -d mcp`. `--insecure-disable-host-check`
  (env `ARGUS_MCP_INSECURE=1`) turns the guard off entirely on a trusted LAN.
- **Auth is opt-in.** Off by default (loopback/LAN behind a firewall). To require OAuth tokens,
  see [Authentication (OAuth 2.1)](#authentication-oauth-21) below. Don't expose `:8000` publicly
  without enabling it (or fronting the server with an auth proxy + TLS).
- **Not yet exposed:** face-ID ingest and face search (those need SDK additions) — see
  [face-id-design.md](face-id-design.md).
