"""MCP server exposing argus' triage/track facade as tools (streamable HTTP).

A pure consumer of the public facade — the MCP analogue of ``argus/cli.py``. The triage/track
tools let an agent discover footage and run the cheap-then-expensive workflow (``list_clips`` ->
``peek_folder``/``peek_clip`` -> ``track_clip``), plus ``classify_audio`` for the sound track. A
face-ID group mirrors the CLI's store-backed verbs over ``ARGUS_DB``: ``ingest_clip`` populates the
sighting store; ``list_sightings``/``list_identities`` browse it; ``search_face`` (probe image) and
``search_similar`` (existing sighting) re-identify faces; ``enroll_identity`` and
``cluster_sightings`` build the identity gallery; ``audit_log`` reads the compliance trail. Run it
with:

    argus-mcp --host 0.0.0.0 --port 8000     # serves MCP over HTTP at /mcp

Tool inputs are SERVER-SIDE paths (in the container, footage is mounted read-only at /data).
``device=None`` auto-selects CUDA when GPU torch is installed, so the server inherits the warm
GPU in the CUDA image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from argus import (
    DEFAULT_AUDIO_MODEL,
    OpenVocabularyDetector,
    QualityGate,
    analyze_audio,
    audit_log as _audit_log,  # tool below is named `audit_log`; alias the facade to avoid shadowing
    enroll,
    ingest_video,
    peek_video,
    peek_videos,
    run_clustering,
    search_by_image,
    search_by_sighting,
    track_video,
)
from argus.core import TARGET_CLASSES

from ._serialize import (
    audio_to_dict,
    audit_to_dict,
    cluster_to_dict,
    identities_to_dict,
    ingest_to_dict,
    peek_to_dict,
    search_to_dict,
    sightings_to_dict,
    tracking_to_dict,
)
from .auth import (
    AuthConfig,
    TOOL_SCOPES,
    build_auth,
    require_scope,
    set_auth_enabled,
    set_tool_scopes_enabled,
)

mcp = FastMCP("argus")

_DEFAULT_TARGETS = ["person", "vehicle"]
# The fixed YOLO model's COCO groups. Targets within this set ride the fast detector; anything
# else triggers the open-vocabulary model (see _detection_kwargs).
_COCO_TARGET_NAMES = frozenset(TARGET_CLASSES)


def _detection_kwargs(targets: list[str] | None, device: str | None) -> dict[str, Any]:
    """Pick the detector for a peek/track call from the requested ``targets``.

    The default (``person``/``vehicle``) — or any subset of those COCO groups — rides the fast,
    fixed YOLO model via ``targets=``. Any *other* free-text class (e.g. ``"forklift"``,
    ``"hard hat"``) swaps in the open-vocabulary YOLO-World detector via ``detector=``, so an MCP
    agent can ask for arbitrary objects without picking a model itself. The open-vocab path needs
    the ``open-vocab`` extra (ultralytics' CLIP fork) and is heavier than the fixed model.

    ``device`` is folded into the right place: passed straight through on the fixed path, and
    handed to the detector on the open-vocab path (the facade infers device from the detector).
    """
    chosen = list(targets) if targets else list(_DEFAULT_TARGETS)
    if {t.lower() for t in chosen} <= _COCO_TARGET_NAMES:
        return {"targets": tuple(t.lower() for t in chosen), "device": device}
    return {"detector": OpenVocabularyDetector(chosen, device=device)}


def _open_store():
    """Open the face-sighting store the search tool queries (server-side, lazy).

    ``ARGUS_DB`` is the server-side DB path (default ``argus.db``); ``ARGUS_EMBED_DIM`` must match
    the embedder the footage was ingested with (512 for ArcFace). Importing ``SqliteStore`` here
    keeps the ``sqlite-vec``/``face`` extras off the import path until ``search_face`` is called.
    """
    from argus import SqliteStore

    return SqliteStore(
        os.environ.get("ARGUS_DB", "argus.db"),
        dim=int(os.environ.get("ARGUS_EMBED_DIM", "512")),
    )


@mcp.tool()
def list_clips(directory: str, glob: str = "*.mp4") -> dict[str, Any]:
    """List video clips in a server-side directory without analyzing them.

    Use this first to discover what footage is available before peeking or tracking. Paths are
    server-side/container paths (footage is typically mounted read-only at /data). Returns each
    clip's path and size in bytes.
    """
    require_scope(TOOL_SCOPES["list_clips"])
    base = Path(directory)
    clips = sorted(base.glob(glob))
    return {
        "directory": str(base),
        "glob": glob,
        "n_clips": len(clips),
        "clips": [{"path": str(p), "size_bytes": p.stat().st_size} for p in clips],
    }


@mcp.tool()
def peek_folder(
    directory: str,
    glob: str = "*.mp4",
    targets: list[str] | None = None,
    n_samples: int = 24,
    min_hits: int = 2,
    limit: int | None = 25,
    device: str | None = None,
) -> dict[str, Any]:
    """Fast-triage clips in a folder: which ones contain people/vehicles worth tracking.

    Samples a few frames per clip and runs a small detector in one batched pass — cheap relative
    to ``track_clip``. Run this to narrow a footage dump down to the interesting clips, then
    ``track_clip`` those. Returns counts of interesting/total/unreadable plus a per-clip verdict.

    Peeking is ~1s per clip, and this is ONE synchronous call — a large folder can exceed the
    client's request timeout. ``limit`` (default 25) caps how many of the matched clips are peeked
    (the first N by name); ``n_peeked`` vs ``n_matched`` in the result shows whether more remain.
    Pass ``limit=null`` to peek everything (only on small folders), or narrow ``glob`` to select a
    subset.

    ``targets`` defaults to ``["person", "vehicle"]`` on the fast fixed YOLO model. Pass other
    free-text classes (e.g. ``["forklift", "hard hat"]``) to switch to the open-vocabulary
    detector automatically (heavier; needs the ``open-vocab`` extra).
    """
    require_scope(TOOL_SCOPES["peek_folder"])
    base = Path(directory)
    matched = sorted(base.glob(glob))
    clips = matched[:limit] if limit is not None else matched
    results = peek_videos(
        clips,
        n_samples=n_samples,
        min_hits=min_hits,
        **_detection_kwargs(targets, device),
    )
    verdicts = [peek_to_dict(r) for _, r in sorted(results.items()) if r is not None]
    unreadable = [str(p) for p, r in sorted(results.items()) if r is None]
    return {
        "directory": str(base),
        "n_matched": len(matched),
        "n_peeked": len(clips),
        "truncated": len(clips) < len(matched),
        "n_interesting": sum(v["interesting"] for v in verdicts),
        "n_unreadable": len(unreadable),
        "clips": verdicts,
        "unreadable": unreadable,
    }


@mcp.tool()
def peek_clip(
    path: str,
    targets: list[str] | None = None,
    n_samples: int = 24,
    min_hits: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    """Fast-triage a single clip: does it contain target objects (person/vehicle)?

    Samples frames and reports per-category counts, frames-with-hits, and an ``interesting``
    boolean. Much cheaper than ``track_clip`` — use it to decide whether full tracking is worth it.

    ``targets`` defaults to ``["person", "vehicle"]`` on the fast fixed YOLO model. Pass other
    free-text classes (e.g. ``["forklift", "hard hat"]``) to switch to the open-vocabulary
    detector automatically (heavier; needs the ``open-vocab`` extra).
    """
    require_scope(TOOL_SCOPES["peek_clip"])
    r = peek_video(
        Path(path),
        n_samples=n_samples,
        min_hits=min_hits,
        **_detection_kwargs(targets, device),
    )
    return peek_to_dict(r)


@mcp.tool()
def track_clip(
    path: str,
    targets: list[str] | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    render: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Detect and track objects through a single clip, returning per-track metrics.

    Each track row has id, category, type, first_s/last_s, duration_s, n_frames, continuity,
    average size/confidence, and entry/exit edge. This is HEAVY (decodes the whole clip and
    detects every frame) — run ``peek_clip`` first to confirm the clip is interesting, and use
    ``max_frames``/``stride`` to bound work on long clips. Set ``render=true`` to also write an
    annotated H.264 video; its server-side path is returned under ``rendered``.

    ``targets`` defaults to ``["person", "vehicle"]`` on the fast fixed YOLO model. Pass other
    free-text classes (e.g. ``["forklift", "hard hat"]``) to switch to the open-vocabulary
    detector automatically (heavier; needs the ``open-vocab`` extra).
    """
    require_scope(TOOL_SCOPES["track_clip"])
    r = track_video(
        Path(path),
        max_frames=max_frames,
        stride=stride,
        **_detection_kwargs(targets, device),
    )
    rendered = None
    if render:
        rendered = str(r.render(Path("out") / f"{Path(path).stem}_tracked.mp4"))
    return tracking_to_dict(r, rendered)


@mcp.tool()
def search_face(
    image: str,
    top_k: int = 20,
    cameras: list[str] | None = None,
    since: float | None = None,
    min_quality: float = 0.0,
    device: str | None = None,
    actor: str = "mcp",
) -> dict[str, Any]:
    """Find where a face appears across already-ingested footage (face-ID re-identification).

    ``image`` is a server-side path to a probe photo; its strongest face is embedded and matched
    against the sighting store (footage must have been ingested first). Optionally filter by
    ``cameras``, ``since`` (video timestamp seconds), and ``min_quality``. Returns up to ``top_k``
    ranked hits, each with a cosine ``score`` in [0,1] and a ``chip_path`` to the matched face for
    review. These are CANDIDATES for human adjudication, never an automated identity decision.
    Needs the ``face`` + ``store`` extras and a populated DB (``ARGUS_DB``); the search is audited.
    """
    require_scope(TOOL_SCOPES["search_face"])
    store = _open_store()
    try:
        hits = search_by_image(
            image,
            store=store,
            top_k=top_k,
            cameras=cameras,
            since=since,
            min_quality=min_quality,
            device=device,
            actor=actor,
        )
        return search_to_dict(image, hits)
    finally:
        store.close()


@mcp.tool()
def ingest_clip(
    path: str,
    camera_id: str,
    device: str | None = None,
    conf: float = 0.25,
    stride: int = 1,
    face_stride: int = 1,
    max_frames: int | None = None,
    min_face_px: float | None = None,
    min_blur_var: float | None = None,
    max_yaw_ratio: float | None = None,
    min_det_score: float | None = None,
) -> dict[str, Any]:
    """Ingest one clip into the face store: detect→track people, embed the best face per track.

    Writes one ``Sighting`` (512-d ArcFace vector + metadata + an aligned face chip) per tracked
    person to ``ARGUS_DB`` — the footage ``search_face``/``search_similar`` then query. HEAVY: it
    decodes the whole clip and runs detection + the face stage; bound long clips with
    ``max_frames``/``stride``. The quality-gate floors (``min_face_px``/``min_blur_var``/
    ``max_yaw_ratio``/``min_det_score``) default to the calibrated values when left null — lower the
    size/blur floors to admit smaller, softer faces on distant-camera footage. ``path`` is
    server-side. Needs the ``face`` + ``store`` extras.
    """
    require_scope(TOOL_SCOPES["ingest_clip"])
    overrides = {
        "min_face_px": min_face_px,
        "min_blur_var": min_blur_var,
        "max_yaw_ratio": max_yaw_ratio,
        "min_det_score": min_det_score,
    }
    gate = QualityGate(**{k: v for k, v in overrides.items() if v is not None})
    store = _open_store()
    try:
        r = ingest_video(
            Path(path),
            camera_id,
            store=store,
            gate=gate,
            device=device,
            conf=conf,
            stride=stride,
            face_stride=face_stride,
            max_frames=max_frames,
        )
        return ingest_to_dict(r)
    finally:
        store.close()


@mcp.tool()
def list_sightings(
    cameras: list[str] | None = None,
    min_quality: float = 0.0,
    limit: int | None = 100,
) -> dict[str, Any]:
    """List stored face sightings (metadata only) from ``ARGUS_DB`` for inspection.

    Each row carries camera/timestamp/quality and the evidence ``chip_path`` — no embedding vector.
    Optionally filter by ``cameras`` / ``min_quality``; ``limit`` caps the rows returned (default
    100, pass null for all). Use this to discover sighting ids to feed ``search_similar``.
    """
    require_scope(TOOL_SCOPES["list_sightings"])
    store = _open_store()
    try:
        rows = store.list_sightings()
        if cameras:
            rows = [s for s in rows if s["camera_id"] in cameras]
        if min_quality > 0.0:
            rows = [s for s in rows if (s["quality"] or 0.0) >= min_quality]
        if limit is not None:
            rows = rows[:limit]
        return sightings_to_dict(rows)
    finally:
        store.close()


@mcp.tool()
def list_identities(type: str | None = None) -> dict[str, Any]:
    """List stored identities (``known`` enrolled + ``provisional`` clusters) from ``ARGUS_DB``.

    Optionally filter by ``type`` (``"known"`` or ``"provisional"``). Identities are created by
    ``enroll_identity`` (known) and ``cluster_sightings`` (provisional).
    """
    require_scope(TOOL_SCOPES["list_identities"])
    store = _open_store()
    try:
        return identities_to_dict(store.list_identities(type=type))
    finally:
        store.close()


@mcp.tool()
def search_similar(
    sighting_id: int,
    top_k: int = 20,
    cameras: list[str] | None = None,
    since: float | None = None,
    min_quality: float = 0.0,
    actor: str = "mcp",
) -> dict[str, Any]:
    """Find more sightings of the person in an existing sighting ("more like this").

    Uses the stored embedding of ``sighting_id`` (no probe image needed) and excludes that sighting
    from the results. Same filters/ranking as ``search_face``; returns ranked hits with cosine
    ``score`` + evidence ``chip_path`` for human review. The search is audited.
    """
    require_scope(TOOL_SCOPES["search_similar"])
    store = _open_store()
    try:
        hits = search_by_sighting(
            sighting_id,
            store=store,
            top_k=top_k,
            cameras=cameras,
            since=since,
            min_quality=min_quality,
            actor=actor,
        )
        return search_to_dict(f"sighting:{sighting_id}", hits)
    finally:
        store.close()


@mcp.tool()
def enroll_identity(
    label: str,
    images: list[str],
    source: str = "id_photo",
    device: str | None = None,
    actor: str = "mcp",
) -> dict[str, Any]:
    """Enroll a known person into the watchlist gallery from one or more face photos.

    Creates a ``known`` identity, embeds each image's strongest face, and stores an enrollment (with
    its aligned chip). ``images`` are server-side paths. Returns the new identity id. Audited; needs
    the ``face`` + ``store`` extras.
    """
    require_scope(TOOL_SCOPES["enroll_identity"])
    store = _open_store()
    try:
        identity_id = enroll(
            label,
            [Path(i) for i in images],
            store=store,
            source=source,
            device=device,
            actor=actor,
        )
        return {"identity_id": identity_id, "label": label, "n_images": len(images)}
    finally:
        store.close()


@mcp.tool()
def cluster_sightings(
    space_id: str = "arcface_w600k_r50_v1",
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    include_assigned: bool = False,
    actor: str = "mcp",
) -> dict[str, Any]:
    """Group unlabeled sightings into provisional identities (HDBSCAN over the embeddings).

    Each dense cluster becomes a ``provisional`` identity an operator can later name or merge.
    ``space_id`` is the embedding space (ArcFace by default). By default only not-yet-identified
    sightings are clustered; set ``include_assigned`` to cluster all. Audited; needs the ``cluster``
    extra (scikit-learn).
    """
    require_scope(TOOL_SCOPES["cluster_sightings"])
    store = _open_store()
    try:
        r = run_clustering(
            store=store,
            space_id=space_id,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            only_unassigned=not include_assigned,
            actor=actor,
        )
        return cluster_to_dict(r)
    finally:
        store.close()


@mcp.tool()
def audit_log(actor: str | None = None, since: str | None = None) -> dict[str, Any]:
    """Read the compliance audit trail (every search/enroll/cluster/assignment is logged).

    Optionally filter by ``actor`` and/or ``since`` (ISO timestamp; rows with ``ts >= since``).
    """
    require_scope(TOOL_SCOPES["audit_log"])
    store = _open_store()
    try:
        return audit_to_dict(_audit_log(store=store, actor=actor, since=since))
    finally:
        store.close()


@mcp.tool()
def classify_audio(
    path: str,
    model: str = DEFAULT_AUDIO_MODEL,
    overlap_seconds: float = 1.0,
    segment_seconds: float = 5.0,
    top_k: int = 2,
    candidate_labels: list[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Classify the audio track of a server-side clip into per-segment sound labels.

    Extracts 16kHz mono audio, windows it into overlapping ``segment_seconds`` segments, and runs
    an audio model on each. The default is zero-shot CLAP: it scores each window against
    ``candidate_labels`` (a surveillance-oriented default set if you pass none), so to look for
    specific sounds just pass your own labels, e.g. ``["gunshot", "glass breaking", "speech"]``.
    Pass a fixed-label ``model`` (e.g. ``"bioamla/ast-esc50"``) to use its built-in taxonomy
    instead. Returns per-segment top-k ``{class, confidence}`` predictions with time spans. Needs
    the ``audio`` extra (transformers + soundfile).
    """
    require_scope(TOOL_SCOPES["classify_audio"])
    return audio_to_dict(
        analyze_audio(
            Path(path),
            model=model,
            overlap_seconds=overlap_seconds,
            segment_seconds=segment_seconds,
            top_k=top_k,
            candidate_labels=candidate_labels,
            device=device,
        )
    )


# DNS-rebinding allow-list entries that always work for a local client (FastMCP's own default).
_LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _csv(value: str | None) -> list[str]:
    """Split a comma-separated CLI/env value into a clean list."""
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _transport_security(
    allowed_hosts: list[str], allowed_origins: list[str], insecure: bool
) -> TransportSecuritySettings | None:
    """Build DNS-rebinding settings for a LAN-exposed server.

    The MCP SDK rejects any request whose ``Host`` header isn't in an allow-list (a DNS-rebinding
    guard), and FastMCP seeds that list with localhost only — so a server bound to ``0.0.0.0`` is
    still unreachable from another machine (HTTP 421) until its LAN host is allowed. We always keep
    the localhost entries and add the operator's hosts on top:

    - ``allowed_hosts`` entries without a port get a ``:*`` companion (match any port), so
      ``--allowed-hosts 192.168.1.14`` just works; entries with a port are taken verbatim.
    - ``insecure`` turns the guard off entirely (any Host accepted) — only for a trusted LAN.

    Returns ``None`` when nothing LAN-related was requested, leaving FastMCP's localhost default.
    """
    if insecure:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    if not allowed_hosts and not allowed_origins:
        return None
    hosts = list(_LOCAL_HOSTS)
    origins = list(_LOCAL_ORIGINS)
    for h in allowed_hosts:
        hosts.append(h)
        if ":" not in h:  # bare host -> allow any port + a matching http origin
            hosts.append(f"{h}:*")
            origins.append(f"http://{h}:*")
        else:
            origins.append(f"http://{h}")
    origins.extend(allowed_origins)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


# The tool functions, in registration order. Used to re-register them on a fresh FastMCP instance
# when auth is enabled (the module-level ``mcp`` has no token verifier). Keep in sync with
# ``TOOL_SCOPES`` in auth.py (tests/test_mcp.py::test_all_tools_have_scopes enforces this).
_TOOLS = [
    list_clips,
    peek_folder,
    peek_clip,
    track_clip,
    search_face,
    search_similar,
    ingest_clip,
    list_sightings,
    list_identities,
    enroll_identity,
    cluster_sightings,
    audit_log,
    classify_audio,
]


def build_server(cfg: AuthConfig) -> FastMCP:
    """Return the FastMCP instance to serve, applying OAuth when ``cfg.enabled``.

    Auth off: reuse the module-level ``mcp`` (tools already registered) — identical to before.
    Auth on: build a fresh FastMCP with the JWT ``token_verifier`` + ``AuthSettings`` (so the SDK
    serves the RFC 9728 metadata + ``401`` challenge and enforces any blanket scopes) and
    re-register the same tool functions on it.
    """
    verifier, settings = build_auth(cfg)
    if verifier is None:
        return mcp
    server = FastMCP("argus", token_verifier=verifier, auth=settings)
    for fn in _TOOLS:
        server.tool()(fn)
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus-mcp", description="argus MCP server (HTTP)")
    ap.add_argument("--host", default=os.environ.get("ARGUS_MCP_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ARGUS_MCP_PORT", "8000")))
    ap.add_argument(
        "--allowed-hosts",
        default=os.environ.get("ARGUS_MCP_ALLOWED_HOSTS", ""),
        help="comma-separated Host values to accept when exposing on the LAN (DNS-rebinding "
        "allow-list), e.g. '192.168.1.14' (any port) or '192.168.1.14:8765'. localhost is "
        "always allowed.",
    )
    ap.add_argument(
        "--allowed-origins",
        default=os.environ.get("ARGUS_MCP_ALLOWED_ORIGINS", ""),
        help="comma-separated Origin values to accept (only needed for browser-based clients)",
    )
    ap.add_argument(
        "--insecure-disable-host-check",
        action="store_true",
        default=os.environ.get("ARGUS_MCP_INSECURE", "") not in ("", "0", "false"),
        help="disable DNS-rebinding (Host/Origin) protection entirely — only on a trusted LAN",
    )
    args = ap.parse_args(argv)

    # Build the server with OAuth applied iff ARGUS_MCP_AUTH=on. set_auth_enabled toggles the
    # per-tool scope checks (no-op when off, so the loopback/dev path is unchanged).
    cfg = AuthConfig.from_env()
    set_auth_enabled(cfg.enabled)
    # Per-tool scope checks default on; set ARGUS_MCP_TOOL_SCOPES=off to require only a valid token
    # (DCR clients' tokens often lack the argus:* scopes).
    set_tool_scopes_enabled(
        os.environ.get("ARGUS_MCP_TOOL_SCOPES", "on").strip().lower() in ("on", "1", "true", "yes")
    )
    server = build_server(cfg)
    if cfg.enabled:
        print(
            f"auth: ON — issuer={cfg.issuer} resource={cfg.resource} "
            f"blanket_scopes={cfg.scopes or '(none; per-tool only)'}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "auth: OFF (set ARGUS_MCP_AUTH=on to require OAuth tokens)",
            file=sys.stderr,
            flush=True,
        )

    server.settings.host = args.host
    server.settings.port = args.port
    # Return tool results as a single JSON response rather than an SSE stream. Browser-based MCP
    # clients (e.g. MCPJam) read a complete JSON body and proceed immediately, whereas a streamed
    # SSE response to the POST can stall them. Our tools return final results (no incremental
    # progress), so nothing is lost. Override with ARGUS_MCP_JSON_RESPONSE=off to force SSE.
    server.settings.json_response = os.environ.get(
        "ARGUS_MCP_JSON_RESPONSE", "on"
    ).strip().lower() in ("on", "1", "true", "yes")
    # Stateless mode: don't require an Mcp-Session-Id header on follow-up requests (each request is
    # self-contained). Some browser MCP clients don't echo the session id, so they stall after
    # initialize against a stateful server. Our tools hold no per-session state, so this is safe.
    # Default off (spec-standard stateful); enable with ARGUS_MCP_STATELESS=on.
    server.settings.stateless_http = os.environ.get(
        "ARGUS_MCP_STATELESS", "off"
    ).strip().lower() in ("on", "1", "true", "yes")

    allowed_hosts = _csv(args.allowed_hosts)
    allowed_origins = _csv(args.allowed_origins)
    ts = _transport_security(allowed_hosts, allowed_origins, args.insecure_disable_host_check)
    if ts is not None:
        server.settings.transport_security = ts
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        # Bound to a non-loopback address but no allow-list given: FastMCP locked the guard to
        # localhost at construction time, so LAN clients will hit 421. Fail closed, but say why.
        print(
            f"WARNING: serving on {args.host} but only localhost is allowed; LAN clients will be "
            "rejected (HTTP 421). Pass --allowed-hosts <ip> (or --insecure-disable-host-check).",
            file=sys.stderr,
            flush=True,
        )

    # Serve over streamable HTTP. We build the ASGI app ourselves (instead of server.run()) to layer
    # CORS on top: browser-based MCP clients (e.g. the MCPJam inspector) send a cross-origin
    # preflight OPTIONS before each call, and the auth layer wraps the /mcp route — so without CORS
    # the preflight is 401'd with no Access-Control-* headers and the browser blocks the connection.
    # CORS as the outermost middleware answers the preflight before auth runs; the bearer token and
    # the DNS-rebinding Host/Origin guard remain the actual access controls.
    import uvicorn
    from starlette.middleware.cors import CORSMiddleware

    app = server.streamable_http_app()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id", "WWW-Authenticate"],
    )
    uvicorn.run(app, host=server.settings.host, port=server.settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
