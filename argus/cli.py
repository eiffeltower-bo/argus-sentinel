"""Thin command-line surface over the argus facade — the container entrypoint.

Stateless analysis (no store):

    argus peek  FOLDER  [--targets ... | --prompt TEXT...] [--glob '*.mp4'] [--device cuda] [--workers N] [--json]
    argus track VIDEO   [--targets ... | --prompt TEXT...] [--device cuda] [--max-frames N] [--render OUT] [--json]
    argus audio CLIP    [--model M] [--overlap S] [--segment S] [--labels ...] [--device cuda] [--json]

Face-ID over a persistent store (all take ``--db PATH``, default ``./argus.db``):

    argus ingest VIDEO  --camera ID [--device cuda] [--stride N] [--max-frames N] [--json]
    argus ls    [videos|sightings|identities]  [--camera ...] [--min-quality F] [--json]
    argus search (--image PATH | --sighting ID) [--top-k N] [--cameras ...] [--since S] [--min-quality F] [--json]
    argus enroll LABEL IMAGE [IMAGE ...] [--source S] [--device cuda] [--json]
    argus cluster [--space-id ID] [--min-cluster-size N] [--include-assigned] [--json]
    argus audit  [--actor NAME] [--since ISO] [--json]

``--prompt`` swaps the fixed COCO detector for the open-vocabulary YOLO-World detector
(detect one or more free-text classes, e.g. ``--prompt forklift "hard hat"``), overriding
``--targets``.

It is a pure dispatch layer: it imports only the cheap re-exports from ``argus`` (heavy libs
stay lazy inside the backends) and adds no logic of its own — the CLI analogue of the MCP server,
calling the same ``ingest_video``/``search_by_image``/``enroll``/… facade functions. ``--json``
emits one machine-readable object to stdout (handy for smoke tests and scripting).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from argus import (
    DEFAULT_AUDIO_MODEL,
    OpenVocabularyDetector,
    QualityGate,
    analyze_audio,
    audit_log,
    enroll,
    ingest_video,
    peek_videos,
    run_clustering,
    search_by_image,
    search_by_sighting,
    track_video,
)

_TARGETS = ("person", "vehicle")
_DEFAULT_DB = "argus.db"
_DEFAULT_SPACE_ID = "arcface_w600k_r50_v1"  # ArcFace (insightface) — the default embedder's space
_GATE = QualityGate()  # calibrated defaults; ingest exposes each threshold as a flag


def _add_store(ap: argparse.ArgumentParser) -> None:
    """Options shared by every store-backed subcommand: where the DB is + machine output."""
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(_DEFAULT_DB),
        help=f"sqlite store path (default: ./{_DEFAULT_DB})",
    )
    ap.add_argument(
        "--dim",
        type=int,
        default=512,
        help="embedding dim; must match the embedder (512 for ArcFace)",
    )
    ap.add_argument("--json", action="store_true", help="emit one JSON object to stdout")


def _open_store(args: argparse.Namespace):
    """Open the sqlite-vec store the face-ID commands read/write (lazy import of the extra)."""
    from argus import SqliteStore

    return SqliteStore(args.db, dim=args.dim)


def _hit_dict(hit) -> dict:
    """One ``SearchHit`` as a JSON-able row: similarity + the matched sighting's evidence."""
    s = hit.sighting
    return {
        "sighting_id": s.id,
        "score": hit.score,
        "distance": hit.distance,
        "camera_id": s.camera_id,
        "ts": s.ts,
        "video_id": s.video_id,
        "quality": s.quality,
        "chip_path": s.chip_path,
        "identity_id": s.identity_id,
    }


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--targets",
        nargs="+",
        default=list(_TARGETS),
        choices=list(_TARGETS),
        help="COCO class groups that count as interesting",
    )
    ap.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto-detect)")
    ap.add_argument(
        "--prompt",
        nargs="+",
        default=None,
        help="open-vocabulary text class(es) (YOLO-World); overrides --targets",
    )
    ap.add_argument("--json", action="store_true", help="emit one JSON object to stdout")


def _peek(args: argparse.Namespace) -> int:
    clips = sorted(args.folder.glob(args.glob))
    if not clips:
        raise SystemExit(f"no files matching {args.glob!r} in {args.folder}")

    if args.prompt:
        det = OpenVocabularyDetector(args.prompt, device=args.device)
        results = peek_videos(clips, detector=det, max_workers=args.workers)
    else:
        results = peek_videos(
            clips, targets=tuple(args.targets), device=args.device, max_workers=args.workers
        )
    interesting = {p: r for p, r in results.items() if r and r.interesting}
    unreadable = [p for p, r in results.items() if r is None]

    if args.json:
        print(
            json.dumps(
                {
                    "clips": len(clips),
                    "interesting": [
                        {"path": str(p), "counts": r.counts, "summary": r.summary()}
                        for p, r in sorted(interesting.items())
                    ],
                    "unreadable": [str(p) for p in unreadable],
                }
            )
        )
        return 0

    print(f"{len(interesting)} of {len(clips)} clips look interesting:")
    for p, r in sorted(interesting.items()):
        print(f"  {p.name:42s} {r.summary()}")
    if unreadable:
        print(f"{len(unreadable)} unreadable: " + ", ".join(p.name for p in unreadable))
    return 0


def _track(args: argparse.Namespace) -> int:
    if args.prompt:
        det = OpenVocabularyDetector(args.prompt, device=args.device)
        result = track_video(args.video, detector=det, max_frames=args.max_frames)
    else:
        result = track_video(
            args.video,
            targets=tuple(args.targets),
            device=args.device,
            max_frames=args.max_frames,
        )
    rendered = None
    if args.render is not None:
        out = Path(args.render) if args.render else Path("out") / f"{args.video.stem}_tracked.mp4"
        rendered = str(result.render(out))

    if args.json:
        print(
            json.dumps(
                {
                    "video": str(args.video),
                    "n_frames": len(result.frames),
                    "n_tracks": len(result.track_ids),
                    "tracks": result.metrics().to_dicts(),
                    "rendered": rendered,
                }
            )
        )
        return 0

    print(
        f"{args.video.name}: {len(result.frames)} frames · {len(result.track_ids)} distinct tracks"
    )
    print(
        result.metrics().select(
            "id", "category", "type", "first_s", "last_s", "duration_s", "n_frames", "avg_conf"
        )
    )
    if rendered is not None:
        print(f"rendered -> {rendered}")
    return 0


def _audio(args: argparse.Namespace) -> int:
    result = analyze_audio(
        args.clip,
        model=args.model,
        overlap_seconds=args.overlap,
        segment_seconds=args.segment,
        top_k=args.top_k,
        candidate_labels=args.labels,
        device=args.device,
    )
    if args.json:
        print(json.dumps(result.to_dict()))
        return 0

    print(result.summary())
    print(result.metrics())
    return 0


def _ingest(args: argparse.Namespace) -> int:
    camera = args.camera or args.video.stem
    gate = QualityGate(
        min_face_px=args.min_face_px,
        min_blur_var=args.min_blur_var,
        max_yaw_ratio=args.max_yaw_ratio,
        min_det_score=args.min_det_score,
    )
    store = _open_store(args)
    try:
        result = ingest_video(
            args.video,
            camera,
            store=store,
            gate=gate,
            device=args.device,
            conf=args.conf,
            stride=args.stride,
            face_stride=args.face_stride,
            max_frames=args.max_frames,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "video_id": result.video_id,
                        "video_path": str(result.video_path),
                        "camera_id": camera,
                        "n_frames": result.n_frames,
                        "n_tracks": result.n_tracks,
                        "n_faces_detected": result.n_faces_detected,
                        "n_gated_out": result.n_gated_out,
                        "n_sightings": result.n_sightings,
                        "avg_quality": result.avg_quality,
                        "db": str(args.db),
                    }
                )
            )
            return 0
        print(result.summary())
        print(f"store: {args.db}  ({store.count_vectors()} vectors)")
        return 0
    finally:
        store.close()


def _ls(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        if args.what == "videos":
            rows = store.list_videos()
            if args.json:
                print(json.dumps(rows))
                return 0
            print(f"{len(rows)} videos:")
            for v in rows:
                print(f"  v{v['id']:<3} {v['camera_id']:<10} {v['path']}")
            return 0

        if args.what == "identities":
            idents = store.list_identities()
            rows = [dataclasses.asdict(i) for i in idents]
            if args.json:
                print(json.dumps(rows))
                return 0
            print(f"{len(rows)} identities:")
            for i in rows:
                label = i["label"] or "(unnamed)"
                print(f"  #{i['id']:<3} {i['type']:<11} {label:<24} by {i['created_by']}")
            return 0

        # sightings (default) — filter the metadata rows in-CLI by camera / quality.
        rows = store.list_sightings()
        if args.camera:
            rows = [s for s in rows if s["camera_id"] in args.camera]
        if args.min_quality > 0.0:
            rows = [s for s in rows if (s["quality"] or 0.0) >= args.min_quality]
        if args.json:
            print(json.dumps(rows))
            return 0
        print(f"{len(rows)} sightings:")
        for s in rows:
            ident = s["identity_id"] if s["identity_id"] is not None else "-"
            print(
                f"  #{s['id']:<3} {s['camera_id']:<10} t={s['ts']:7.2f}s  q={s['quality']:.3f}  "
                f"id={ident!s:<4} {s['chip_path']}"
            )
        return 0
    finally:
        store.close()


def _search(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        if args.sighting is not None:
            hits = search_by_sighting(
                args.sighting,
                store=store,
                top_k=args.top_k,
                cameras=args.cameras,
                since=args.since,
                min_quality=args.min_quality,
                actor=args.actor,
            )
            query_ref = f"sighting:{args.sighting}"
        else:
            hits = search_by_image(
                args.image,
                store=store,
                top_k=args.top_k,
                cameras=args.cameras,
                since=args.since,
                min_quality=args.min_quality,
                device=args.device,
                actor=args.actor,
            )
            query_ref = str(args.image)

        if args.json:
            print(
                json.dumps(
                    {"query": query_ref, "n_hits": len(hits), "hits": [_hit_dict(h) for h in hits]}
                )
            )
            return 0
        print(f"{len(hits)} hits for {query_ref}:")
        for h in hits:
            s = h.sighting
            print(
                f"  score={h.score:.3f}  #{s.id:<3} {s.camera_id:<10} t={s.ts:7.2f}s  "
                f"q={s.quality:.3f}  {s.chip_path}"
            )
        return 0
    finally:
        store.close()


def _enroll(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        identity_id = enroll(
            args.label,
            args.images,
            store=store,
            source=args.source,
            device=args.device,
            actor=args.actor,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "identity_id": identity_id,
                        "label": args.label,
                        "n_images": len(args.images),
                        "db": str(args.db),
                    }
                )
            )
            return 0
        print(f"enrolled {args.label!r} as identity #{identity_id} ({len(args.images)} image(s))")
        return 0
    finally:
        store.close()


def _cluster(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        result = run_clustering(
            store=store,
            space_id=args.space_id,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            only_unassigned=not args.include_assigned,
            actor=args.actor,
        )
        if args.json:
            print(json.dumps(dataclasses.asdict(result)))
            return 0
        print(result.summary())
        return 0
    finally:
        store.close()


def _audit(args: argparse.Namespace) -> int:
    store = _open_store(args)
    try:
        rows = audit_log(store=store, actor=args.actor, since=args.since)
        if args.json:
            print(json.dumps(rows))
            return 0
        print(f"{len(rows)} audit rows:")
        for r in rows:
            print(f"  {r['ts']}  {r['actor']:<10} {r['action']:<18} {r.get('details') or ''}")
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="argus", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_peek = sub.add_parser("peek", help="fast-triage a folder of clips")
    p_peek.add_argument("folder", type=Path, help="folder of video files")
    p_peek.add_argument("--glob", default="*.mp4", help="which files to scan")
    p_peek.add_argument("--workers", type=int, default=8, help="parallel decode workers")
    _add_common(p_peek)
    p_peek.set_defaults(func=_peek)

    p_track = sub.add_parser("track", help="detect + track one clip")
    p_track.add_argument("video", type=Path, help="path to a video clip")
    p_track.add_argument("--max-frames", type=int, default=None, help="cap frames processed")
    p_track.add_argument(
        "--render",
        nargs="?",
        const="",
        default=None,
        help="also write an annotated H.264 clip; PATH optional (default: out/<name>_tracked.mp4)",
    )
    _add_common(p_track)
    p_track.set_defaults(func=_track)

    p_audio = sub.add_parser("audio", help="classify the audio track of one clip")
    p_audio.add_argument("clip", type=Path, help="path to an audio or video file")
    p_audio.add_argument(
        "--model",
        default=DEFAULT_AUDIO_MODEL,
        help="HuggingFace audio model (default: zero-shot CLAP)",
    )
    p_audio.add_argument(
        "--overlap",
        type=float,
        default=1.0,
        help="seconds of overlap between adjacent 5s segments",
    )
    p_audio.add_argument("--segment", type=float, default=5.0, help="segment length in seconds")
    p_audio.add_argument("--top-k", type=int, default=2, help="predictions kept per segment")
    p_audio.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="candidate labels for the zero-shot CLAP model "
        "(default: a surveillance-oriented set)",
    )
    p_audio.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    p_audio.add_argument("--json", action="store_true", help="emit one JSON object to stdout")
    p_audio.set_defaults(func=_audio)

    p_ingest = sub.add_parser(
        "ingest", help="detect→track→embed best face per track; persist sightings"
    )
    p_ingest.add_argument("video", type=Path, help="path to a video clip")
    p_ingest.add_argument("--camera", default=None, help="camera id (default: clip stem)")
    p_ingest.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    p_ingest.add_argument("--conf", type=float, default=0.25, help="person-detector confidence")
    p_ingest.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p_ingest.add_argument(
        "--face-stride",
        type=int,
        default=1,
        help="run the face detector every Nth processed frame",
    )
    p_ingest.add_argument("--max-frames", type=int, default=None, help="cap frames processed")
    # Quality-gate thresholds (defaults are the calibrated QualityGate values). Lower
    # --min-face-px / --min-blur-var to admit smaller/softer faces on distant-camera footage.
    p_ingest.add_argument(
        "--min-face-px", type=float, default=_GATE.min_face_px, help="min face size (px)"
    )
    p_ingest.add_argument(
        "--min-blur-var",
        type=float,
        default=_GATE.min_blur_var,
        help="min Laplacian blur variance",
    )
    p_ingest.add_argument(
        "--max-yaw-ratio", type=float, default=_GATE.max_yaw_ratio, help="max pose (yaw) ratio"
    )
    p_ingest.add_argument(
        "--min-det-score", type=float, default=_GATE.min_det_score, help="min face detector score"
    )
    _add_store(p_ingest)
    p_ingest.set_defaults(func=_ingest)

    p_ls = sub.add_parser("ls", help="list stored videos / sightings / identities")
    p_ls.add_argument(
        "what",
        nargs="?",
        default="sightings",
        choices=["videos", "sightings", "identities"],
        help="what to list (default: sightings)",
    )
    p_ls.add_argument("--camera", nargs="+", default=None, help="filter sightings by camera id")
    p_ls.add_argument(
        "--min-quality", type=float, default=0.0, help="filter sightings by minimum face quality"
    )
    _add_store(p_ls)
    p_ls.set_defaults(func=_ls)

    p_search = sub.add_parser("search", help="face search over ingested footage (re-ID)")
    src = p_search.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="probe photo: find this face in the store")
    src.add_argument("--sighting", type=int, help="'more like this' from an existing sighting id")
    p_search.add_argument("--top-k", type=int, default=20, help="max ranked hits to return")
    p_search.add_argument(
        "--cameras", nargs="+", default=None, help="restrict to these camera ids"
    )
    p_search.add_argument(
        "--since", type=float, default=None, help="restrict to ts >= S (seconds)"
    )
    p_search.add_argument("--min-quality", type=float, default=0.0, help="minimum face quality")
    p_search.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    p_search.add_argument("--actor", default="cli", help="actor recorded in the audit log")
    _add_store(p_search)
    p_search.set_defaults(func=_search)

    p_enroll = sub.add_parser("enroll", help="enroll a known person into the watchlist gallery")
    p_enroll.add_argument("label", help="person's name/label")
    p_enroll.add_argument("images", nargs="+", type=Path, help="one or more face photos")
    p_enroll.add_argument("--source", default="id_photo", help="provenance tag for the enrollment")
    p_enroll.add_argument("--device", default=None, help="'cuda', 'cpu', ... (default: auto)")
    p_enroll.add_argument("--actor", default="cli", help="actor recorded in the audit log")
    _add_store(p_enroll)
    p_enroll.set_defaults(func=_enroll)

    p_cluster = sub.add_parser(
        "cluster", help="group unlabeled sightings into provisional identities"
    )
    p_cluster.add_argument(
        "--space-id",
        default=_DEFAULT_SPACE_ID,
        help=f"embedding space to cluster (default: {_DEFAULT_SPACE_ID})",
    )
    p_cluster.add_argument(
        "--min-cluster-size", type=int, default=5, help="HDBSCAN min cluster size"
    )
    p_cluster.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min samples")
    p_cluster.add_argument(
        "--include-assigned", action="store_true", help="also cluster already-identified sightings"
    )
    p_cluster.add_argument("--actor", default="cli", help="actor recorded in the audit log")
    _add_store(p_cluster)
    p_cluster.set_defaults(func=_cluster)

    p_audit = sub.add_parser("audit", help="show the compliance audit trail")
    p_audit.add_argument("--actor", default=None, help="filter by actor")
    p_audit.add_argument("--since", default=None, help="filter by ISO timestamp (ts >= SINCE)")
    _add_store(p_audit)
    p_audit.set_defaults(func=_audit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
