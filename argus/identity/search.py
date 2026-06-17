"""Face search — find where/whether a person appears across ingested footage.

Both calls return ranked ``SearchHit``s (matched sighting + cosine score + evidence chip) for
**human adjudication** — never an automated identity decision. Every search writes an audit row.
"""

from __future__ import annotations

from pathlib import Path

from ._embed import image_to_embedding


def search_by_image(
    probe,
    *,
    store,
    top_k: int = 20,
    cameras: list[str] | None = None,
    since: float | None = None,
    min_quality: float = 0.0,
    face_detector=None,
    embedder=None,
    device: str | None = None,
    actor: str = "unknown",
):
    """Find sightings whose face most resembles the face in ``probe`` (path or BGR ndarray).

    Filters by ``cameras`` / ``since`` (video timestamp) / ``min_quality``. Returns up to
    ``top_k`` ranked ``SearchHit``s.
    """
    vec, space_id, _chip = image_to_embedding(
        probe, face_detector=face_detector, embedder=embedder, device=device
    )
    hits = store.search_sightings(
        vec, space_id, top_k=top_k, cameras=cameras, since=since, min_quality=min_quality
    )
    query_ref = str(probe) if isinstance(probe, (str, Path)) else None
    store.audit(
        actor=actor,
        action="search_by_image",
        query_ref=query_ref,
        details=f"top_k={top_k} hits={len(hits)}",
    )
    return hits


def search_by_sighting(
    sighting_id: int,
    *,
    store,
    top_k: int = 20,
    cameras: list[str] | None = None,
    since: float | None = None,
    min_quality: float = 0.0,
    actor: str = "unknown",
):
    """Find more sightings of the person in an existing sighting ("more like this").

    Uses the stored embedding of ``sighting_id`` and excludes that sighting from the results.
    """
    src = store.get_sighting(sighting_id)
    vec = store.get_embedding(sighting_id)
    if src is None or vec is None:
        raise ValueError(f"no sighting {sighting_id} (or it has no embedding)")
    hits = store.search_sightings(
        vec,
        src.embedding_space_id,
        top_k=top_k + 1,
        cameras=cameras,
        since=since,
        min_quality=min_quality,
    )
    hits = [h for h in hits if h.sighting.id != sighting_id][:top_k]
    store.audit(
        actor=actor,
        action="search_by_sighting",
        target_type="sighting",
        target_id=sighting_id,
        details=f"top_k={top_k} hits={len(hits)}",
    )
    return hits
