"""Watchlist + identity administration and compliance ops.

Enrollment builds the known-identity gallery; label_cluster/merge/reassign let an operator
curate provisional clusters into named identities; audit_log/purge/export_case are the
compliance surface. All ops take an ``actor`` for the audit trail.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..core import Enrollment, Identity
from ._embed import image_to_embedding


def enroll(
    label: str,
    images,
    *,
    store,
    source: str = "id_photo",
    face_detector=None,
    embedder=None,
    device: str | None = None,
    actor: str = "unknown",
) -> int:
    """Enroll a known person from one or more face ``images`` (paths or BGR ndarrays).

    Creates a ``known`` identity, embeds each image's best face, and stores an enrollment (with
    its aligned chip) in the watchlist gallery. Returns the new identity id.
    """
    identity_id = store.add_identity(Identity(type="known", label=label, created_by=actor))
    for i, image in enumerate(images):
        vec, space_id, chip = image_to_embedding(
            image, face_detector=face_detector, embedder=embedder, device=device
        )
        chip_path = store.chips_dir / f"enroll_i{identity_id}_{i}.png"
        cv2.imwrite(str(chip_path), chip)
        store.add_enrollment(
            Enrollment(identity_id=identity_id, chip_path=str(chip_path),
                       embedding_space_id=space_id, source=source),
            vec,
        )
    store.audit(actor=actor, action="enroll", target_type="identity", target_id=identity_id,
                details=f"label={label!r} n={len(list(images)) if hasattr(images, '__len__') else '?'}")
    return identity_id


def reassign(sighting_id: int, identity_id: int | None, *, store, actor: str = "unknown") -> None:
    """Move one sighting to a different identity (or ``None`` to clear). Audited."""
    store.assign_identity(sighting_id, identity_id, actor=actor)


def merge(into: int, source_cluster: int, *, store, actor: str = "unknown") -> int:
    """Merge all sightings of a provisional cluster into known identity ``into``. Returns count."""
    return store.merge_cluster_into_identity(source_cluster, into, actor=actor)


def label_cluster(cluster_id: int, label: str, *, store, actor: str = "unknown") -> int:
    """Promote a provisional cluster to a named ``known`` identity. Returns the new identity id."""
    identity_id = store.add_identity(Identity(type="known", label=label, created_by=actor))
    store.merge_cluster_into_identity(cluster_id, identity_id, actor=actor)
    return identity_id


def audit_log(*, store, actor: str | None = None, since: str | None = None) -> list[dict]:
    """Return audit rows, optionally filtered by ``actor`` and/or ``since`` (ISO timestamp)."""
    return store.list_audit(actor=actor, since=since)


def purge(*, store, before: str, actor: str = "unknown") -> int:
    """Delete sightings (rows + vectors + chip files) created before ``before`` (ISO). Audited."""
    return store.purge(before=before, actor=actor)


def export_case(identity_id: int, dest, *, store, actor: str = "unknown") -> Path:
    """Export an identity's sightings (chips + manifest.json) to ``dest`` for handoff. Audited."""
    return store.export_case(identity_id, dest, actor=actor)
