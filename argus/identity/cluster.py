"""Open-set re-ID: cluster unlabeled sightings into provisional identities (HDBSCAN).

Groups the embeddings of not-yet-identified sightings; each dense cluster becomes a
``provisional`` ``Identity`` and its members are tagged with that identity's id as ``cluster_id``.
An operator then names a cluster (``label_cluster``) or merges it into a known identity
(``merge``). Embeddings are L2-normalized, so euclidean-distance clustering == cosine clustering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..core import Identity


@dataclass(frozen=True)
class ClusterResult:
    """Summary of a ``run_clustering`` pass."""

    n_sightings: int
    n_clusters: int
    n_noise: int
    run_id: int | None
    identity_ids: list[int]

    def summary(self) -> str:
        return (
            f"clustered {self.n_sightings} sightings -> {self.n_clusters} provisional "
            f"identities ({self.n_noise} noise), run {self.run_id}"
        )


def run_clustering(
    *,
    store,
    space_id: str,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    only_unassigned: bool = True,
    actor: str = "unknown",
) -> ClusterResult:
    """Cluster sightings in ``space_id`` into provisional identities. Returns a ``ClusterResult``.

    ``only_unassigned`` restricts to sightings not yet linked to an identity. Needs the
    ``argus[cluster]`` extra (scikit-learn). Deterministic for a fixed input (sorted by id).
    """
    sightings = sorted(
        store.iter_sightings(space_id=space_id, unassigned_only=only_unassigned),
        key=lambda s: s.id,
    )
    if len(sightings) < min_cluster_size:
        return ClusterResult(len(sightings), 0, len(sightings), None, [])

    try:
        import numpy as np
        from sklearn.cluster import HDBSCAN
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "run_clustering needs scikit-learn; install the extra: pip install argus[cluster]"
        ) from e

    X = np.stack([np.asarray(s.embedding, dtype=np.float32) for s in sightings])
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean", copy=True
    ).fit_predict(X)

    params = json.dumps(
        {"min_cluster_size": min_cluster_size, "min_samples": min_samples, "metric": "euclidean"}
    )
    run_id = store.add_cluster_run("hdbscan", params, space_id)
    n_noise = int((labels == -1).sum())
    identity_ids: list[int] = []
    for lbl in sorted(set(int(x) for x in labels) - {-1}):
        member_ids = [sightings[i].id for i in range(len(sightings)) if int(labels[i]) == lbl]
        pid = store.add_identity(
            Identity(type="provisional", created_by=actor, notes=f"run {run_id} cluster {lbl}")
        )
        store.assign_cluster(member_ids, pid)
        identity_ids.append(pid)

    store.audit(
        actor=actor,
        action="run_clustering",
        details=f"run={run_id} clusters={len(identity_ids)} noise={n_noise} n={len(sightings)}",
    )
    return ClusterResult(len(sightings), len(identity_ids), n_noise, run_id, identity_ids)
