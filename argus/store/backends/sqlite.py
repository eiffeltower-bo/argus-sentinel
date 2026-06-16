"""SQLite + sqlite-vec store backend (implements the core ``Store`` protocol).

A single self-contained file holds both the relational metadata and the face-embedding
vectors (via the ``sqlite-vec`` extension's ``vec0`` virtual table). No daemon, trivially
air-gapped — the design's "single-box" datastore. Aligned chips live next to the db under
``chips/``; the rows reference them by path.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ...core import Enrollment, Identity, SearchHit, Sighting, WatchlistHit

_EMPTY_VEC = np.empty((0,), dtype=np.float32)  # search hits carry metadata, not the vector


def _row_to_identity(row) -> Identity:
    return Identity(type=row["type"], label=row["label"], created_by=row["created_by"],
                    created_at=row["created_at"], notes=row["notes"], id=row["id"])


def _row_to_identity_dict(ident: Identity | None) -> dict | None:
    if ident is None:
        return None
    return {"id": ident.id, "type": ident.type, "label": ident.label,
            "created_by": ident.created_by, "created_at": ident.created_at, "notes": ident.notes}

# Relational schema. identities/enrollments/cluster_runs/audit_log back the search, clustering,
# and compliance phases; ingest writes only videos + sightings (+ the vector index).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id           INTEGER PRIMARY KEY,
    camera_id    TEXT NOT NULL,
    path         TEXT NOT NULL,
    fps          REAL,
    duration_s   REAL,
    width        INTEGER,
    height       INTEGER,
    ingested_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS tracks (
    id           INTEGER PRIMARY KEY,
    video_id     INTEGER NOT NULL REFERENCES videos(id),
    track_id     INTEGER NOT NULL,
    camera_id    TEXT
);
CREATE TABLE IF NOT EXISTS sightings (
    id                 INTEGER PRIMARY KEY,
    video_id           INTEGER NOT NULL REFERENCES videos(id),
    camera_id          TEXT NOT NULL,
    track_id           INTEGER NOT NULL,
    frame_idx          INTEGER NOT NULL,
    ts                 REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    quality            REAL,
    chip_path          TEXT,
    embedding_space_id TEXT NOT NULL,
    identity_id        INTEGER,
    cluster_id         INTEGER,
    created_at         TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sightings_camera_ts ON sightings(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_sightings_space ON sightings(embedding_space_id);
CREATE INDEX IF NOT EXISTS idx_sightings_identity ON sightings(identity_id);
CREATE INDEX IF NOT EXISTS idx_sightings_cluster ON sightings(cluster_id);
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY, type TEXT, label TEXT,
    created_by TEXT, created_at TEXT DEFAULT (datetime('now')), notes TEXT
);
CREATE TABLE IF NOT EXISTS enrollments (
    id INTEGER PRIMARY KEY, identity_id INTEGER REFERENCES identities(id),
    chip_path TEXT, embedding_space_id TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS cluster_runs (
    id INTEGER PRIMARY KEY, algo TEXT, params TEXT,
    embedding_space_id TEXT, created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY, actor TEXT, action TEXT,
    target_type TEXT, target_id INTEGER, query_ref TEXT,
    ts TEXT DEFAULT (datetime('now')), details TEXT
);
"""


class SqliteStore:
    """SQLite-backed ``Store``: relational metadata + a ``vec0`` vector index in one file.

    ``dim`` must match the embedder's output dimension (512 for ArcFace; tests use a small
    fake). Aligned chips are written by the pipeline into ``chips_dir`` (defaults to a
    ``chips/`` folder beside the database).
    """

    def __init__(self, db_path, *, chips_dir=None, dim: int = 512) -> None:
        import sqlite_vec  # lazy: optional extension only loaded when the store is used

        self.db_path = Path(db_path)
        self.dim = dim
        self.chips_dir = Path(chips_dir) if chips_dir is not None else self.db_path.parent / "chips"
        self.chips_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._serialize = sqlite_vec.serialize_float32

        self.conn.executescript(_SCHEMA)
        # Cosine metric (vectors are L2-normalized, so this only makes the reported distance a
        # true cosine — ranking is unchanged). embedding_space_id is a partition key so searches
        # never compare across embedding spaces; camera_id/ts/quality are metadata columns so
        # those filters pre-filter *during* the KNN scan (correct recall, not filter-after-k).
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sightings USING vec0("
            f"embedding float[{dim}] distance_metric=cosine, "
            f"embedding_space_id text partition key, camera_id text, ts float, quality float)"
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_enrollments USING vec0("
            f"embedding float[{dim}] distance_metric=cosine, embedding_space_id text partition key)"
        )
        self.conn.commit()

    def add_video(
        self,
        camera_id: str,
        path: str,
        *,
        fps: float,
        duration_s: float,
        width: int,
        height: int,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO videos (camera_id, path, fps, duration_s, width, height) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, path, fps, duration_s, width, height),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_sightings(self, rows: list[Sighting]) -> None:
        for s in rows:
            x1, y1, x2, y2 = s.bbox
            cur = self.conn.execute(
                "INSERT INTO sightings (video_id, camera_id, track_id, frame_idx, ts, "
                "x1, y1, x2, y2, quality, chip_path, embedding_space_id, identity_id, cluster_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (s.video_id, s.camera_id, s.track_id, s.frame_idx, s.ts,
                 x1, y1, x2, y2, s.quality, s.chip_path, s.embedding_space_id,
                 s.identity_id, s.cluster_id),
            )
            s.id = int(cur.lastrowid)
            self.conn.execute(
                "INSERT INTO vec_sightings (rowid, embedding, embedding_space_id, camera_id, "
                "ts, quality) VALUES (?, ?, ?, ?, ?, ?)",
                (s.id, self._serialize([float(v) for v in s.embedding]),
                 s.embedding_space_id, s.camera_id, s.ts, s.quality),
            )
        self.conn.commit()

    def list_sightings(self) -> list[dict]:
        """All sighting rows (metadata only), for inspection/verification."""
        cur = self.conn.execute("SELECT * FROM sightings ORDER BY id")
        return [dict(r) for r in cur.fetchall()]

    def count_vectors(self) -> int:
        """How many embedding vectors are indexed — should equal the sighting count."""
        return int(self.conn.execute("SELECT count(*) FROM vec_sightings").fetchone()[0])

    # ------------------------------------------------------------------ search

    def search_sightings(
        self,
        vec,
        space_id: str,
        *,
        top_k: int,
        cameras: list[str] | None = None,
        since: float | None = None,
        min_quality: float = 0.0,
    ) -> list[SearchHit]:
        """KNN over the sighting index, filtered in-scan by space/camera/time/quality."""
        where = ["embedding MATCH ?", "k = ?", "embedding_space_id = ?"]
        params: list = [self._serialize([float(v) for v in vec]), int(top_k), space_id]
        if cameras:
            where.append(f"camera_id IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        if since is not None:
            where.append("ts >= ?")
            params.append(float(since))
        if min_quality > 0.0:
            where.append("quality >= ?")
            params.append(float(min_quality))
        sql = f"SELECT rowid, distance FROM vec_sightings WHERE {' AND '.join(where)} ORDER BY distance"
        hits: list[SearchHit] = []
        for r in self.conn.execute(sql, params).fetchall():
            s = self.get_sighting(int(r["rowid"]), with_embedding=False)
            if s is not None:
                dist = float(r["distance"])
                hits.append(SearchHit(sighting=s, distance=dist, score=1.0 - dist))
        return hits

    def search_enrollments(self, vec, space_id: str, *, top_k: int) -> list[WatchlistHit]:
        """KNN over the watchlist gallery; one (best) hit per enrolled identity."""
        knn = self.conn.execute(
            "SELECT rowid, distance FROM vec_enrollments "
            "WHERE embedding MATCH ? AND k = ? AND embedding_space_id = ? ORDER BY distance",
            (self._serialize([float(v) for v in vec]), int(top_k), space_id),
        ).fetchall()
        hits: list[WatchlistHit] = []
        seen: set[int] = set()
        for r in knn:
            enr = self.conn.execute(
                "SELECT * FROM enrollments WHERE id = ?", (int(r["rowid"]),)
            ).fetchone()
            if enr is None or enr["identity_id"] in seen:
                continue
            seen.add(enr["identity_id"])
            ident = self.get_identity(enr["identity_id"])
            if ident is not None:
                dist = float(r["distance"])
                hits.append(
                    WatchlistHit(identity=ident, distance=dist, score=1.0 - dist,
                                 chip_path=enr["chip_path"])
                )
        return hits

    def get_embedding(self, sighting_id: int):
        row = self.conn.execute(
            "SELECT embedding FROM vec_sightings WHERE rowid = ?", (sighting_id,)
        ).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def get_sighting(self, sighting_id: int, *, with_embedding: bool = True) -> Sighting | None:
        row = self.conn.execute(
            "SELECT * FROM sightings WHERE id = ?", (sighting_id,)
        ).fetchone()
        if row is None:
            return None
        emb = self.get_embedding(sighting_id) if with_embedding else _EMPTY_VEC
        return self._row_to_sighting(row, emb if emb is not None else _EMPTY_VEC)

    def iter_sightings(self, *, space_id: str, unassigned_only: bool = False) -> Iterator[Sighting]:
        sql = "SELECT * FROM sightings WHERE embedding_space_id = ?"
        if unassigned_only:
            sql += " AND identity_id IS NULL"
        sql += " ORDER BY id"
        for row in self.conn.execute(sql, (space_id,)).fetchall():
            emb = self.get_embedding(row["id"])
            yield self._row_to_sighting(row, emb if emb is not None else _EMPTY_VEC)

    @staticmethod
    def _row_to_sighting(row, embedding) -> Sighting:
        return Sighting(
            video_id=row["video_id"], camera_id=row["camera_id"], track_id=row["track_id"],
            frame_idx=row["frame_idx"], ts=row["ts"],
            bbox=(row["x1"], row["y1"], row["x2"], row["y2"]),
            quality=row["quality"], chip_path=row["chip_path"],
            embedding_space_id=row["embedding_space_id"], embedding=embedding,
            identity_id=row["identity_id"], cluster_id=row["cluster_id"], id=row["id"],
        )

    # ------------------------------------------------------ identity / enrollment

    def add_identity(self, identity: Identity) -> int:
        cur = self.conn.execute(
            "INSERT INTO identities (type, label, created_by, notes) VALUES (?, ?, ?, ?)",
            (identity.type, identity.label, identity.created_by, identity.notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_identity(self, identity_id: int) -> Identity | None:
        row = self.conn.execute(
            "SELECT * FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        return None if row is None else _row_to_identity(row)

    def list_identities(self, *, type: str | None = None) -> list[Identity]:
        if type is None:
            rows = self.conn.execute("SELECT * FROM identities ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM identities WHERE type = ? ORDER BY id", (type,)
            ).fetchall()
        return [_row_to_identity(r) for r in rows]

    def add_enrollment(self, enrollment: Enrollment, vec) -> int:
        cur = self.conn.execute(
            "INSERT INTO enrollments (identity_id, chip_path, embedding_space_id, source) "
            "VALUES (?, ?, ?, ?)",
            (enrollment.identity_id, enrollment.chip_path, enrollment.embedding_space_id,
             enrollment.source),
        )
        eid = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO vec_enrollments (rowid, embedding, embedding_space_id) VALUES (?, ?, ?)",
            (eid, self._serialize([float(v) for v in vec]), enrollment.embedding_space_id),
        )
        self.conn.commit()
        return eid

    # ------------------------------------------------------ assignment / clustering

    def assign_identity(
        self, sighting_id: int, identity_id: int | None, *, actor: str = "unknown"
    ) -> None:
        self.conn.execute(
            "UPDATE sightings SET identity_id = ? WHERE id = ?", (identity_id, sighting_id)
        )
        self._audit(actor=actor, action="assign_identity", target_type="sighting",
                    target_id=sighting_id, details=f"identity_id={identity_id}")
        self.conn.commit()

    def assign_cluster(self, sighting_ids: list[int], cluster_id: int) -> None:
        self.conn.executemany(
            "UPDATE sightings SET cluster_id = ? WHERE id = ?",
            [(cluster_id, sid) for sid in sighting_ids],
        )
        self.conn.commit()

    def merge_cluster_into_identity(
        self, cluster_id: int, identity_id: int, *, actor: str = "unknown"
    ) -> int:
        cur = self.conn.execute(
            "UPDATE sightings SET identity_id = ?, cluster_id = NULL WHERE cluster_id = ?",
            (identity_id, cluster_id),
        )
        n = int(cur.rowcount)
        self._audit(actor=actor, action="merge", target_type="identity", target_id=identity_id,
                    details=f"cluster_id={cluster_id} n={n}")
        self.conn.commit()
        return n

    def add_cluster_run(self, algo: str, params: str, space_id: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO cluster_runs (algo, params, embedding_space_id) VALUES (?, ?, ?)",
            (algo, params, space_id),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------- compliance

    def _audit(self, *, actor, action, target_type=None, target_id=None,
               query_ref=None, details=None) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (actor, action, target_type, target_id, query_ref, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (actor, action, target_type, target_id, query_ref, details),
        )

    def audit(self, *, actor, action, target_type=None, target_id=None,
              query_ref=None, details=None) -> None:
        self._audit(actor=actor, action=action, target_type=target_type,
                    target_id=target_id, query_ref=query_ref, details=details)
        self.conn.commit()

    def list_audit(self, *, actor: str | None = None, since: str | None = None) -> list[dict]:
        where, params = [], []
        if actor is not None:
            where.append("actor = ?")
            params.append(actor)
        if since is not None:
            where.append("ts >= ?")
            params.append(since)
        sql = "SELECT * FROM audit_log"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def purge(self, *, before: str, actor: str = "unknown") -> int:
        """Delete sightings (rows + vectors + chip files) created before ``before`` (ISO ts)."""
        rows = self.conn.execute(
            "SELECT id, chip_path FROM sightings WHERE created_at < ?", (before,)
        ).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            ph = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM vec_sightings WHERE rowid IN ({ph})", ids)
            self.conn.execute(f"DELETE FROM sightings WHERE id IN ({ph})", ids)
        self._audit(actor=actor, action="purge", details=f"before={before} n={len(ids)}")
        self.conn.commit()
        for r in rows:  # unlink chips last: a crash leaves orphan files, not orphan rows
            if r["chip_path"]:
                Path(r["chip_path"]).unlink(missing_ok=True)
        return len(ids)

    def export_case(self, identity_id: int, dest, *, actor: str = "unknown") -> Path:
        """Gather an identity's sightings (chips + a manifest.json) into ``dest`` for handoff."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        ident = self.get_identity(identity_id)
        rows = self.conn.execute(
            "SELECT * FROM sightings WHERE identity_id = ? ORDER BY id", (identity_id,)
        ).fetchall()
        for r in rows:
            cp = r["chip_path"]
            if cp and Path(cp).exists():
                shutil.copy2(cp, dest / Path(cp).name)
        manifest = {
            "identity": _row_to_identity_dict(ident),
            "sightings": [
                {"id": r["id"], "camera_id": r["camera_id"], "ts": r["ts"],
                 "quality": r["quality"], "video_id": r["video_id"],
                 "chip": Path(r["chip_path"]).name if r["chip_path"] else None}
                for r in rows
            ],
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._audit(actor=actor, action="export", target_type="identity",
                    target_id=identity_id, details=f"n={len(rows)}")
        self.conn.commit()
        return dest

    def close(self) -> None:
        self.conn.close()
