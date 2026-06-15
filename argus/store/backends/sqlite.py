"""SQLite + sqlite-vec store backend (implements the core ``Store`` protocol).

A single self-contained file holds both the relational metadata and the face-embedding
vectors (via the ``sqlite-vec`` extension's ``vec0`` virtual table). No daemon, trivially
air-gapped — the design's "single-box" datastore. Aligned chips live next to the db under
``chips/``; the rows reference them by path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ...core import Sighting

# Relational schema. The later-phase tables (identities/enrollments/audit_log) are created
# now so migrations are a no-op when search/clustering land; ingest only writes the rest.
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
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sightings USING vec0(embedding float[{dim}])"
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
                "INSERT INTO vec_sightings (rowid, embedding) VALUES (?, ?)",
                (s.id, self._serialize([float(v) for v in s.embedding])),
            )
        self.conn.commit()

    def list_sightings(self) -> list[dict]:
        """All sighting rows (metadata only), for inspection/verification."""
        cur = self.conn.execute("SELECT * FROM sightings ORDER BY id")
        return [dict(r) for r in cur.fetchall()]

    def count_vectors(self) -> int:
        """How many embedding vectors are indexed — should equal the sighting count."""
        return int(self.conn.execute("SELECT count(*) FROM vec_sightings").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
