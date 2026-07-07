"""Overlay metadata persistence. All annotation the frontend draws is stored
as JSON chunks keyed by (video, source, chunk_index) — the original video is
never edited or copied; visuals are regenerated from this metadata on
demand. Chunks cover CHUNK_FRAMES frames so the player only fetches what is
near the playhead."""
from __future__ import annotations

from collections import defaultdict

from .. import db

CHUNK_FRAMES = 900  # 30 s at 30 fps


def chunk_index(frame: int) -> int:
    return frame // CHUNK_FRAMES


def clear(video_id: str, source: str) -> None:
    with db.connect() as con:
        con.execute("DELETE FROM overlays WHERE video_id=? AND source=?", (video_id, source))


def write_entries(video_id: str, source: str, entries: list[dict]) -> None:
    """Entries must each carry an 'f' frame field. Merges with any chunk
    already stored (used for incremental streaming during Pass 1)."""
    by_chunk: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_chunk[chunk_index(e["f"])].append(e)
    with db.connect() as con:
        for ci, ents in by_chunk.items():
            row = con.execute(
                "SELECT payload FROM overlays WHERE video_id=? AND source=? AND chunk_index=?",
                (video_id, source, ci),
            ).fetchone()
            existing = db.loads(row["payload"], []) if row else []
            merged = {e["f"]: e for e in existing}
            merged.update({e["f"]: e for e in ents})
            payload = db.dumps([merged[f] for f in sorted(merged)])
            con.execute(
                "INSERT OR REPLACE INTO overlays (video_id, source, chunk_index, payload)"
                " VALUES (?,?,?,?)",
                (video_id, source, ci, payload),
            )


def read_range(video_id: str, source: str, start_frame: int, end_frame: int) -> list[dict]:
    c0, c1 = chunk_index(start_frame), chunk_index(end_frame)
    with db.connect() as con:
        rows = con.execute(
            "SELECT payload FROM overlays WHERE video_id=? AND source=?"
            " AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
            (video_id, source, c0, c1),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.extend(e for e in db.loads(r["payload"], [])
                   if start_frame <= e["f"] <= end_frame)
    return out
