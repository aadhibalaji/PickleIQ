"""Export / import ground-truth ball marks (ball_corrections) to a git-tracked
JSON file so a partner can read, review, and re-import them.

`ball_corrections` doubles as the ground-truth mark store for ball events
(the viewer's "Mark ground truth" mode; see app/api/annotate.py). Those marks
are the PRIMARY signal the ball evaluation harness scores against
(scripts/eval_ball_filter.py) and the training data source for a fine-tuned
TrackNet — but they live only in each person's local SQLite DB. This tool
serialises them to reference_data/ball_gt.json (checked into git) and reads
them back.

Marks are keyed by video FILENAME, not the clip-local DB id, so the file is
portable across databases (several DB rows often point at the same clip file;
eval_ball_filter.py dedupes the same way). On import the filename is resolved
to a local video row; the human's call (pixel position, event type, in/out,
court projection) is preserved exactly as exported.

Usage (from the repo root):
    python -m backend.scripts.ball_gt_io export
    python -m backend.scripts.ball_gt_io export --video-id <id> --out path.json
    python -m backend.scripts.ball_gt_io import
    python -m backend.scripts.ball_gt_io import --replace   # clear existing first

Export is idempotent (stable ordering, pretty-printed for clean git diffs).
Import is idempotent too: a mark already present (same frame/pixel/type) is
skipped unless --replace wipes the clip's existing marks first.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import REPO_ROOT  # noqa: E402

FORMAT_VERSION = 1
DEFAULT_PATH = REPO_ROOT / "reference_data" / "ball_gt.json"

# Semantic mark fields worth sharing. DB-local columns (id, created_at, the
# clip-local video_id) are intentionally dropped — the ground truth is the
# human's call, not the row bookkeeping.
_MARK_FIELDS = ("frame", "x", "y", "event_type", "in_out",
                "court_x", "court_y", "in_out_suggested", "disagree", "scope")


def _mark_from_row(row) -> dict:
    k = row.keys()
    m = {}
    for f in _MARK_FIELDS:
        m[f] = row[f] if f in k else None
    if not m.get("event_type"):        # legacy rows predate event_type
        m["event_type"] = "position"
    if m.get("scope") is None:
        m["scope"] = "clip_local"
    m["disagree"] = int(m.get("disagree") or 0)
    return m


def _mark_key(m: dict) -> tuple:
    """Identity of a mark for dedupe — same frame, pixel (rounded to whole px),
    and event type is the same ground-truth mark."""
    return (int(m["frame"]), round(float(m["x"])), round(float(m["y"])),
            m.get("event_type") or "position")


def export(out_path: Path, video_id: str | None) -> None:
    with db.connect() as con:
        if video_id:
            vrows = con.execute(
                "SELECT id, filename FROM videos WHERE id=?", (video_id,)).fetchall()
        else:
            vrows = con.execute("SELECT id, filename FROM videos").fetchall()
        by_name: dict[str, list[dict]] = {}
        for v in vrows:
            rows = con.execute(
                "SELECT * FROM ball_corrections WHERE video_id=? ORDER BY frame",
                (v["id"],)).fetchall()
            if not rows:
                continue
            marks = by_name.setdefault(v["filename"], [])
            seen = {_mark_key(m) for m in marks}
            for r in rows:
                m = _mark_from_row(r)
                key = _mark_key(m)
                if key not in seen:      # dedupe across duplicate DB rows / clip
                    seen.add(key)
                    marks.append(m)

    for name in by_name:                 # stable ordering for clean git diffs
        by_name[name].sort(key=lambda m: (m["frame"], m["x"], m["y"]))
    payload = {
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos": {name: {"marks": by_name[name]} for name in sorted(by_name)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    n_marks = sum(len(v["marks"]) for v in payload["videos"].values())
    print(f"exported {n_marks} ground-truth mark(s) across "
          f"{len(payload['videos'])} clip(s) -> {out_path}")
    if not payload["videos"]:
        print("  (no ball_corrections found — mark some ground truth in the "
              "labeling tab first)")


def _resolve_video(con, filename: str) -> str | None:
    """Pick the local DB row for a filename. Several rows can point at the same
    clip; prefer the one that already carries the most marks, then the most
    recently uploaded — the same 'best row' preference eval_ball_filter uses."""
    rows = con.execute(
        "SELECT v.id,"
        " (SELECT COUNT(*) FROM ball_corrections b WHERE b.video_id=v.id) n,"
        " v.uploaded_at ua"
        " FROM videos v WHERE v.filename=?"
        " ORDER BY n DESC, ua DESC", (filename,)).fetchall()
    return rows[0]["id"] if rows else None


def import_(in_path: Path, replace: bool) -> None:
    if not in_path.exists():
        sys.exit(f"no ground-truth file at {in_path} — run 'export' first "
                 "(or point --in at the shared file)")
    payload = json.loads(in_path.read_text())
    if payload.get("format_version") != FORMAT_VERSION:
        print(f"warning: format_version {payload.get('format_version')} != "
              f"{FORMAT_VERSION}; attempting import anyway")
    videos = payload.get("videos", {})

    n_ins = n_skip = n_clips = 0
    unmatched: list[str] = []
    with db.connect() as con:
        for filename, entry in videos.items():
            vid = _resolve_video(con, filename)
            if vid is None:
                unmatched.append(filename)
                continue
            n_clips += 1
            if replace:
                con.execute("DELETE FROM ball_corrections WHERE video_id=?", (vid,))
                existing: set[tuple] = set()
            else:
                existing = {
                    _mark_key(_mark_from_row(r))
                    for r in con.execute(
                        "SELECT * FROM ball_corrections WHERE video_id=?", (vid,))}
            for m in entry.get("marks", []):
                key = _mark_key(m)
                if key in existing:
                    n_skip += 1
                    continue
                existing.add(key)
                con.execute(
                    "INSERT INTO ball_corrections (id, video_id, frame, x, y,"
                    " scope, created_at, event_type, court_x, court_y, in_out,"
                    " in_out_suggested, disagree) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (db.new_id(), vid, int(m["frame"]), float(m["x"]),
                     float(m["y"]), m.get("scope") or "clip_local", db.now(),
                     m.get("event_type") or "position", m.get("court_x"),
                     m.get("court_y"), m.get("in_out"), m.get("in_out_suggested"),
                     int(m.get("disagree") or 0)))
                n_ins += 1

    print(f"imported {n_ins} mark(s) into {n_clips} clip(s); skipped {n_skip} "
          f"already-present mark(s)")
    if unmatched:
        print(f"  {len(unmatched)} clip(s) in the file have no matching video "
              f"in this DB (upload them first): {', '.join(sorted(unmatched))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="write ball_corrections -> JSON")
    pe.add_argument("--video-id", default=None,
                    help="export a single clip (default: all clips)")
    pe.add_argument("--out", type=Path, default=DEFAULT_PATH,
                    help=f"output path (default: {DEFAULT_PATH})")

    pi = sub.add_parser("import", help="read JSON -> ball_corrections")
    pi.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_PATH,
                    help=f"input path (default: {DEFAULT_PATH})")
    pi.add_argument("--replace", action="store_true",
                    help="clear each clip's existing marks before importing")

    args = ap.parse_args()
    if args.cmd == "export":
        export(args.out, args.video_id)
    else:
        import_(args.in_path, args.replace)


if __name__ == "__main__":
    main()
