"""Export / import ground-truth ball marks so they can be shared via git.

The live marks live in the `ball_corrections` table inside data/app.db, which is
gitignored (large binary + WAL + videos). This dumps just those rows to a small
JSON file under reference_data/ that IS tracked, so a partner can read it as
reference or re-import it into their own DB.

Usage:
  python -m backend.scripts.ball_gt_io export   # DB  -> reference_data/ball_gt.json
  python -m backend.scripts.ball_gt_io import    # JSON -> DB (upsert by id)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.config import REPO_ROOT
from backend.app.db import connect

OUT = REPO_ROOT / "reference_data" / "ball_gt.json"


def export() -> None:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM ball_corrections ORDER BY video_id, frame"
        ).fetchall()
    data = [dict(r) for r in rows]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"exported {len(data)} ball marks -> {OUT}")


def import_() -> None:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    cols = sorted({k for row in data for k in row})
    placeholders = ",".join("?" for _ in cols)
    with connect() as con:
        for row in data:
            con.execute(
                f"INSERT OR REPLACE INTO ball_corrections ({','.join(cols)}) "
                f"VALUES ({placeholders})",
                [row.get(c) for c in cols],
            )
        con.commit()
    print(f"imported {len(data)} ball marks from {OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    {"export": export, "import": import_}[cmd]()
