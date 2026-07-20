"""Stage E (labels) — ground-truth loading + per-timestep label expansion.

Two GT sources, same output shape (list of (start_s, end_s) rally spans):

* the app DB: ``rally_labels source='human'`` (frames -> seconds via the
  clip's fps) — the labeling tab already produces these;
* external files: one CSV (``start,end`` seconds per row, header optional) or
  JSON (``[{"start": s, "end": s}, ...]`` or ``[[s, e], ...]``) per clip.

``expand_labels`` converts rally spans into the per-timestep 0/1 live vector
aligned with a feature timeline — what the classifier trains on.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io_loaders import load_video


def load_ground_truth(video_id: str) -> list[tuple[float, float]]:
    """Human rally spans from the DB, in seconds."""
    from .. import db
    fps = float(load_video(video_id)["fps"])
    with db.connect() as con:
        rows = con.execute(
            "SELECT start_frame, end_frame FROM rally_labels"
            " WHERE video_id=? AND source='human' ORDER BY start_frame",
            (video_id,)).fetchall()
    return [(r["start_frame"] / fps, r["end_frame"] / fps) for r in rows]


def load_label_file(path: str | Path) -> list[tuple[float, float]]:
    """Rally spans from a CSV or JSON label file (seconds)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        spans = []
        for item in data:
            if isinstance(item, dict):
                spans.append((float(item["start"]), float(item["end"])))
            else:
                spans.append((float(item[0]), float(item[1])))
        return sorted(spans)
    spans = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            try:
                spans.append((float(row[0]), float(row[1])))
            except ValueError:
                continue        # header row
    return sorted(spans)


def expand_labels(t: np.ndarray,
                  spans: list[tuple[float, float]]) -> np.ndarray:
    """Per-timestep 0/1 live labels for a feature timeline `t` (seconds)."""
    y = np.zeros(len(t), dtype=np.float64)
    for s, e in spans:
        y[(t >= s) & (t <= e)] = 1.0
    return y
