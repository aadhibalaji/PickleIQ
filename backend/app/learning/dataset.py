"""Training-dataset assembly from stored human labels + Pass 2 feature traces.

Turns each human rally into windowed training examples using the SAME
featurizer the model sees at inference (bounce.trace_to_features) — no
train/serve skew. Weak supervision (the plan's cold-start strategy):

  * a human-confirmed rally END frame is a confirmed terminal bounce (class 1);
  * mid-rally frames where the heuristic already saw a hit are pseudo-labelled
    hits (class 2) — this expands the tiny positive set well beyond the few end
    frames, which is essential at low data volume;
  * augmentation (temporal jitter + light feature noise) multiplies the windows.

Court-flip augmentation is intentionally NOT used: the ball x/y features are
per-camera PIXEL coordinates (fixed camera), so a horizontal flip is not a
valid view — jitter/noise are the honest augmentations here.
"""
from __future__ import annotations

import numpy as np

from .. import db
from ..cv import overlays
from ..cv.bounce import BOUNCE_MODEL_FEATURES, trace_to_features

WINDOW = 31
T = 2 * WINDOW + 1  # fixed padded window length
IGNORE = -100       # CrossEntropyLoss ignore_index for padded frames


def _labeled_videos() -> list[dict]:
    with db.connect() as con:
        return db.rows_to_dicts(con.execute(
            "SELECT DISTINCT v.id AS id, v.fps AS fps FROM videos v"
            " JOIN rally_labels r ON r.video_id=v.id AND r.source='human'").fetchall())


def _add(X, Y, f, lab, augment, jitter, noise) -> None:
    X.append(f)
    Y.append(lab)
    if not augment:
        return
    for k in (jitter, -jitter):                       # temporal jitter
        X.append(np.roll(f, k, axis=0))
        Y.append(np.roll(lab, k))
    g = f.copy()                                      # light noise on pixel/diam channels
    g[:, :5] = g[:, :5] + np.random.normal(0, noise, g[:, :5].shape).astype(np.float32)
    X.append(g)
    Y.append(lab)


def build_end_dataset(augment: bool = True, jitter: int = 3, noise: float = 0.01,
                      pseudo: bool = True, video_ids: list[str] | None = None):
    """Return (windows, labels, meta). windows: list of (Ti, F) arrays; labels:
    list of (Ti,) int arrays with {0 none, 1 terminal bounce, 2 hit}.
    Pass video_ids to restrict to a subset of clips (for leave-one-clip-out)."""
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    vids: set[str] = set()
    n_pos = n_neg = 0
    for v in _labeled_videos():
        vid = v["id"]
        if video_ids is not None and vid not in video_ids:
            continue
        with db.connect() as con:
            human = con.execute("SELECT start_frame, end_frame FROM rally_labels"
                                " WHERE video_id=? AND source='human'", (vid,)).fetchall()
        for r in human:
            end = r["end_frame"]
            ents = overlays.read_range(vid, "pass2", end - WINDOW, end + WINDOW)
            if len(ents) >= WINDOW:
                f = trace_to_features(ents)
                lab = np.array([1 if abs(e["f"] - end) <= 2 else 0 for e in ents], dtype=np.int64)
                _add(X, Y, f, lab, augment, jitter, noise)
                n_pos += 1
                vids.add(vid)
            mid = (r["start_frame"] + r["end_frame"]) // 2
            ents = overlays.read_range(vid, "pass2", mid - WINDOW, mid + WINDOW)
            if len(ents) >= WINDOW:
                f = trace_to_features(ents)
                lab = np.array([2 if (pseudo and (e.get("evt") or {}).get("kind") == "hit") else 0
                                for e in ents], dtype=np.int64)
                _add(X, Y, f, lab, augment, jitter, noise)
                n_neg += 1
                vids.add(vid)
    meta = {"video_ids": sorted(vids), "n_clips": len(vids), "n_windows": len(X),
            "n_pos": n_pos, "n_neg": n_neg}
    return X, Y, meta


def to_padded(X: list[np.ndarray], Y: list[np.ndarray]):
    """Pad variable-length windows to a fixed T with IGNORE labels (masked in
    the loss) — replaces the old brittle truncate-to-min-and-stack."""
    Xp = np.zeros((len(X), T, BOUNCE_MODEL_FEATURES), dtype=np.float32)
    Yp = np.full((len(Y), T), IGNORE, dtype=np.int64)
    for i, (f, lab) in enumerate(zip(X, Y)):
        n = min(len(f), T)
        Xp[i, :n] = f[:n]
        Yp[i, :n] = lab[:n]
    return Xp, Yp
