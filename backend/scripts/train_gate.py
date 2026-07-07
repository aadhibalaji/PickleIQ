"""Train the small Pass 1 active/dead gate classifier from accumulated data.

The Pass 1 gate is currently a thresholded linear rule over low-dimensional
features (motion energy, ball recency). Those exact features are logged per
sampled frame in the pass1 overlays, and human rally labels provide the
active/dead ground truth — so this script fits a logistic regression over
them (pure numpy, no extra deps) and reports how it compares to the current
thresholds. Once trained it writes learned thresholds back into a new
LearnedParams version, so ALL future clips inherit it (global learning).

Usage:
    .venv\\Scripts\\python backend/scripts/train_gate.py [--apply]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.cv import overlays  # noqa: E402
from app.learning import params as learned_params  # noqa: E402


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """X = [motion_energy, frames_since_ball_norm], y = inside human rally."""
    X, y = [], []
    with db.connect() as con:
        videos = con.execute("SELECT id, fps, frame_count FROM videos").fetchall()
    for v in videos:
        with db.connect() as con:
            human = con.execute(
                "SELECT start_frame, end_frame FROM rally_labels"
                " WHERE video_id=? AND source='human'", (v["id"],)).fetchall()
        if not human:
            continue
        spans = [(r["start_frame"], r["end_frame"]) for r in human]
        ents = overlays.read_range(v["id"], "pass1", 0, v["frame_count"] or 10**9)
        fps = v["fps"] or 30.0
        last_ball = -10**9
        for e in ents:
            if "ball" in e:
                last_ball = e["f"]
            since_ball = min((e["f"] - last_ball) / fps, 5.0)
            X.append([e.get("me", 0.0), since_ball])
            y.append(1.0 if any(s <= e["f"] <= t for s, t in spans) else 0.0)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 500, lr: float = 0.5):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - y) / len(y)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the learned motion threshold into a new global "
                         "LearnedParams version")
    args = ap.parse_args()

    X, y = build_dataset()
    if len(X) < 500:
        print(f"only {len(X)} samples — label and process more clips first.")
        return
    w = fit_logistic(X, y)
    p = 1.0 / (1.0 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))
    acc = float(((p > 0.5) == y).mean())
    print(f"samples: {len(X)}   logistic gate accuracy: {acc:.3f}")
    print(f"weights: motion={w[0]:.3f}  since_ball={w[1]:.3f}  bias={w[2]:.3f}")

    _, cur = learned_params.get_latest()
    ambient = float(np.median(X[y == 0, 0])) if (y == 0).any() else 0.1
    # Decision boundary at since_ball=0 -> equivalent absolute motion
    # threshold; the gate itself is relative, so convert via the ambient
    # (dead-time) baseline into a p1_motion_ratio.
    if abs(w[0]) > 1e-9 and ambient > 1e-6:
        thresh = float(-w[2] / w[0])
        ratio = thresh / ambient
        print(f"implied threshold {thresh:.2f} over ambient {ambient:.2f} "
              f"-> p1_motion_ratio {ratio:.2f} (current {cur['p1_motion_ratio']})")
        if args.apply:
            v = learned_params.apply_updates(
                {"p1_motion_ratio": ratio}, origin="train_gate")
            print(f"applied as LearnedParams v{v} — all future uploads inherit it")
    else:
        print("degenerate fit; not applying")


if __name__ == "__main__":
    main()
