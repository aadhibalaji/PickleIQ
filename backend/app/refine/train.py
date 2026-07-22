"""Stage E (training) — build per-timestep datasets and train the learned
live/dead classifier.

Splitting is ALWAYS by clip, never by timestep: adjacent timesteps are nearly
identical, so a timestep split leaks the eval set into training. Clips listed
in ``cfg.holdout_clips`` (video ids or filenames) are frozen out of both
training and sweep-tuning entirely.

Usage::

    python -m backend.app.refine.train                # all labeled clips
    python -m backend.app.refine.train --eval-frac 0.3
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from .classifiers import XGBoostClassifier
from .config import RefineConfig, load_config
from .features import compute_features
from .io_loaders import (clips_with_human_labels, load_pass1_windows,
                         load_tracking, load_video)
from .labels import expand_labels, load_ground_truth

log = logging.getLogger("refine.train")


def is_holdout(clip: dict, cfg: RefineConfig) -> bool:
    """Holdout entries may be video ids or filenames."""
    return clip["id"] in cfg.holdout_clips or clip["filename"] in cfg.holdout_clips


def build_clip_dataset(video_id: str, cfg: RefineConfig,
                       ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature matrix + expanded 0/1 live labels for every Pass 1 window of
    one clip (timesteps outside pass-1 windows are never included)."""
    video = load_video(video_id)
    fps = float(video["fps"])
    last = int(video["frame_count"]) - 1
    margin = int(round(cfg.window_margin_s * fps))
    gt = load_ground_truth(video_id)
    Xs, ys, names = [], [], None
    for w in load_pass1_windows(video_id):
        td = load_tracking(video_id, max(0, w.start_frame - margin),
                           min(last, w.end_frame + margin), fps=fps)
        if td.present.mean() < cfg.min_tracking_coverage:
            # stale overlays (Pass 1 re-run without Pass 2): the features are
            # empty, not evidence of dead play — don't teach the model that.
            log.warning("skipping window %s of %s: only %.0f%% tracked",
                        w.id[:8], video_id[:8], 100 * td.present.mean())
            continue
        feats = compute_features(td, cfg)
        Xs.append(feats.matrix())
        ys.append(expand_labels(feats.t, gt))
        names = feats.names
    if not Xs:
        return np.zeros((0, 0)), np.zeros(0), names or []
    return np.vstack(Xs), np.concatenate(ys), names


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC AUC (no sklearn)."""
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def split_by_clip(clip_ids: list[str], eval_frac: float,
                  seed: int = 0) -> tuple[set[str], set[str]]:
    """Disjoint (train, eval) clip-id sets. The unit of splitting is the CLIP
    — timesteps of one clip never appear on both sides."""
    if len(clip_ids) < 2:
        raise RuntimeError(f"need >= 2 labeled clips to split by clip, "
                           f"have {len(clip_ids)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clip_ids))
    n_eval = max(1, int(round(eval_frac * len(clip_ids))))
    eval_ids = {clip_ids[i] for i in order[:n_eval]}
    return set(clip_ids) - eval_ids, eval_ids


def train(cfg: RefineConfig, eval_frac: float = 0.25,
          seed: int = 0) -> XGBoostClassifier:
    """Train on all non-holdout labeled clips; report per-clip eval AUC."""
    clips = [c for c in clips_with_human_labels() if not is_holdout(c, cfg)]
    _, eval_ids = split_by_clip([c["id"] for c in clips], eval_frac, seed)

    data = {c["id"]: build_clip_dataset(c["id"], cfg) for c in clips}
    names = next(n for _, _, n in data.values() if n)
    Xtr = np.vstack([X for cid, (X, y, _) in data.items()
                     if cid not in eval_ids and len(X)])
    ytr = np.concatenate([y for cid, (X, y, _) in data.items()
                          if cid not in eval_ids and len(X)])
    log.info("training on %d clips (%d timesteps, %.1f%% live); eval on %d clips",
             len(clips) - len(eval_ids), len(ytr), 100 * ytr.mean(), len(eval_ids))

    model = XGBoostClassifier(cfg)
    model.fit(Xtr, ytr, names)
    for cid in sorted(eval_ids):
        X, y, _ = data[cid]
        if not len(X):
            continue
        p = model.predict_matrix(X)
        log.info("eval clip %s: AUC %.3f, acc@0.5 %.3f",
                 cid[:8], auc(y, p), float(((p > 0.5) == (y > 0.5)).mean()))
    from .refine import resolve_path
    model.save(resolve_path(cfg.model_path))
    log.info("saved model to %s", cfg.model_path)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the refine live classifier")
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    train(load_config(args.config), args.eval_frac, args.seed)


if __name__ == "__main__":
    main()
