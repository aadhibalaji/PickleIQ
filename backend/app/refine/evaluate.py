"""Stage E (evaluation) — segment-level eval harness + threshold sweep.

Metrics per clip and pooled:

* segment precision / recall / F1 — a predicted rally counts as correct when
  its IoU with an unmatched ground-truth rally exceeds ``cfg.eval_iou``;
* boundary error (seconds) on matched pairs — median and p90, reported
  separately for starts and ends;
* rejection accuracy — Pass 1 windows with no GT overlap should come out
  ``verdict='rejected'``; fraction that actually do.

The sweep grid-searches Stage C/D knobs on the NON-holdout labeled clips and
prints the best configs by F1, then median boundary error. Holdout clips
(``cfg.holdout_clips``) are excluded from the sweep entirely — never tune on
them.

Also includes the serve-gate validation utility: a directory of labeled
spans (JSON: {"video_id", "start", "end", "label": "serve"|"non-serve"})
is scored by the gate and reported as accuracy at ``serve_gate_min``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classifiers import LiveClassifier, make_classifier
from .config import RefineConfig, load_config
from .features import compute_features
from .io_loaders import (clips_with_human_labels, load_pass1_windows,
                         load_tracking, load_video)
from .labels import load_ground_truth
from .refine import refine_video
from .serve_gate import score_serve
from .train import is_holdout

log = logging.getLogger("refine.evaluate")


# ---------------------------------------------------------------------------
# segment matching + metrics
# ---------------------------------------------------------------------------

def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_segments(pred: list[tuple[float, float]],
                   gt: list[tuple[float, float]],
                   min_iou: float) -> list[tuple[int, int]]:
    """Greedy best-IoU matching; each GT rally is matched at most once."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred)
                    for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < min_iou:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        out.append((i, j))
    return out


@dataclass
class EvalResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    start_errs: list[float] = field(default_factory=list)  # pred - gt, seconds
    end_errs: list[float] = field(default_factory=list)
    reject_correct: int = 0     # non-rally windows correctly rejected
    reject_total: int = 0

    def add(self, other: "EvalResult") -> None:
        self.tp += other.tp; self.fp += other.fp; self.fn += other.fn
        self.start_errs += other.start_errs
        self.end_errs += other.end_errs
        self.reject_correct += other.reject_correct
        self.reject_total += other.reject_total

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def boundary_stats(self) -> dict:
        out = {}
        for name, errs in (("start", self.start_errs), ("end", self.end_errs)):
            a = np.abs(errs) if len(errs) else np.array([np.nan])
            out[f"{name}_median_s"] = float(np.median(a))
            out[f"{name}_p90_s"] = float(np.percentile(a, 90))
        return out

    def summary(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 3),
                "recall": round(self.recall, 3), "f1": round(self.f1, 3),
                **{k: round(v, 3) for k, v in self.boundary_stats().items()},
                "rejection_acc": round(self.reject_correct / self.reject_total, 3)
                                 if self.reject_total else None}


def evaluate_clip(video_id: str, cfg: RefineConfig,
                  classifier: LiveClassifier | None = None) -> EvalResult:
    """Refine one clip and score it against the human rally labels."""
    gt = load_ground_truth(video_id)
    records, _ = refine_video(video_id, cfg, classifier)
    pred = [(r["refined_start"], r["refined_end"]) for r in records
            if r["verdict"] == "rally"]
    matches = match_segments(pred, gt, cfg.eval_iou)
    res = EvalResult(tp=len(matches), fp=len(pred) - len(matches),
                     fn=len(gt) - len(matches))
    for i, j in matches:
        res.start_errs.append(pred[i][0] - gt[j][0])
        res.end_errs.append(pred[i][1] - gt[j][1])
    # rejection accuracy on windows with no GT overlap at all
    fps = float(load_video(video_id)["fps"])
    by_window: dict[str, list[dict]] = {}
    for r in records:
        by_window.setdefault(r["pass1_window_id"], []).append(r)
    for wid, recs in by_window.items():
        w = (recs[0]["pass1_start"], recs[0]["pass1_end"])
        if any(iou(w, g) > 0 or (g[0] >= w[0] and g[1] <= w[1]) for g in gt):
            continue
        res.reject_total += 1
        if all(r["verdict"] == "rejected" for r in recs):
            res.reject_correct += 1
    return res


def evaluate(cfg: RefineConfig, clip_ids: list[str] | None = None,
             classifier: LiveClassifier | None = None,
             include_holdout: bool = False, verbose: bool = True) -> EvalResult:
    """Pooled evaluation over labeled clips (holdout excluded by default)."""
    clips = clips_with_human_labels()
    if clip_ids:
        clips = [c for c in clips if c["id"] in clip_ids
                 or c["filename"] in clip_ids]
    if not include_holdout:
        clips = [c for c in clips if not is_holdout(c, cfg)]
    classifier = classifier or make_classifier(cfg)
    total = EvalResult()
    for c in clips:
        r = evaluate_clip(c["id"], cfg, classifier)
        if verbose:
            log.info("clip %s (%s): %s", c["id"][:8], c["filename"], r.summary())
        total.add(r)
    if verbose:
        log.info("POOLED: %s", total.summary())
    return total


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------

DEFAULT_GRID: dict[str, list] = {
    "enter_thresh": [0.6, 0.7, 0.8],
    "exit_thresh": [0.25, 0.35],
    "live_ratio": [0.7, 0.8],
    "merge_gap_s": [1.5, 2.0, 3.0],
    "min_rally_s": [1.5, 2.5],
    "serve_gate_min": [0.25, 0.30, 0.35, 0.40],
}


def sweep(cfg: RefineConfig, grid: dict[str, list] | None = None,
          top: int = 5) -> list[tuple[dict, dict]]:
    """Grid-search Stage C/D knobs on the non-holdout labeled clips. Returns
    the top combos as (overrides, summary), best first."""
    grid = grid or DEFAULT_GRID
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    log.info("sweeping %d combos over %s", len(combos), keys)
    classifier = make_classifier(cfg)
    results = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        r = evaluate(cfg.copy(**overrides), classifier=classifier,
                     verbose=False)
        results.append((overrides, r))
    results.sort(key=lambda x: (-x[1].f1,
                                x[1].boundary_stats()["start_median_s"]
                                + x[1].boundary_stats()["end_median_s"]))
    for ov, r in results[:top]:
        log.info("F1 %.3f  %s  %s", r.f1, r.summary(), ov)
    return [(ov, r.summary()) for ov, r in results[:top]]


# ---------------------------------------------------------------------------
# serve-gate validation on labeled spans
# ---------------------------------------------------------------------------

def validate_serve_gate(dir_path: str | Path, cfg: RefineConfig) -> dict:
    """Score labeled serve / non-serve spans and report gate accuracy.
    Each JSON file: {"video_id", "start", "end", "label": "serve"|"non-serve"}
    (times in seconds; lists of such objects also accepted)."""
    items: list[dict] = []
    for f in sorted(Path(dir_path).glob("*.json")):
        data = json.loads(f.read_text())
        items.extend(data if isinstance(data, list) else [data])
    correct = 0
    scores = []
    for it in items:
        vid = it["video_id"]
        fps = float(load_video(vid)["fps"])
        margin = cfg.window_margin_s
        f0 = max(0, int((it["start"] - margin) * fps))
        f1 = int((it["end"] + margin) * fps)
        td = load_tracking(vid, f0, f1, fps=fps)
        feats = compute_features(td, cfg)
        g = score_serve(feats, it["start"], it["end"], cfg)
        pred_serve = g.score >= cfg.serve_gate_min
        ok = pred_serve == (it["label"] == "serve")
        correct += ok
        scores.append({**it, "score": round(g.score, 3), "correct": ok,
                       "components": {k: round(v, 3)
                                      for k, v in g.components.items()}})
        log.info("%s %.1f-%.1f label=%s score=%.3f %s", vid[:8], it["start"],
                 it["end"], it["label"], g.score, "OK" if ok else "WRONG")
    acc = correct / len(items) if items else float("nan")
    log.info("serve-gate accuracy: %d/%d = %.3f", correct, len(items), acc)
    return {"accuracy": acc, "n": len(items), "items": scores}


def main() -> None:
    ap = argparse.ArgumentParser(description="Refinement eval / sweep")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", default=None, help="comma-separated ids/filenames")
    ap.add_argument("--classifier", choices=["rules", "xgboost"], default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--include-holdout", action="store_true",
                    help="FINAL evaluation only — never tune with this on")
    ap.add_argument("--serve-dir", default=None,
                    help="directory of labeled serve/non-serve spans")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    logging.getLogger("refine.run").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.classifier:
        cfg = cfg.copy(classifier=args.classifier)
    if args.serve_dir:
        validate_serve_gate(args.serve_dir, cfg)
    elif args.sweep:
        sweep(cfg)
    else:
        clips = args.clips.split(",") if args.clips else None
        evaluate(cfg, clips, include_holdout=args.include_holdout)


if __name__ == "__main__":
    main()
