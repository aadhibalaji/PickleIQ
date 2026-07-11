"""Grade-scored serve-gate recommendation (measure + recommend, no auto-apply).

The offline harness (evaluate.py) scores refine output against human RALLY spans
— positive-only ground truth. This module instead scores against per-window
USER GRADES (keep/drop over ALL Pass 1 windows, collected in the grading UI) and
recommends the ``serve_gate_min`` that best matches them. Nothing is applied
automatically; the API surfaces the suggestion for manual apply (the override
YAML). Boundary refinement is out of scope — only the keep/drop verdict matters
here, so windows are compared by frame-span IoU, not boundary error.
"""
from __future__ import annotations

import itertools
import logging

from .classifiers import LiveClassifier, make_classifier
from .config import RefineConfig
from .evaluate import iou            # reuse the harness IoU (frame-span safe)
from .refine import refine_video

log = logging.getLogger("refine.recommend")

MATCH_IOU = 0.5     # a grade maps to a window when their spans overlap this much
MIN_GRADED = 8      # below this the recommendation is not trustworthy

# Serve-gate knobs only. serve_weights is a dict — a nested search over it is a
# follow-up; v1 recommends the scalar threshold(s).
DEFAULT_GRID: dict[str, list] = {
    "serve_gate_min": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
}


def _windows_from_records(records: list[dict]) -> list[tuple[tuple[int, int], bool]]:
    """One ((start_frame, end_frame), ai_kept) per Pass 1 window. A split window
    counts as kept if ANY of its segments survived."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["pass1_window_id"], []).append(r)
    out = []
    for recs in groups.values():
        span = (recs[0]["pass1_start_frame"], recs[0]["pass1_end_frame"])
        ai_kept = any(r.get("verdict") == "rally" for r in recs)
        out.append((span, ai_kept))
    return out


def score_against_grades(cfg: RefineConfig, grades_by_clip: dict[str, list[dict]],
                         classifier: LiveClassifier | None = None) -> dict:
    """Keep/drop accuracy of ``cfg`` against the user grades, pooled across clips.
    Positives = human 'keep'. A grade whose span matches no current window (under
    this cfg) is skipped."""
    classifier = classifier or make_classifier(cfg)
    tp = fp = fn = tn = 0
    for vid, grades in grades_by_clip.items():
        if not grades:
            continue
        records, _ = refine_video(vid, cfg, classifier)
        windows = _windows_from_records(records)
        for g in grades:
            gspan = (g["pass1_start_frame"], g["pass1_end_frame"])
            best_kept, best_iou = None, MATCH_IOU
            for span, ai_kept in windows:
                v = iou(gspan, span)
                if v >= best_iou:
                    best_iou, best_kept = v, ai_kept
            if best_kept is None:
                continue
            human_keep = g["human_verdict"] == "keep"
            if human_keep and best_kept:
                tp += 1
            elif human_keep and not best_kept:
                fn += 1          # AI dropped a real rally (the target failure)
            elif not human_keep and best_kept:
                fp += 1          # AI kept junk
            else:
                tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / n if n else 0.0, "f1": f1,
            "precision": prec, "recall": rec, "n": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "dropped_wrong": fn, "kept_wrong": fp}


def recommend(base_cfg: RefineConfig, grades_by_clip: dict[str, list[dict]],
              grid: dict[str, list] | None = None) -> dict:
    """Grid-search serve-gate knobs against the grades; return current vs best.
    Ranks by keep/drop accuracy, then F1. Reuses one classifier across cfgs (the
    swept knobs don't affect P(live), only the serve gate), exactly as
    evaluate.sweep does."""
    grid = grid or DEFAULT_GRID
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    classifier = make_classifier(base_cfg)
    results = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        s = score_against_grades(base_cfg.copy(**overrides), grades_by_clip,
                                 classifier)
        results.append((overrides, s))
    results.sort(key=lambda x: (-x[1]["accuracy"], -x[1]["f1"]))
    current = score_against_grades(base_cfg, grades_by_clip, classifier)
    best_ov, best = results[0]
    return {
        "current": {"serve_gate_min": base_cfg.serve_gate_min, **current},
        "recommended": {"overrides": best_ov,
                        "delta": round(best["accuracy"] - current["accuracy"], 4),
                        **best},
        "candidates": [{"overrides": ov, "accuracy": round(s["accuracy"], 4),
                        "f1": round(s["f1"], 4), "dropped_wrong": s["dropped_wrong"],
                        "kept_wrong": s["kept_wrong"]} for ov, s in results],
    }
