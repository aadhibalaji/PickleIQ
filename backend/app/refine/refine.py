"""Orchestrator: Pass 1 candidate windows in -> refinement records out.

Per candidate window: load tracking (window ± margin only), Stage A features,
Stage B P(live), Stage C temporal smoothing, Stage D serve gate + boundary
snapping — then emit one record per surviving segment (a window containing
two rallies emits two records with the same ``pass1_window_id``) or a single
rejection record.

Output contract (per record)::

    {"pass1_window_id": str, "pass1_start": s, "pass1_end": s,
     "pass1_start_frame": int, "pass1_end_frame": int,
     "verdict": "rally"|"rejected",
     "rejection_reason": null|"no_serve"|"low_live_ratio"|"too_short",
     "refined_start": s|null, "refined_end": s|null,
     "refined_start_frame": int|null, "refined_end_frame": int|null,
     "confidence": {"mean_p_live": f, "serve_gate_score": f,
                    "net_crossings_per_10s": f},
     "splits": [[s,e], ...]}   # sibling segments when the window split, else []

All timestamps are seconds from clip start; frames are also included since
the rest of the repo speaks frames. JSON-only output for now — writes no
``rally_labels`` rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classifiers import LiveClassifier, make_classifier
from .config import RefineConfig, load_config
from .features import Features, compute_features
from .io_loaders import (Pass1Window, TrackingData, load_pass1_windows,
                         load_tracking, load_video)
from .postprocess import postprocess
from .serve_gate import ServeGateResult, score_serve, snap_end
from ..paths import repo_root

log = logging.getLogger("refine.run")

_ROOT = repo_root()

# ponytail: process-lifetime cache, unbounded — fine for sweep runs (a handful
# of labeled clips), would leak if used against many distinct videos long-term.
_STAGE_A_CACHE: dict[tuple, list[tuple[Pass1Window, TrackingData]]] = {}


def _stage_a_windows(video_id: str, cfg: RefineConfig,
                     p1_source: str = "pass1") -> list[tuple[Pass1Window, TrackingData]]:
    """Pass 1 windows + their Stage A tracking (ballistic reconstruction),
    cached per (video, ballistic settings). Stage A is ~5s/clip and doesn't
    depend on Stage C/D/serve-gate knobs, so parameter sweeps over those can
    reuse it instead of re-reconstructing on every grid point. Keyed on the
    ballistic settings too — a sweep must not vary those, or grade the cache
    stale by resetting _STAGE_A_CACHE first."""
    key = (video_id, cfg.use_ballistic_3d, cfg.ballistic_max_rms_px,
          cfg.window_margin_s, p1_source)
    cached = _STAGE_A_CACHE.get(key)
    if cached is not None:
        return cached
    video = load_video(video_id)
    fps = float(video["fps"])
    last_frame = int(video["frame_count"]) - 1
    margin = int(round(cfg.window_margin_s * fps))
    out = []
    for w in load_pass1_windows(video_id, p1_source):
        f0 = max(0, w.start_frame - margin)
        f1 = min(last_frame, w.end_frame + margin)
        td = load_tracking(video_id, f0, f1, fps=fps,
                           reconstruct_3d=cfg.use_ballistic_3d,
                           max_rms_px=cfg.ballistic_max_rms_px)
        out.append((w, td))
    _STAGE_A_CACHE[key] = out
    return out


def refine_video_fast(video_id: str, cfg: RefineConfig, classifier: LiveClassifier,
                      p1_source: str = "pass1") -> list[dict]:
    """Like refine_video, but reuses cached Stage A tracking across calls —
    for sweeps only, where only Stage B/C/D/gate knobs vary between calls."""
    records: list[dict] = []
    for w, td in _stage_a_windows(video_id, cfg, p1_source):
        records.extend(refine_window(w, td, cfg, classifier).records)
    return records


def _gate_checks(gate: ServeGateResult, cfg: RefineConfig, mean_p: float) -> list[dict]:
    """The mean_p_live + serve-gate hard-check breakdown shared by every
    per-segment diagnostic (rejected at serve_gate_min, rejected too_short,
    or kept) — the pass 2 "why" panel renders this list directly."""
    return [
        {"name": "mean_p_live", "value": round(mean_p, 4),
         "threshold": cfg.mean_p_live_min, "op": ">=", "pass": True},
        {"name": "pre_serve_formation", "value": round(gate.components.get("pre_serve", 0.0), 4),
         "threshold": cfg.serve_pre_serve_min, "op": ">=", "pass": gate.pre_serve_ok,
         "pose_coverage": round(gate.pre_serve_pose_coverage, 3)},
        {"name": "pre_serve_formation (compensation bar)",
         "value": round(gate.components.get("pre_serve", 0.0), 4),
         "threshold": cfg.serve_pre_serve_compensate_min, "op": ">=",
         "pass": gate.pre_serve_compensate_ok},
        {"name": "contact", "value": round(gate.components.get("contact", 0.0), 4),
         "threshold": cfg.serve_contact_min, "op": ">=", "pass": gate.contact_ok},
        {"name": "contact (near-floor compensation bar)",
         "value": round(gate.components.get("contact", 0.0), 4),
         "threshold": cfg.serve_contact_near_floor_min, "op": ">=",
         "pass": gate.contact_near_floor_ok},
        {"name": "flight", "value": round(gate.components.get("flight", 0.0), 4),
         "threshold": "0.8 (crossing found, undirected) | 1.0 (direction confirmed)",
         "op": "==", "pass": gate.flight_directional,
         "directional": gate.flight_directional},
        {"name": "flight_or_landing_confirmed", "value": gate.flight_or_landing_ok,
         "threshold": "landing > 0 or flight directionally confirmed (== 1.0)",
         "op": "==", "pass": gate.flight_or_landing_ok},
        {"name": "gate_pass_path", "value": gate.pass_path or "none",
         "threshold": "contact | formation_compensation | directional_flight_compensation",
         "op": "in", "pass": gate.pass_path is not None},
        {"name": "serve_gate_score", "value": round(gate.score, 4),
         "threshold": cfg.serve_gate_min, "op": ">=", "pass": gate.score >= cfg.serve_gate_min},
    ]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else _ROOT / p


@dataclass
class WindowDebug:
    """Everything diagnostics needs to draw one window's timeline."""
    window: Pass1Window
    feats: Features
    p_live: np.ndarray
    state: np.ndarray
    segments: list[tuple[float, float]]           # Stage C segments (seconds)
    serve: list[ServeGateResult] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


def _record(window: Pass1Window, fps: float, verdict: str,
            reason: str | None, start: float | None, end: float | None,
            mean_p: float, gate: float, cross10: float,
            diag: dict | None = None) -> dict:
    return {
        "pass1_window_id": window.id,
        "pass1_start": round(window.start_frame / fps, 3),
        "pass1_end": round(window.end_frame / fps, 3),
        "pass1_start_frame": window.start_frame,
        "pass1_end_frame": window.end_frame,
        "verdict": verdict,
        "rejection_reason": reason,
        "refined_start": None if start is None else round(start, 3),
        "refined_end": None if end is None else round(end, 3),
        "refined_start_frame": None if start is None else int(round(start * fps)),
        "refined_end_frame": None if end is None else int(round(end * fps)),
        "confidence": {"mean_p_live": round(mean_p, 4),
                       "serve_gate_score": round(gate, 4),
                       "net_crossings_per_10s": round(cross10, 3)},
        "splits": [],
        "diagnostics": diag or {},
    }


def refine_window(window: Pass1Window, td: TrackingData, cfg: RefineConfig,
                  classifier: LiveClassifier) -> WindowDebug:
    """Run stages A-D over one candidate window's tracking span."""
    feats = compute_features(td, cfg)
    p = classifier.predict_proba(feats)
    segs, state = postprocess(p, cfg)
    fps = td.fps
    dbg = WindowDebug(window=window, feats=feats, p_live=p, state=state,
                      segments=[(float(feats.t[s.start]),
                                 float(feats.t[s.end - 1])) for s in segs])
    span_lo, span_hi = float(feats.t[0]), float(feats.t[-1])

    if not segs:
        pmax = float(p.max(initial=0.0))
        reason = "low_live_ratio" if pmax < cfg.enter_thresh else "too_short"
        diag = {
            "checks": [{"name": "p_live_max", "value": round(pmax, 4),
                        "threshold": cfg.enter_thresh, "op": ">=",
                        "pass": pmax >= cfg.enter_thresh}],
            "stage": "enter_thresh (Stage B)" if reason == "low_live_ratio"
                     else "min_rally_s_early (Stage C, pre-serve-gate)",
            "note": ("no timestep ever crossed enter_thresh" if reason == "low_live_ratio"
                     else "P(live) crossed enter_thresh, but postprocess.py's early "
                          "min_rally_s_early noise-blip filter dropped every Stage C "
                          "segment before it reached the serve gate / anchor / "
                          "snap_end stage — the raw segment length that got filtered "
                          "isn't retained here"),
        }
        dbg.records.append(_record(window, fps, "rejected", reason,
                                   None, None, float(p.mean()) if len(p) else 0.0,
                                   0.0, 0.0, diag))
        return dbg

    accepted: list[dict] = []
    for si, (seg, (t_start, t_end)) in enumerate(zip(segs, dbg.segments)):
        mean_p = float(p[seg.start:seg.end].mean())
        # For the window's FIRST segment the serve sits near the window start,
        # which the smoothed rising edge can lag — search from there.
        search_from = window.start_frame / fps if si == 0 else None
        gate = score_serve(feats, t_start, t_end, cfg, search_from=search_from)
        dbg.serve.append(gate)
        n_cross = int(np.sum((feats.crossings_t >= t_start)
                             & (feats.crossings_t <= t_end)))
        dur = max(t_end - t_start, 1e-6)
        cross10 = n_cross / dur * 10.0
        facing_mean = (float(feats.cols["facing_away_retreat"][seg.start:seg.end].mean())
                       if seg.end > seg.start else 0.0)
        raw_seg = [round(t_start, 3), round(t_end, 3)]
        # Explicit raw-live-segment-start vs. contact-anchor comparison — a
        # contact anchor sitting well BEFORE the segment even starts is a red
        # flag that it latched onto a stray ball event unrelated to this
        # segment's serve (see _formation_window's contact-search
        # localization), not a legitimately early-detected real contact.
        contact_vs_segment_start = {
            "contact_t": round(gate.contact_t, 3) if gate.contact_t is not None else None,
            "raw_segment_start": round(t_start, 3),
            "contact_before_segment_start": (gate.contact_t is not None
                                             and gate.contact_t < t_start),
        }

        if mean_p < cfg.mean_p_live_min:
            diag = {"checks": [{"name": "mean_p_live", "value": round(mean_p, 4),
                                 "threshold": cfg.mean_p_live_min, "op": ">=", "pass": False}],
                    "stage": "mean_p_live_min",
                    "raw_segment": raw_seg, "facing_away_retreat_mean": round(facing_mean, 4)}
            dbg.records.append(_record(window, fps, "rejected", "low_live_ratio",
                                       None, None, mean_p, gate.score, cross10, diag))
            continue
        if gate.score < cfg.serve_gate_min:
            diag = {"checks": _gate_checks(gate, cfg, mean_p),
                    "stage": "serve_gate_min",
                    "serve_gate_components": {k: round(v, 4) for k, v in gate.components.items()},
                    "contact_vs_segment_start": contact_vs_segment_start,
                    "raw_segment": raw_seg, "facing_away_retreat_mean": round(facing_mean, 4)}
            dbg.records.append(_record(window, fps, "rejected", "no_serve",
                                       None, None, mean_p, gate.score, cross10, diag))
            continue

        # Start anchor: serve contact when found; else back off from the first
        # net crossing by a typical serve-flight time (the rising edge lags the
        # real start, a crossing never precedes the serve); else rising edge.
        if gate.contact_t is not None:
            anchor = gate.contact_t
            anchor_source = "serve_contact"
        else:
            cross = feats.crossings_t[(feats.crossings_t >= t_start - cfg.serve_lookback_s)
                                      & (feats.crossings_t <= t_end)]
            if len(cross):
                anchor = min(t_start, float(cross[0]) - 0.9)
                anchor_source = "crossing_fallback"
            else:
                anchor = t_start
                anchor_source = "rising_edge"
        # A late-detected serve contact (anchor well inside the segment)
        # shouldn't eat into rally time the segment already established as
        # live — clamp to whichever is EARLIER: the segment's own detected
        # start, or the anchor's pre-roll. Previously this always used
        # anchor - pre_roll_s regardless of t_start, so a contact found e.g.
        # 2.9s into a 5.0s segment chopped the first 2.4s of real play.
        start = max(span_lo, min(t_start, anchor - cfg.pre_roll_s))
        ev_t, snap_kind = snap_end(feats, max(t_start, start), t_end, cfg)
        end = min(span_hi, ev_t + cfg.post_roll_s)
        diag = {
            "checks": _gate_checks(gate, cfg, mean_p) + [
                {"name": "refined_duration_s", "value": round(end - start, 4),
                 "threshold": cfg.min_rally_s, "op": ">=",
                 "pass": (end - start) >= cfg.min_rally_s}],
            "stage": "min_rally_s (post-trim)",
            "serve_gate_components": {k: round(v, 4) for k, v in gate.components.items()},
            "contact_vs_segment_start": contact_vs_segment_start,
            "facing_away_retreat_mean": round(facing_mean, 4),
            "raw_segment": raw_seg,
            "anchor_t": round(anchor, 4), "anchor_source": anchor_source,
            "snap_end_t": round(ev_t, 4), "snap_end_kind": snap_kind,
        }
        if end - start < cfg.min_rally_s:
            dbg.records.append(_record(window, fps, "rejected", "too_short",
                                       start, end, mean_p, gate.score, cross10, diag))
            continue
        rec = _record(window, fps, "rally", None, start, end,
                      mean_p, gate.score, cross10, diag)
        accepted.append(rec)
        dbg.records.append(rec)

    if len(accepted) > 1:   # window split into several rallies
        spans = [[r["refined_start"], r["refined_end"]] for r in accepted]
        for r in accepted:
            r["splits"] = spans
    return dbg


class IncrementalRefiner:
    """Refine Pass 1 windows one at a time, as soon as Pass 2 has written the
    overlay span covering them — so the UI sees verdicts + refined boundaries
    grow live during the pass instead of all-at-once at the end.

    ``advance(upto_frame)`` refines every not-yet-done window whose span is
    fully covered and rewrites the output JSON with ``partial: true``; the
    normal full refine after the pass (:func:`run_refine_json`) replaces it
    with the canonical final artifact. Construction drops any stale JSON from
    a previous run so old verdicts never linger while the new pass scans.
    """

    def __init__(self, video_id: str, cfg: RefineConfig | None = None,
                 p1_source: str = "pass1"):
        self.video_id = video_id
        self.cfg = cfg or RefineConfig()
        self.classifier = make_classifier(self.cfg)
        video = load_video(video_id)
        self.fps = float(video["fps"])
        self.last_frame = int(video["frame_count"]) - 1
        self.windows = load_pass1_windows(video_id, p1_source)
        self.records: list[dict] = []
        self._done: set[str] = set()
        self.out = resolve_path(self.cfg.out_dir) / f"{video_id}.json"
        self.out.unlink(missing_ok=True)

    def advance(self, upto_frame: int) -> None:
        margin = int(round(self.cfg.window_margin_s * self.fps))
        new = [w for w in self.windows
               if w.id not in self._done and w.end_frame <= upto_frame]
        if not new:
            return
        for w in new:
            f0 = max(0, w.start_frame - margin)
            f1 = min(self.last_frame, w.end_frame + margin)
            td = load_tracking(self.video_id, f0, f1, fps=self.fps,
                               reconstruct_3d=self.cfg.use_ballistic_3d,
                               max_rms_px=self.cfg.ballistic_max_rms_px)
            dbg = refine_window(w, td, self.cfg, self.classifier)
            self.records.extend(dbg.records)
            self._done.add(w.id)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(
            {"video_id": self.video_id, "partial": True,
             "records": self.records}, indent=2))
        log.info("incremental refine %s: %d/%d windows",
                 self.video_id, len(self._done), len(self.windows))


def refine_video(video_id: str, cfg: RefineConfig | None = None,
                 classifier: LiveClassifier | None = None,
                 p1_source: str = "pass1") -> tuple[list[dict], list[WindowDebug]]:
    """Refine every Pass 1 window of one clip. Returns (records, debug)."""
    cfg = cfg or RefineConfig()
    classifier = classifier or make_classifier(cfg)
    video = load_video(video_id)
    fps = float(video["fps"])
    last_frame = int(video["frame_count"]) - 1
    margin = int(round(cfg.window_margin_s * fps))

    t0 = time.perf_counter()
    records: list[dict] = []
    debugs: list[WindowDebug] = []
    for w in load_pass1_windows(video_id, p1_source):
        f0 = max(0, w.start_frame - margin)
        f1 = min(last_frame, w.end_frame + margin)
        td = load_tracking(video_id, f0, f1, fps=fps,
                           reconstruct_3d=cfg.use_ballistic_3d,
                           max_rms_px=cfg.ballistic_max_rms_px)
        dbg = refine_window(w, td, cfg, classifier)
        records.extend(dbg.records)
        debugs.append(dbg)
    log.info("refine %s: %d windows -> %d records (%d rallies) in %.0f ms",
             video_id, len(debugs), len(records),
             sum(r["verdict"] == "rally" for r in records),
             (time.perf_counter() - t0) * 1e3)
    return records, debugs


def run_refine_json(video_id: str, cfg: RefineConfig | None = None,
                    plot: bool = False) -> Path:
    """Refine one clip and write ``<out_dir>/<video_id>.json`` plus the
    per-segment confidence log (JSONL, feeds the human-review queue)."""
    cfg = cfg or RefineConfig()
    records, debugs = refine_video(video_id, cfg)
    out_dir = resolve_path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video_id}.json"
    out.write_text(json.dumps({"video_id": video_id, "records": records},
                              indent=2))
    from .diagnostics import write_confidence_log
    write_confidence_log(out_dir / f"{video_id}.segments.jsonl",
                         video_id, records)
    if plot:
        from .diagnostics import plot_clip
        plot_clip(video_id, debugs, cfg, out_dir / f"{video_id}.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass-2 rally refinement")
    ap.add_argument("video_id")
    ap.add_argument("--config", default=None, help="YAML config overrides")
    ap.add_argument("--classifier", choices=["rules", "xgboost"], default=None)
    ap.add_argument("--plot", action="store_true", help="write diagnostic PNG")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config)
    if args.classifier:
        cfg = cfg.copy(classifier=args.classifier)
    out = run_refine_json(args.video_id, cfg, plot=args.plot)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
