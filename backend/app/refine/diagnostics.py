"""Stage F — diagnostics: the one-look timeline plot + confidence logs.

``plot_clip`` renders, on a single time axis: raw P(live), the smoothed
LIVE state, Pass 1 candidate windows, refined segments (accepted green /
rejected red), serve-gate contact events, and ground-truth rallies. One PNG
per clip — this is the first thing to open when a boundary looks wrong.

``write_confidence_log`` appends one JSON line per output record so
low-confidence segments can be queued for human review (the future
retraining loop reads this).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import RefineConfig
from .refine import WindowDebug


def write_confidence_log(path: str | Path, video_id: str,
                         records: list[dict]) -> None:
    """One JSON line per record: verdict + confidence, for review queues."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({"video_id": video_id, **r}) + "\n")


def plot_clip(video_id: str, debugs: list[WindowDebug], cfg: RefineConfig,
              out_path: str | Path,
              ground_truth: list[tuple[float, float]] | None = None,
              title: str | None = None) -> Path:
    """Render the full-clip refinement timeline to PNG. `ground_truth` is a
    list of (start_s, end_s); when None, human rally_labels are loaded from
    the DB if present."""
    if ground_truth is None:
        ground_truth = _load_gt(video_id, debugs)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    # lanes (y): 0-1 P(live) curve; spans drawn in bands below/above
    for dbg in debugs:
        t = dbg.feats.t
        ax.plot(t, dbg.p_live, color="#4477aa", lw=1.0, zorder=3)
        ax.fill_between(t, 0, dbg.state.astype(float), color="#4477aa",
                        alpha=0.15, step="mid", zorder=2)
        w = dbg.window
        fps = dbg.feats.fps
        ax.axvspan(w.start_frame / fps, w.end_frame / fps, ymin=0.86, ymax=0.97,
                   color="#cccccc", alpha=0.8, zorder=1)
        for gate in dbg.serve:
            if gate.contact_t is not None:
                ax.plot([gate.contact_t], [1.02], marker="v", color="#aa3377",
                        ms=7, clip_on=False, zorder=5)
        for rec in dbg.records:
            if rec["verdict"] == "rally":
                ax.axvspan(rec["refined_start"], rec["refined_end"],
                           ymin=0.74, ymax=0.85, color="#228833", alpha=0.75)
            else:
                ax.axvspan(w.start_frame / fps, w.end_frame / fps,
                           ymin=0.74, ymax=0.85, color="#ee6677", alpha=0.35)
    for gs, ge in ground_truth or []:
        ax.axvspan(gs, ge, ymin=0.62, ymax=0.73, color="#ccbb44", alpha=0.8)

    ax.axhline(cfg.enter_thresh, color="#888", lw=0.6, ls="--")
    ax.axhline(cfg.exit_thresh, color="#888", lw=0.6, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(live)")
    ax.set_title(title or f"rally refinement — {video_id}")
    ax.legend(handles=[
        Patch(color="#cccccc", label="pass-1 window"),
        Patch(color="#228833", label="refined rally"),
        Patch(color="#ee6677", label="rejected"),
        Patch(color="#ccbb44", label="ground truth"),
        Patch(color="#4477aa", label="P(live) / LIVE state"),
        Patch(color="#aa3377", label="serve contact (▼)"),
    ], loc="lower right", ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _load_gt(video_id: str, debugs: list[WindowDebug]) -> list[tuple[float, float]]:
    try:
        from .labels import load_ground_truth
        return load_ground_truth(video_id)
    except Exception:
        return []
