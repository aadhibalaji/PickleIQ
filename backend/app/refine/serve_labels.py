"""Stage E (serve labels) — turn human rally-start/end marks into a
model-ready SERVE dataset.

A rally cannot begin without a serve, so every human rally-START timestamp IS
a serve-contact timestamp. From the marks a clip already has (``rally_labels``
source='human', loaded through :mod:`labels` — untouched here) this module
derives, per clip:

* the per-instant **serve-events sidecar** (``<id>.serves.json``): serve times +
  rally spans + the exclusion buffer. The primary artifact, ready for the
  reserved ``serve_start`` detector;
* the per-timestep serve target + a ±buffer **exclusion mask**
  (:func:`expand_serve_labels`, mirroring :func:`labels.expand_labels`): 1.0 at
  the marked contact, 0.0 on clean negatives, and masked-OUT (neither positive
  nor negative) inside the buffer around each serve;
* the span-level ``{"video_id","start","end","label"}`` items
  (``serve_gate/<id>.json``) that :func:`evaluate.validate_serve_gate` already
  consumes.

Negatives are IMPLICIT: everything outside every ±buffer is the serve-negative
class — nothing is hand-labeled. Times are SECONDS throughout (what the serve
consumers speak); ``fps`` travels in the sidecar so frames are recoverable.

Rally spans still load through :mod:`labels`; only serves live here. The
ball-tracking labels (``ball_corrections`` / ``ball_tracks``, keyed by the same
``video_id``) are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import RefineConfig
from .io_loaders import load_video
from .labels import load_ground_truth

SERVE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# projections from rally spans
# ---------------------------------------------------------------------------

def build_sidecar(video_id: str, rally_spans: list[tuple[float, float]],
                  fps: float, duration_s: float, cfg: RefineConfig) -> dict:
    """Per-instant serve-events sidecar. Each serve time == the rally start."""
    return {
        "schema_version": SERVE_SCHEMA_VERSION,
        "video_id": video_id,
        "fps": round(float(fps), 6),
        "duration_s": round(float(duration_s), 3),
        "buffer_s": round(float(cfg.serve_buffer_s), 3),
        "rally_spans": [[round(s, 3), round(e, 3)] for s, e in rally_spans],
        "serve_events": [round(s, 3) for s, _ in rally_spans],
    }


def expand_serve_labels(t: np.ndarray, serve_events: list[float],
                        buffer_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep serve labels for a feature timeline ``t`` (seconds).

    Returns ``(y, sample_mask)``:

    * ``y[i]``           1.0 at the timestep nearest each serve, else 0.0;
    * ``sample_mask[i]`` False inside ±``buffer_s`` of a serve (omit from the
      NEGATIVE set — it is neither a positive nor a hard negative), True
      elsewhere. The serve instant itself is always kept (mask True there).

    Train on ``y[sample_mask]`` vs features ``X[sample_mask]``: positives are
    the marked contacts, negatives are every kept timestep with ``y==0``, and
    the buffered neighbourhood is simply absent.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.zeros(len(t), dtype=np.float64)
    mask = np.ones(len(t), dtype=bool)
    if not len(t):
        return y, mask
    for s in serve_events:
        mask[(t >= s - buffer_s) & (t <= s + buffer_s)] = False
        i = int(np.argmin(np.abs(t - s)))
        y[i] = 1.0          # the marked contact is the sole positive ...
        mask[i] = True      # ... and is kept even though it sits in the buffer
    return y, mask


def to_gate_spans(video_id: str, rally_spans: list[tuple[float, float]],
                  duration_s: float, cfg: RefineConfig) -> list[dict]:
    """Span-level serve/non-serve items for :func:`evaluate.validate_serve_gate`.

    * positives — each rally span (a real rally opens with a serve);
    * negatives — the dead-time gaps between/around rallies, trimmed by the
      buffer and kept only when at least ``serve_neg_min_span_s`` long.
    """
    buf = cfg.serve_buffer_s
    items: list[dict] = [
        {"video_id": video_id, "start": round(s, 3), "end": round(e, 3),
         "label": "serve"} for s, e in rally_spans]

    def add_gap(g0: float, g1: float) -> None:
        if g1 - g0 >= cfg.serve_neg_min_span_s:
            items.append({"video_id": video_id, "start": round(g0, 3),
                          "end": round(g1, 3), "label": "non-serve"})

    prev_end = 0.0
    for s, e in rally_spans:
        add_gap(prev_end + buf if prev_end > 0.0 else 0.0, s - buf)
        prev_end = e
    add_gap(prev_end + buf, duration_s)
    return items


def rally_file_items(rally_spans: list[tuple[float, float]]) -> list[dict]:
    """Rally intervals in the exact shape :func:`labels.load_label_file` reads
    (seconds JSON ``[{"start": s, "end": e}, ...]``)."""
    return [{"start": round(s, 3), "end": round(e, 3)} for s, e in rally_spans]


# ---------------------------------------------------------------------------
# orchestration + IO
# ---------------------------------------------------------------------------

def _labels_dir(cfg: RefineConfig) -> Path:
    from .refine import resolve_path
    return resolve_path(cfg.out_dir) / "labels"


def write_dataset(video_id: str, sidecar: dict, rallies: list[dict],
                  gate_spans: list[dict], cfg: RefineConfig,
                  out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write the three per-clip files. Serve-gate spans go in their own
    subdir so ``validate_serve_gate``'s ``*.json`` glob stays clean."""
    base = Path(out_dir) if out_dir is not None else _labels_dir(cfg)
    gate_dir = base / "serve_gate"
    base.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "serves": base / f"{video_id}.serves.json",
        "rallies": base / f"{video_id}.rallies.json",
        "serve_gate": gate_dir / f"{video_id}.json",
    }
    paths["serves"].write_text(json.dumps(sidecar, indent=2))
    paths["rallies"].write_text(json.dumps(rallies, indent=2))
    paths["serve_gate"].write_text(json.dumps(gate_spans, indent=2))
    return paths


def build_dataset(video_id: str, cfg: RefineConfig | None = None,
                  out_dir: str | Path | None = None,
                  write: bool = True) -> dict:
    """Build (and, unless ``write=False``, write) the serve dataset for one clip
    from its human rally marks. Returns a preview/summary dict — pass
    ``write=False`` for a dry run that touches no files."""
    cfg = cfg or RefineConfig()
    video = load_video(video_id)
    fps = float(video["fps"])
    rally_spans = sorted(load_ground_truth(video_id))       # seconds, from DB
    fc = video.get("frame_count")
    duration_s = (float(fc) / fps if fc
                  else max((e for _, e in rally_spans), default=0.0) + cfg.post_roll_s)

    sidecar = build_sidecar(video_id, rally_spans, fps, duration_s, cfg)
    rallies = rally_file_items(rally_spans)
    gate_spans = to_gate_spans(video_id, rally_spans, duration_s, cfg)

    result = {
        "video_id": video_id,
        "filename": video.get("filename"),
        "fps": fps,
        "duration_s": round(duration_s, 3),
        "buffer_s": cfg.serve_buffer_s,
        "n_rallies": len(rally_spans),
        "n_serves": len(rally_spans),
        "n_gate_negatives": sum(1 for it in gate_spans if it["label"] == "non-serve"),
        "serves_sidecar": sidecar,
        "rallies_file": rallies,
        "serve_gate_spans": gate_spans,
    }
    if write:
        paths = write_dataset(video_id, sidecar, rallies, gate_spans, cfg, out_dir)
        result["written"] = {k: str(v) for k, v in paths.items()}
    return result
