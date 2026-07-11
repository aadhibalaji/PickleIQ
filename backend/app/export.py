"""Rallies-only video export.

Concatenates the refined rally segments (Pass 2's keep verdicts with snapped
boundaries, from data/refine/<id>.json) into one browser-playable H.264 mp4 —
the clip with all dead time cut out. Uses the ffmpeg binary bundled with
imageio-ffmpeg (no system ffmpeg needed). Output lives next to the source
video as ``cropped.mp4``; it is stale (and regenerated on demand) whenever the
refine JSON is newer than it.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from . import db
from .jobs import JobHandle

log = logging.getLogger("export")


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _refine_json_path(video_id: str) -> Path:
    from .refine.config import load_config, override_path
    from .refine.refine import resolve_path
    cfg = load_config(override_path())
    return resolve_path(cfg.out_dir) / f"{video_id}.json"


def rally_spans(video_id: str) -> list[tuple[float, float]]:
    """Kept-rally (start_s, end_s) spans from the refine JSON, sorted and
    merged where overlapping (adjacent windows' pre/post rolls can touch)."""
    path = _refine_json_path(video_id)
    if not path.exists():
        return []
    records = json.loads(path.read_text()).get("records", [])
    spans = sorted(
        (r["refined_start"], r["refined_end"]) for r in records
        if r.get("verdict") == "rally" and r.get("refined_start") is not None)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def cropped_path(video_id: str) -> Path:
    with db.connect() as con:
        row = con.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise ValueError("video not found")
    return Path(row["path"]).parent / "cropped.mp4"


def cropped_status(video_id: str) -> dict:
    """What the UI needs: does a cropped video exist, is it stale (refine
    results changed since it was built), and what would/does it contain."""
    spans = rally_spans(video_id)
    out = cropped_path(video_id)
    refine_json = _refine_json_path(video_id)
    exists = out.exists()
    stale = (exists and refine_json.exists()
             and refine_json.stat().st_mtime > out.stat().st_mtime)
    return {
        "exists": exists, "stale": stale, "n_rallies": len(spans),
        "total_s": round(sum(e - s for s, e in spans), 1),
        "size_bytes": out.stat().st_size if exists else None,
    }


def build_cropped(video_id: str, handle: JobHandle) -> Path:
    """Cut + concatenate the rally spans into cropped.mp4 (H.264, faststart).

    One ffmpeg invocation with a trim/concat filter graph — frame-accurate
    boundaries (stream-copy would snap to keyframes and drift seconds). Tried
    with audio first; source clips without an audio stream get a video-only
    retry.
    """
    spans = rally_spans(video_id)
    if not spans:
        raise RuntimeError("no kept rallies to export — run Pass 2 first")
    with db.connect() as con:
        row = con.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()
    src = Path(row["path"])
    out = cropped_path(video_id)
    tmp = out.with_suffix(".tmp.mp4")

    def _cmd(with_audio: bool) -> list[str]:
        parts, v_labels, a_labels = [], [], []
        for i, (s, e) in enumerate(spans):
            parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},"
                         f"setpts=PTS-STARTPTS[v{i}]")
            v_labels.append(f"[v{i}]")
            if with_audio:
                parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},"
                             f"asetpts=PTS-STARTPTS[a{i}]")
                a_labels.append(f"[a{i}]")
        n = len(spans)
        if with_audio:
            pairs = "".join(v + a for v, a in zip(v_labels, a_labels))
            parts.append(f"{pairs}concat=n={n}:v=1:a=1[outv][outa]")
            maps = ["-map", "[outv]", "-map", "[outa]", "-c:a", "aac"]
        else:
            parts.append(f"{''.join(v_labels)}concat=n={n}:v=1[outv]")
            maps = ["-map", "[outv]", "-an"]
        return [_ffmpeg_exe(), "-y", "-i", str(src),
                "-filter_complex", ";".join(parts), *maps,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-movflags", "+faststart", str(tmp)]

    handle.update(progress=0.05,
                  message=f"Cropping {len(spans)} rallies "
                          f"({sum(e - s for s, e in spans):.0f}s)…")
    for with_audio in (True, False):
        proc = subprocess.run(_cmd(with_audio), capture_output=True, text=True)
        if proc.returncode == 0:
            break
        if with_audio:
            log.info("cropped export retrying without audio (%s)",
                     proc.stderr.strip().splitlines()[-1] if proc.stderr else "?")
            continue
        raise RuntimeError(f"ffmpeg failed: "
                           f"{proc.stderr.strip().splitlines()[-1] if proc.stderr else 'unknown'}")
    tmp.replace(out)
    handle.update(progress=1.0, message=f"Cropped video ready ({len(spans)} rallies)")
    log.info("cropped video for %s: %s (%d rallies)", video_id, out, len(spans))
    return out
