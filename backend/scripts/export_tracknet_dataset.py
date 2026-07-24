"""Export ball_corrections + video frames into a TrackNet fine-tuning dataset.

Builds one training sample per LABELED frame: a 3-frame triplet ending at
that frame (oldest-first, matching ball_tracknet.TrackNetBall.infer_triplet's
own ordering and the model's "read the LAST heatmap" convention), each frame
resized to the model's native (INPUT_W, INPUT_H) via the exact same
cv2.resize call ball_tracknet.py uses at inference — no train/serve skew.
Neighbouring frames in a triplet are supervised too, WHEN they also happen to
carry their own separate label; otherwise that heatmap channel is left
unlabeled (the training script masks it out of the loss).

Frames are read via app.cv.video_io.iter_frames — ONE sequential pass per
video, never per-frame seeking (iter_frames' own docstring: seeking is "slow
and inaccurate on long GOP videos" — a wrong-frame label would silently
corrupt the dataset).

Output layout, under --out (default DATA_DIR/tracknet_dataset):
    frames/{video_id}/{frame:06d}.jpg   resized (INPUT_W x INPUT_H) BGR, as
                                         read off the real video — the
                                         training script does the same
                                         BGR->RGB conversion ball_tracknet.py
                                         does, from this same file.
    manifest.json                       samples + per-video split assignment

Usage:
    .venv/bin/python backend/scripts/export_tracknet_dataset.py
        [--video-ids ID [ID ...]] [--val-videos ID [ID ...]] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db, storage_s3  # noqa: E402
from app.config import DATA_DIR, STORAGE_BACKEND  # noqa: E402
from app.cv.ball_tracknet import INPUT_H, INPUT_W  # noqa: E402
from app.cv.video_io import iter_frames, probe  # noqa: E402


def _video_path(video_id: str) -> Path:
    """Resolve a real local file for this video — DB `path` may be a stale
    container path (e.g. /app/data/... from an S3-mode run mirrored down
    locally); fall back to the scratch convention storage_s3 uses, then
    finally try an actual S3 download if neither is already present."""
    with db.connect() as con:
        row = con.execute("SELECT path, s3_key FROM videos WHERE id=?", (video_id,)).fetchone()
    if row is None:
        raise KeyError(f"no such video: {video_id}")
    for candidate in (Path(row["path"]) if row["path"] else None, storage_s3.scratch_path(video_id)):
        if candidate is not None and candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    if row["s3_key"] and STORAGE_BACKEND == "s3":
        return storage_s3.download_to_scratch(row["s3_key"], video_id)
    raise FileNotFoundError(
        f"video {video_id}: no local file at {row['path']} or scratch, and "
        f"no S3 fallback available (STORAGE_BACKEND={STORAGE_BACKEND!r})")


def _labeled_frames(video_id: str) -> dict[int, tuple[float, float]]:
    """frame -> (x, y) in ORIGINAL video pixels, one mark per frame (last
    write wins in the unlikely case of duplicates)."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT frame, x, y FROM ball_corrections WHERE video_id=? ORDER BY frame",
            (video_id,)).fetchall()
    return {r["frame"]: (r["x"], r["y"]) for r in rows}


def _export_video(video_id: str, out_dir: Path) -> list[dict]:
    labels = _labeled_frames(video_id)
    if not labels:
        return []
    path = _video_path(video_id)
    meta = probe(str(path))

    # Triplet = [f-2, f-1, f] clamped at 0 (oldest-first, matches infer_triplet;
    # the labeled frame is always the NEWEST — the one the model's output is
    # actually read from at inference).
    triplets = {f: [max(0, f - 2), max(0, f - 1), f] for f in labels}
    needed = sorted({idx for trip in triplets.values() for idx in trip})

    frame_dir = out_dir / "frames" / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    found: set[int] = set()
    need_set = set(needed)
    last_needed = needed[-1]
    for idx, frame in iter_frames(str(path), start_frame=0, end_frame=last_needed + 1):
        if idx not in need_set:
            continue
        resized = cv2.resize(frame, (INPUT_W, INPUT_H))  # same call as ball_tracknet.infer_triplet
        cv2.imwrite(str(frame_dir / f"{idx:06d}.jpg"), resized)
        found.add(idx)
    missing = need_set - found
    if missing:
        print(f"  WARNING {video_id}: {len(missing)} needed frame(s) not found in the video "
              f"(short/truncated file?) — samples touching them are dropped: {sorted(missing)[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    samples = []
    for anchor, trip in triplets.items():
        if any(idx in missing for idx in trip):
            continue
        samples.append({
            "video_id": video_id,
            "anchor_frame": anchor,
            "frame_indices": trip,
            # per-triplet-position label in ORIGINAL pixel coords, or null —
            # the training script converts to heatmap coords itself (reusing
            # INPUT_W/INPUT_H from the same ball_tracknet module), so the
            # resize ratio is defined in exactly one place.
            "labels_px": [list(labels[idx]) if idx in labels else None for idx in trip],
            "orig_width": meta.width,
            "orig_height": meta.height,
        })
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids", nargs="*", default=None,
                    help="video_ids to export (default: every video_id with any ball_corrections rows)")
    ap.add_argument("--val-videos", nargs="*", default=None,
                    help="video_ids to hold out for validation (default: auto-pick the "
                         "video with the FEWEST labeled marks, to sacrifice the least "
                         "training data)")
    ap.add_argument("--out", default=None, help="output dir (default DATA_DIR/tracknet_dataset)")
    args = ap.parse_args()

    with db.connect() as con:
        all_ids = [r["video_id"] for r in
                   con.execute("SELECT DISTINCT video_id FROM ball_corrections").fetchall()]
    video_ids = args.video_ids or all_ids
    if not video_ids:
        print("no ball_corrections rows found for any video — nothing to export.")
        return

    out_dir = Path(args.out) if args.out else DATA_DIR / "tracknet_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_video_samples: dict[str, list[dict]] = {}
    per_video_counts: dict[str, int] = {}
    for vid in video_ids:
        print(f"exporting {vid} ...")
        samples = _export_video(vid, out_dir)
        per_video_samples[vid] = samples
        per_video_counts[vid] = len(_labeled_frames(vid))
        print(f"  {len(samples)} triplet(s) from {per_video_counts[vid]} labeled frame(s)")

    exported_ids = [v for v in video_ids if per_video_samples[v]]
    if not exported_ids:
        print("nothing exportable (no video had both labels and a readable file) — stopping.")
        return

    val_videos = args.val_videos
    if val_videos is None:
        val_videos = [min(exported_ids, key=lambda v: per_video_counts[v])]
        print(f"\nauto-selected val video (fewest labels, least training data sacrificed): "
              f"{val_videos[0]} ({per_video_counts[val_videos[0]]} marks)")
    train_videos = [v for v in exported_ids if v not in val_videos]
    if not train_videos:
        print("WARNING: every exported video is in val_videos — no training data left. Adjust --val-videos.")

    manifest = {
        "input_w": INPUT_W, "input_h": INPUT_H,
        "train_videos": train_videos, "val_videos": [v for v in val_videos if v in exported_ids],
        "samples": [s for v in exported_ids for s in per_video_samples[v]],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- report -------------------------------------------------------
    def _split_stats(vids: list[str]) -> dict:
        samples = [s for v in vids for s in per_video_samples.get(v, [])]
        n_labels_total = sum(1 for s in samples for lb in s["labels_px"] if lb is not None)
        n_anchor_only = len(samples)  # every triplet has >=1 label (the anchor)
        return {"videos": vids, "triplets": n_anchor_only, "labeled_heatmap_frames": n_labels_total}

    train_stats = _split_stats(train_videos)
    val_stats = _split_stats([v for v in val_videos if v in exported_ids])
    print("\n=== dataset summary ===")
    print(f"train: {train_stats}")
    print(f"val:   {val_stats}")
    total_triplets = train_stats["triplets"] + val_stats["triplets"]
    print(f"total triplets: {total_triplets}  (from {len(exported_ids)} video(s))")

    # Honest small-dataset flag — report, don't block (Stage 3/4 decide what
    # to do with this; matches the "report negative/marginal findings
    # plainly" pattern used throughout this project).
    if total_triplets < 30:
        print("\nFLAG: fewer than 30 total triplets. Fine-tuning is still worth TRYING (low LR, "
              "few epochs, from a pretrained init, not from scratch) but don't expect this alone "
              "to fix a systemic detector gap — treat Stage 4's before/after eval as the real "
              "verdict, not this count.")
    if train_stats["triplets"] < 10:
        print("FLAG: fewer than 10 TRAINING triplets — a very thin fine-tune. Label more clips "
              "before trusting the result.")
    if val_stats["triplets"] < 5:
        print("FLAG: fewer than 5 VAL triplets — val loss will be noisy; treat Stage 4's direct "
              "eval-clip comparison as more trustworthy than the val loss curve alone.")

    print(f"\nwrote {out_dir}/manifest.json and {out_dir}/frames/<video_id>/<frame>.jpg")


if __name__ == "__main__":
    main()
