"""Build the model-ready SERVE dataset from accumulated human rally marks.

Every human rally-START is a serve-contact timestamp, so this emits — per clip
that has human labels — three files under ``data/refine/labels/``:

* ``<id>.serves.json``       per-instant serve-events sidecar (+ ±buffer);
* ``<id>.rallies.json``      rally intervals in labels.load_label_file shape;
* ``serve_gate/<id>.json``   span-level serve/non-serve for the gate validator.

The buffer around each serve is excluded from the negative set (config
``serve_buffer_s``); everything else outside all rallies is an implicit
negative. Ball-tracking labels are never touched. See app/refine/serve_labels.py.

Usage::

    backend/.venv/Scripts/python backend/scripts/build_serve_dataset.py            # all labeled clips
    ...build_serve_dataset.py --clip "Last 4 mins.mp4"                             # one clip (id or filename)
    ...build_serve_dataset.py --buffer 0.75 --dry-run                              # preview, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.refine.config import RefineConfig            # noqa: E402
from app.refine.io_loaders import clips_with_human_labels  # noqa: E402
from app.refine.serve_labels import build_dataset     # noqa: E402


def resolve_clips(clip_arg: str | None) -> list[dict]:
    clips = clips_with_human_labels()
    if not clip_arg:
        return clips
    sel = [c for c in clips if clip_arg in (c["id"], c["filename"])]
    if not sel:
        raise SystemExit(f"no labeled clip matches {clip_arg!r}; "
                         f"labeled: {[c['filename'] for c in clips]}")
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description="Build serve datasets from human marks")
    ap.add_argument("--clip", default=None,
                    help="video id or filename; default = every labeled clip")
    ap.add_argument("--buffer", type=float, default=None,
                    help="override serve_buffer_s (seconds)")
    ap.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = ap.parse_args()

    cfg = RefineConfig()
    if args.buffer is not None:
        cfg = cfg.copy(serve_buffer_s=args.buffer)

    clips = resolve_clips(args.clip)
    tag = " (dry run)" if args.dry_run else ""
    print(f"{len(clips)} labeled clip(s); buffer=±{cfg.serve_buffer_s}s{tag}")
    for c in clips:
        r = build_dataset(c["id"], cfg, write=not args.dry_run)
        where = "" if args.dry_run else f"  -> {Path(r['written']['serves']).parent}"
        print(f"  {c['filename'][:28]:28}  {r['n_serves']:2d} serves  "
              f"{r['n_rallies']:2d} rallies  {r['n_gate_negatives']:2d} negs{where}")
    if args.dry_run:
        print("dry run — no files written")


if __name__ == "__main__":
    main()
