"""Video decoding helpers (OpenCV). The original file is the single source of
truth — we never write derived/clipped video, only metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        return VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    step: int = 1,
    scale: float = 1.0,
    with_full: bool = False,
) -> Iterator[tuple]:
    """Yield (frame_index, BGR frame). `step` subsamples temporally (Pass 1),
    `scale` subsamples spatially. Sequential grab/retrieve — no per-frame
    seeking, which is slow and inaccurate on long GOP videos.

    `with_full=True` additionally yields the FULL-resolution frame as a third
    element: (idx, scaled, full). Decode cost is unchanged (retrieve already
    returns full-res); this lets Pass 1 run cheap motion/ball detection on the
    downscaled frame while feeding pose the full-res frame so small far-side
    players are still detectable."""
    cap = cv2.VideoCapture(path)
    try:
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while True:
            if end_frame is not None and idx >= end_frame:
                break
            ok = cap.grab()
            if not ok:
                break
            if (idx - start_frame) % step == 0:
                ok, full = cap.retrieve()
                if not ok:
                    break
                frame = full
                if scale != 1.0:
                    frame = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                yield (idx, frame, full) if with_full else (idx, frame)
            idx += 1
    finally:
        cap.release()
