#!/usr/bin/env python3
"""
Audio-based pickleball hit/bounce onset detector — standalone prototype.

Context
-------
CV-Training-Tool currently locates ball contacts purely visually (bounce.py /
ball_filter.py), and the biggest noise source is far-court size->depth error
(see DETECTION_AUDIT.md). Paddle/ball contact makes a short, fairly loud
broadband "pop" that is independent of how far the ball is from the camera,
so it's a candidate extra signal — either to confirm/reject visual hit
candidates, or to help gate BounceGRU / rally_logic in future.

What this script does
----------------------
1. Extracts the audio track from a video file (ffmpeg).
2. Bandpass-filters to the ~1.5-6 kHz range where paddle "pop" transients
   concentrate (vs. crowd/voice/low rumble).
3. Runs spectral-flux onset detection (librosa) to get candidate hit
   timestamps + a confidence-ish onset-strength value.
4. Writes candidates to CSV (frame-aligned to a given fps for direct join
   with your overlays table) and optionally dumps debug audio frames.

Validated informally against "Second 3 mins.mp4": manual spot-check of a few
onsets showed at least one onset landing within ~1 video frame of a visible
paddle-contact / ball-in-flight moment. Also found at least one likely false
positive with no play happening in-frame on this court — consistent with the
expected multi-court audio bleed risk (this venue has several adjacent
courts audible on the same track). Treat the raw onsets as CANDIDATES that
need a secondary gate (e.g. only accept onsets inside Pass 1 rally windows,
and/or ones near a visually plausible ball trajectory) before trusting them
as hit events.

Usage
-----
    python audio_hit_detector.py input.mp4 --out onsets.csv --fps 30

Dependencies: ffmpeg on PATH, librosa, soundfile, numpy, scipy.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def extract_audio(video_path: str, out_wav: str, sr: int = 48000) -> None:
    """Pull mono PCM audio out of a video file via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr[-2000:]}")


def bandpass(y: np.ndarray, sr: int, low: float = 1500, high: float = 6000) -> np.ndarray:
    """Isolate the frequency band where paddle/ball 'pop' transients concentrate."""
    import scipy.signal as sig

    sos = sig.butter(4, [low, high], btype="band", fs=sr, output="sos")
    return sig.sosfilt(sos, y).astype(np.float32)


def detect_onsets(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    delta: float = 0.4,
    wait_s: float = 0.35,
    pre_avg_s: float = 0.1,
    post_avg_s: float = 0.1,
):
    """
    Spectral-flux onset detection tuned to be conservative (fewer, higher-
    confidence hits) rather than exhaustive. Tighten/loosen `delta` and
    `wait_s` to trade precision vs. recall.
    """
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    frames_per_sec = sr / hop_length
    wait = int(round(wait_s * frames_per_sec))
    pre_avg = int(round(pre_avg_s * frames_per_sec))
    post_avg = int(round(post_avg_s * frames_per_sec))

    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, onset_envelope=onset_env,
        backtrack=False, units="time",
        pre_max=max(pre_avg, 1), post_max=max(post_avg, 1),
        pre_avg=max(pre_avg * 2, 1), post_avg=max(post_avg * 2, 1),
        delta=delta, wait=max(wait, 1),
    )

    # Attach the onset-strength value at each detected time as a rough
    # confidence proxy (louder/sharper transient -> higher value).
    onset_frame_idx = librosa.time_to_frames(onset_times, sr=sr, hop_length=hop_length)
    strengths = onset_env[np.clip(onset_frame_idx, 0, len(onset_env) - 1)]

    return onset_times, strengths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--out", default="onsets.csv", help="Output CSV path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video fps, for a frame_index column")
    parser.add_argument("--delta", type=float, default=0.4, help="Onset sensitivity threshold (lower = more onsets)")
    parser.add_argument("--wait", type=float, default=0.35, help="Min seconds between accepted onsets")
    parser.add_argument("--keep-wav", action="store_true", help="Keep the extracted audio wav next to the output")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    wav_path = str(Path(args.out).with_suffix(".wav")) if args.keep_wav else None
    with tempfile.TemporaryDirectory() as tmp:
        wav = wav_path or str(Path(tmp) / "audio.wav")
        print(f"Extracting audio -> {wav}")
        extract_audio(str(video_path), wav)

        import librosa
        y, sr = librosa.load(wav, sr=None, mono=True)
        print(f"Loaded audio: {len(y) / sr:.2f}s @ {sr} Hz")

        yb = bandpass(y, sr)
        times, strengths = detect_onsets(yb, sr, delta=args.delta, wait_s=args.wait)

    print(f"Detected {len(times)} candidate hit onsets")

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "frame_index", "onset_strength"])
        for t, s in zip(times, strengths):
            writer.writerow([f"{t:.4f}", int(round(t * args.fps)), f"{s:.4f}"])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
