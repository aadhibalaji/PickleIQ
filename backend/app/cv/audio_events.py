"""Audio-based paddle/ball hit-onset extraction (Pass-2 confidence input).

Paddle/ball contact makes a short broadband "pop" that is INDEPENDENT of how
far the ball is from the camera — the opposite of the visual detector's biggest
error source (far-court apparent-size->depth noise, DETECTION_AUDIT.md). So a
band-limited onset detector gives cheap extra hit candidates to fuse against the
visual track (see bounce.fuse_audio_events).

This runs once per uploaded video, idempotently, wholly independent of Pass 1 /
Pass 2 (no calibration dependency). It is the productionised form of
backend/scripts/audio_hit_detector.py: same ffmpeg->bandpass->spectral-flux
chain, but band edges / sensitivity come from LearnedParams and results land in
the `audio_events` table instead of a CSV.

Caveat (camera-dependent): on the validated PlaySight clip the two audio
channels are bit-identical (mono duplicated at source) AND several courts are
audible on one track, so cross-court bleed produces false onsets with no local
play. These raw onsets are CANDIDATES; the visual-fusion gate rejects the bleed.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .. import db

log = logging.getLogger("audio_events")

_SR = 48000          # extraction sample rate (matches the source AAC)
_HOP = 256           # librosa onset hop; frames_per_sec = sr/hop


def _ffmpeg_exe() -> str:
    """Resolve an ffmpeg binary. Prefer a system ffmpeg on PATH; else fall back
    to the static one bundled by imageio-ffmpeg (a listed dependency), so audio
    extraction works with no system ffmpeg install."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001 — surface a clear, actionable message
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio-ffmpeg unavailable; "
            "`pip install imageio-ffmpeg` or install ffmpeg to enable audio "
            "onset extraction") from e


def extract_audio(video_path: str, out_wav: str, sr: int = _SR) -> None:
    """Pull mono PCM audio out of a video file via ffmpeg."""
    cmd = [
        _ffmpeg_exe(), "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr[-2000:]}")


def bandpass(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Isolate the band where paddle/ball 'pop' transients concentrate."""
    import scipy.signal as sig
    # Guard against an inverted/degenerate band from clamped params.
    low = max(1.0, min(low, sr / 2 - 1))
    high = max(low + 1.0, min(high, sr / 2 - 1))
    sos = sig.butter(4, [low, high], btype="band", fs=sr, output="sos")
    return sig.sosfilt(sos, y).astype(np.float32)


def detect_onsets(y: np.ndarray, sr: int, delta: float, wait_s: float,
                  hop_length: int = _HOP, pre_avg_s: float = 0.1,
                  post_avg_s: float = 0.1):
    """Spectral-flux onset detection tuned conservative (fewer, higher-confidence
    hits). Returns (onset_times_s, onset_strengths)."""
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    fps = sr / hop_length
    wait = int(round(wait_s * fps))
    pre_avg = int(round(pre_avg_s * fps))
    post_avg = int(round(post_avg_s * fps))

    times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, onset_envelope=onset_env,
        backtrack=False, units="time",
        pre_max=max(pre_avg, 1), post_max=max(post_avg, 1),
        pre_avg=max(pre_avg * 2, 1), post_avg=max(post_avg * 2, 1),
        delta=delta, wait=max(wait, 1),
    )
    frame_idx = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
    strengths = onset_env[np.clip(frame_idx, 0, len(onset_env) - 1)]
    return times, strengths


def compute_and_store(video_id: str, params: dict) -> int:
    """Extract audio onsets for one video and persist to `audio_events`.
    Idempotent: skips if rows already exist for this video_id. Returns the
    number of onsets written (0 if skipped/no audio)."""
    with db.connect() as con:
        if con.execute("SELECT 1 FROM audio_events WHERE video_id=? LIMIT 1",
                       (video_id,)).fetchone():
            log.info("audio_events %s: already computed, skipping", video_id)
            return 0
        vrow = con.execute("SELECT path, fps FROM videos WHERE id=?",
                           (video_id,)).fetchone()
    if vrow is None:
        log.warning("audio_events %s: no video row", video_id)
        return 0
    fps = float(vrow["fps"]) or 30.0

    low = float(params.get("audio_band_low_hz", 1500.0))
    high = float(params.get("audio_band_high_hz", 6000.0))
    delta = float(params.get("audio_onset_delta", 0.4))
    wait_s = float(params.get("audio_onset_wait_s", 0.35))

    import librosa
    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "audio.wav")
        extract_audio(vrow["path"], wav)
        y, sr = librosa.load(wav, sr=None, mono=True)
        yb = bandpass(y, sr, low, high)
        times, strengths = detect_onsets(yb, sr, delta=delta, wait_s=wait_s)

    now = db.now()
    rows = [(db.new_id(), video_id, float(t), int(round(float(t) * fps)),
             float(s), now) for t, s in zip(times, strengths)]
    with db.connect() as con:
        con.executemany(
            "INSERT INTO audio_events "
            "(id, video_id, time_s, frame_index, onset_strength, created_at) "
            "VALUES (?,?,?,?,?,?)", rows)
    log.info("audio_events %s: wrote %d onsets", video_id, len(rows))
    return len(rows)


def load_events(video_id: str, start_frame: int | None = None,
                end_frame: int | None = None) -> list[dict]:
    """Onset rows for a video, optionally clipped to a frame window (inclusive).
    Used by the fusion step to gate onsets to the current Pass 1 rally window."""
    q = "SELECT time_s, frame_index, onset_strength FROM audio_events WHERE video_id=?"
    args: list = [video_id]
    if start_frame is not None:
        q += " AND frame_index >= ?"
        args.append(int(start_frame))
    if end_frame is not None:
        q += " AND frame_index <= ?"
        args.append(int(end_frame))
    q += " ORDER BY frame_index"
    with db.connect() as con:
        return db.rows_to_dicts(con.execute(q, args).fetchall())


def demo() -> None:
    """Self-check: bandpass+onset on a synthetic click train recovers the clicks.
    Runs without ffmpeg/DB. `python -m app.cv.audio_events <video>` runs the real
    extractor on a file instead."""
    import sys
    if len(sys.argv) > 1:
        from ..learning import params as lp
        _, p = lp.get_latest()
        low = float(p["audio_band_low_hz"]); high = float(p["audio_band_high_hz"])
        y, sr = _load_or_die(sys.argv[1])
        times, _ = detect_onsets(bandpass(y, sr, low, high), sr,
                                 delta=p["audio_onset_delta"],
                                 wait_s=p["audio_onset_wait_s"])
        print(f"{len(times)} onsets: {np.round(times, 3)[:20]}")
        return
    sr = 48000
    y = np.zeros(sr * 2, dtype=np.float32)
    click_ts = [0.3, 0.9, 1.5]
    for t in click_ts:                       # short broadband clicks in-band
        i = int(t * sr)
        y[i:i + 8] += np.random.randn(8).astype(np.float32)
    times, strengths = detect_onsets(bandpass(y, sr, 1500, 6000), sr,
                                     delta=0.2, wait_s=0.2)
    assert len(times) >= len(click_ts), f"missed clicks: {times}"
    for ct in click_ts:
        assert min(abs(times - ct)) < 0.05, f"no onset near {ct}: {times}"
    assert (strengths > 0).all()
    print("audio_events demo OK:", np.round(times, 3))


def _load_or_die(video: str):
    import librosa
    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "a.wav")
        extract_audio(video, wav)
        return librosa.load(wav, sr=None, mono=True)


if __name__ == "__main__":
    demo()
