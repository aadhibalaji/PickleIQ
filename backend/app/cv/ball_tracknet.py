"""TrackNetV2 heatmap ball tracker — the real per-frame model for Pass 2
windows.

The architecture below byte-for-byte mirrors the TrackNetV2 implementation in
NTT's WASB-SBDT benchmark (github.com/nttcom/WASB-SBDT, src/models/unet2d.py:
VGG-style U-Net, 3 RGB frames in -> 3 heatmaps out, conv->ReLU->BN order,
nearest-neighbour upsampling), so their released pretrained checkpoints load
directly:

    backend/scripts/get_tracknet_weights.py --sport tennis   (or badminton)

downloads the official WASB TrackNetV2 checkpoint into
data/models/tracknet_v2.pt, which is picked up automatically (or point
TRACKNET_WEIGHTS in .env somewhere else).

TODO(fine-tune): tennis/badminton weights transfer the "small fast ball"
prior but this fixed down-the-baseline pickleball view still needs
fine-tuning on your own clips — especially the far-baseline region where the
ball is a few pixels wide. The labeling UI's ball spot-corrections and the
stored FeatureTraces are the training data source.

Until weights exist, Pass 2 falls back to the classical detector in
ball_coarse.py (a real method, honestly weaker) and logs a warning.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, TRACKNET_ENABLED, TRACKNET_WEIGHTS, HW_PRECISION, HW_OPENVINO_ENABLED
from .device import torch_device

log = logging.getLogger("cv.tracknet")

INPUT_W, INPUT_H = 512, 288
DEFAULT_WEIGHTS = DATA_DIR / "models" / "tracknet_v2.pt"


def weights_path() -> Path | None:
    if TRACKNET_WEIGHTS:                 # explicit override always wins
        p = Path(TRACKNET_WEIGHTS)
        return p if p.exists() else None
    if not TRACKNET_ENABLED:             # installed default gated by the flag
        return None
    return DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else None


def _build_model():
    import torch
    import torch.nn as nn

    def convs(cin, cout, n):
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1),
                       nn.ReLU(inplace=True), nn.BatchNorm2d(cout)]
        return nn.Sequential(*layers)

    class _Block(nn.Module):
        """Wraps the conv stack under the attribute name WASB uses so
        state_dict keys line up (double_conv / triple_conv)."""

        def __init__(self, cin, cout, n):
            super().__init__()
            name = "double_conv" if n == 2 else "triple_conv"
            setattr(self, name, convs(cin, cout, n))
            self._name = name

        def forward(self, x):
            return getattr(self, self._name)(x)

    class _Down(nn.Module):
        def __init__(self, cin, cout, n):
            super().__init__()
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), _Block(cin, cout, n))

        def forward(self, x):
            return self.maxpool_conv(x)

    class _Up(nn.Module):
        def __init__(self, cin, cout, n):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv = _Block(cin, cout, n)

        def forward(self, x1, x2):
            return self.conv(torch.cat([x2, self.up(x1)], dim=1))

    class _OutConv(nn.Module):
        """1x1 output conv under the attribute name WASB uses (outc.conv.*),
        so the released checkpoints load without key remapping."""

        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1)

        def forward(self, x):
            return self.conv(x)

    class TrackNetV2(nn.Module):
        def __init__(self, n_channels: int = 9, n_classes: int = 3):
            super().__init__()
            self.inc = _Block(n_channels, 64, 2)
            self.down1 = _Down(64, 128, 2)
            self.down2 = _Down(128, 256, 3)
            self.down3 = _Down(256, 512, 3)
            self.up1 = _Up(512 + 256, 256, 3)
            self.up2 = _Up(256 + 128, 128, 2)
            self.up3 = _Up(128 + 64, 64, 2)
            self.outc = _OutConv(64, n_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            return self.outc(x)  # logits, (B, 3, H, W) — one heatmap per frame

    return TrackNetV2()


def load_state(path: Path):
    """Load a checkpoint and normalise its state dict (WASB stores
    {'model_state_dict': ...}; also strip any DataParallel 'module.')."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(ck, dict) and key in ck:
            state = ck[key]
            break
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }


def _openvino_infer(weights: Path):
    """Convert the torch model to OpenVINO FP16 (once, IR cached next to the
    weights) and compile it for the Intel Arc GPU — same playbook as pose
    (players._openvino_pose_dir): torch can't drive the Arc from ultralytics
    and eager torch-XPU runs this U-Net at ~110 ms/frame; OpenVINO is the
    production path. Returns infer(x: np.ndarray (1,9,H,W)) -> logits
    (3,H,W), or None when OpenVINO/GPU is unavailable or the active hardware
    profile turns OpenVINO off (non-Intel machines)."""
    if not HW_OPENVINO_ENABLED:
        return None
    try:
        import openvino as ov

        core = ov.Core()
        if not any(d.startswith("GPU") for d in core.available_devices):
            log.info("OpenVINO present but no GPU device — torch path")
            return None
        ir = weights.parent / "tracknet_ov" / (weights.stem.split(".")[0] + ".xml")
        if not ir.exists():
            import torch

            model = _build_model()
            model.load_state_dict(load_state(weights))
            model.eval()
            ir.parent.mkdir(parents=True, exist_ok=True)
            ovm = ov.convert_model(
                model, example_input=torch.zeros(1, 9, INPUT_H, INPUT_W))
            ov.save_model(ovm, str(ir))  # FP16 compression by default
        compiled = core.compile_model(str(ir), "GPU")
        out_port = compiled.output(0)

        def infer(x: np.ndarray) -> np.ndarray:
            return compiled({0: x})[out_port][0]

        log.info("TrackNet on OpenVINO GPU (%s)", ir.name)
        return infer
    except Exception as exc:  # noqa: BLE001
        log.warning("OpenVINO TrackNet unavailable (%s) — torch path", exc)
        return None


class TrackNetBall:
    """Sliding 3-frame inference. Returns (x, y, diameter_px, confidence) in
    ORIGINAL frame pixels, or None per frame when below threshold.

    `weights` overrides the configured path (used by the eval harness to test
    a checkpoint regardless of TRACKNET_ENABLED/TRACKNET_WEIGHTS)."""

    def __init__(self, conf_threshold: float = 0.5, weights: Path | str | None = None):
        self.conf_threshold = conf_threshold
        self.model = None
        self._ov = None
        self.last_peak: tuple[float, float, float] | None = None  # (x, y, conf) raw argmax
        path = Path(weights) if weights else weights_path()
        self.available = path is not None and Path(path).exists()
        if not self.available:
            log.warning(
                "no TrackNet weights (run backend/scripts/get_tracknet_weights.py "
                "or set TRACKNET_WEIGHTS) — Pass 2 uses the classical fallback "
                "ball detector")
            return
        self._ov = _openvino_infer(Path(path))
        self._half = False
        if self._ov is None:
            import torch

            self.device = torch_device()
            self.model = _build_model()
            self.model.load_state_dict(load_state(path))
            self.model.eval()
            try:
                self.model.to(self.device)
            except Exception:  # noqa: BLE001 — XPU op gaps etc.
                log.warning("TrackNet failed to move to %s, using CPU", self.device)
                self.device = "cpu"
                self.model.to("cpu")
            # fp16 only makes sense on an actual GPU backend; CPU fp16 is
            # unsupported/slow for most torch ops.
            if HW_PRECISION == "fp16" and self.device != "cpu":
                try:
                    self.model.half()
                    self._half = True
                except Exception:  # noqa: BLE001
                    log.warning("TrackNet .half() unsupported on %s, using fp32", self.device)
        log.info("TrackNet weights loaded from %s (%s%s)", path,
                 "openvino-gpu" if self._ov else f"torch-{self.device}",
                 "-fp16" if self._half else "")

    def infer_triplet(self, frames_bgr: list[np.ndarray]) -> tuple[float, float, float, float] | None:
        """frames_bgr: 3 consecutive frames, OLDEST FIRST; the prediction is
        read from the last output heatmap (the newest frame)."""
        import cv2

        assert self.available and len(frames_bgr) == 3
        h0, w0 = frames_bgr[0].shape[:2]
        stack = []
        for f in frames_bgr:
            r = cv2.resize(f, (INPUT_W, INPUT_H))
            stack.append(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        x = np.concatenate(stack, axis=2).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
        if self._ov is not None:
            logits = self._ov(x)[-1]
            hm = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        else:
            import torch

            with torch.no_grad():
                xt = torch.from_numpy(x).to(self.device)
                if self._half:
                    xt = xt.half()
                hm = torch.sigmoid(self.model(xt))[0, -1].float().cpu().numpy()
        conf = float(hm.max())
        yy0, xx0 = np.unravel_index(int(hm.argmax()), hm.shape)
        # Raw peak regardless of threshold — diagnostics read this to tell a
        # true detector miss from a sub-threshold detection.
        self.last_peak = (float(xx0) * w0 / INPUT_W, float(yy0) * h0 / INPUT_H, conf)
        if conf < self.conf_threshold:
            return None
        yy, xx = np.unravel_index(int(hm.argmax()), hm.shape)
        # Blob extent at half-max approximates apparent diameter.
        blob = (hm > conf * 0.5).astype(np.uint8)
        n, labels = cv2.connectedComponents(blob)
        area = float((labels == labels[yy, xx]).sum()) if n > 1 else 1.0
        diam_hm = 2.0 * float(np.sqrt(area / np.pi))
        sx, sy = w0 / INPUT_W, h0 / INPUT_H
        return (float(xx) * sx, float(yy) * sy, diam_hm * (sx + sy) / 2.0, conf)
