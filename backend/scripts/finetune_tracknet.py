"""Fine-tune TrackNetV2 from the pretrained WASB-SBDT checkpoint on the
Stage 2 dataset (scripts/export_tracknet_dataset.py).

Design choices (see Stage 1 investigation for the reasoning):

* FULL fine-tune, not frozen-backbone+head. The failure this is meant to fix
  (weak confidence at serve contact / far-baseline) is a visual-feature
  problem spanning the whole network, not a final-layer classification
  problem — freezing everything but the 1x1 outc conv would barely move it.
  Low LR + few epochs is the regularizer for the small dataset instead of
  freezing (--freeze-encoder is available if you want to compare, but is NOT
  the default).
* LR default 1e-5 — a full-network fine-tune from a working pretrained init
  needs to be an order of magnitude (or two) below a from-scratch LR, or the
  first few steps wreck the pretrained prior before the small dataset can
  teach it anything useful.
* Loss: BCEWithLogitsLoss against a Gaussian heatmap target, matching the
  model's own sigmoid read-out at inference (ball_tracknet.py applies
  torch.sigmoid to these exact logits) — an MSE-on-probabilities loss would
  optimize a different quantity than what inference actually thresholds.
  `pos_weight` counters the extreme background/foreground pixel imbalance
  (a few hundred "ball" pixels out of 147,456 per heatmap).
* Per-sample, only the triplet POSITIONS that carry a real label contribute
  to the loss for that sample (see Stage 2's manifest: unlabeled neighbours
  are `null`) — never penalized as "should be background everywhere else",
  since we don't actually know where the ball was in an unlabeled frame.

Input/output preprocessing below is copy-identical to
app.cv.ball_tracknet.TrackNetBall.infer_triplet (same cv2.resize call, same
BGR->RGB, same /255.0, same channel stacking) — any drift between the two
would silently retrain the model to expect something inference never
produces.

Usage:
    .venv/bin/python backend/scripts/finetune_tracknet.py
        [--epochs 6] [--lr 1e-5] [--batch-size 8] [--sigma 3.0]
        [--dataset PATH] [--pretrained PATH] [--out PATH]
        [--limit-train N] [--limit-val N]   (smoke-test a few samples first)
        [--freeze-encoder]                  (off by default, see above)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DATA_DIR  # noqa: E402
from app.cv.ball_tracknet import (INPUT_H, INPUT_W, _build_model,  # noqa: E402
                                  load_state, weights_path)
from app.cv.device import torch_device  # noqa: E402


def _load_input(frame_dir: Path, frame_indices: list[int]) -> np.ndarray:
    """(9, INPUT_H, INPUT_W) float32 in [0,1] — byte-identical preprocessing
    to TrackNetBall.infer_triplet, just reading from saved JPEGs instead of
    live frames."""
    stack = []
    for idx in frame_indices:
        img = cv2.imread(str(frame_dir / f"{idx:06d}.jpg"))
        if img is None:
            raise FileNotFoundError(frame_dir / f"{idx:06d}.jpg")
        stack.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # already INPUT_W x INPUT_H
    x = np.concatenate(stack, axis=2).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _gaussian_heatmap(cx: float, cy: float, sigma: float) -> np.ndarray:
    """(INPUT_H, INPUT_W) float32, peak 1.0 at (cx, cy). sigma=3.0px default:
    the far-baseline ball is only a few pixels wide even at native resolution
    (per ball_tracknet.py's own docstring) and INPUT_W/INPUT_H is already a
    ~3.75x downscale from 1920x1080, so a tight peak is appropriate — a wide
    sigma would teach the model a fuzzier, less precise localization than the
    ball actually has."""
    yy, xx = np.mgrid[0:INPUT_H, 0:INPUT_W]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)


def _build_target(sample: dict, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (target (3,H,W) float32, supervised_mask (3,) bool)."""
    target = np.zeros((3, INPUT_H, INPUT_W), dtype=np.float32)
    mask = np.zeros(3, dtype=bool)
    ow, oh = sample["orig_width"], sample["orig_height"]
    for i, lb in enumerate(sample["labels_px"]):
        if lb is None:
            continue
        x, y = lb
        cx, cy = x * INPUT_W / ow, y * INPUT_H / oh
        target[i] = _gaussian_heatmap(cx, cy, sigma)
        mask[i] = True
    return target, mask


class TrackNetDataset:
    def __init__(self, dataset_dir: Path, video_ids: list[str], sigma: float, limit: int | None = None):
        manifest = json.loads((dataset_dir / "manifest.json").read_text())
        self.samples = [s for s in manifest["samples"] if s["video_id"] in video_ids]
        if limit:
            self.samples = self.samples[:limit]
        self.frames_root = dataset_dir / "frames"
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        x = _load_input(self.frames_root / s["video_id"], s["frame_indices"])
        y, mask = _build_target(s, self.sigma)
        return x, y, mask


def _iterate_batches(ds: TrackNetDataset, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(len(ds)) if shuffle else np.arange(len(ds))
    for start in range(0, len(ds), batch_size):
        idxs = order[start:start + batch_size]
        xs, ys, ms = zip(*(ds[i] for i in idxs))
        yield np.stack(xs), np.stack(ys), np.stack(ms)


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Stage 2 output dir (default DATA_DIR/tracknet_dataset)")
    ap.add_argument("--pretrained", default=None, help="checkpoint to fine-tune FROM (default: whatever weights_path() resolves)")
    ap.add_argument("--out", default=None, help="output checkpoint (default DATA_DIR/models/tracknet_v2_finetuned.pt)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--pos-weight", type=float, default=100.0,
                    help="BCE pos_weight countering background/foreground pixel imbalance")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="freeze inc/down1-3, fine-tune only up1-3+outc (NOT the default — see module docstring)")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke-test: cap train samples")
    ap.add_argument("--limit-val", type=int, default=None, help="smoke-test: cap val samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else DATA_DIR / "tracknet_dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    train_ds = TrackNetDataset(dataset_dir, manifest["train_videos"], args.sigma, args.limit_train)
    val_ds = TrackNetDataset(dataset_dir, manifest["val_videos"], args.sigma, args.limit_val)
    if len(train_ds) == 0:
        print("no training samples (empty train split or --limit-train 0) — stopping.")
        return
    print(f"train: {len(train_ds)} samples ({manifest['train_videos']})")
    print(f"val:   {len(val_ds)} samples ({manifest['val_videos']})")

    pretrained = Path(args.pretrained) if args.pretrained else weights_path()
    if pretrained is None or not Path(pretrained).exists():
        print("no pretrained checkpoint found — run scripts/get_tracknet_weights.py first "
              "(fine-tuning needs a real initialization, not random weights).")
        return
    print(f"initializing from {pretrained}")

    device = torch_device()
    print(f"device: {device}")
    model = _build_model()
    model.load_state_dict(load_state(Path(pretrained)))
    try:
        model.to(device)
    except Exception:  # noqa: BLE001 — mirrors ball_tracknet.py's own fallback
        print(f"model.to({device!r}) failed, falling back to CPU")
        device = "cpu"
        model.to("cpu")

    if args.freeze_encoder:
        for name, p in model.named_parameters():
            if name.split(".")[0] in ("inc", "down1", "down2", "down3"):
                p.requires_grad = False
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"--freeze-encoder: {n_frozen} parameter tensors frozen (inc/down1-3)")

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    pos_weight = torch.full((1, INPUT_H, INPUT_W), args.pos_weight, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    rng = np.random.default_rng(args.seed)

    def _masked_loss(logits: "torch.Tensor", y: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        per_px = loss_fn(logits, y)              # (B,3,H,W)
        per_channel = per_px.mean(dim=(2, 3))     # (B,3)
        m = mask.float()
        denom = m.sum()
        if denom.item() == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        return (per_channel * m).sum() / denom

    def _run_epoch(ds: TrackNetDataset, train: bool) -> float:
        model.train(train)
        total, n_batches = 0.0, 0
        for xb, yb, mb in _iterate_batches(ds, args.batch_size, shuffle=train, rng=rng):
            xt = torch.from_numpy(xb).to(device)
            yt = torch.from_numpy(yb).to(device)
            mt = torch.from_numpy(mb).to(device)
            if train:
                opt.zero_grad()
                logits = model(xt)
                loss = _masked_loss(logits, yt, mt)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    logits = model(xt)
                    loss = _masked_loss(logits, yt, mt)
            total += float(loss.item())
            n_batches += 1
        return total / max(1, n_batches)

    print(f"\nfine-tuning: {args.epochs} epoch(s), lr={args.lr}, batch_size={args.batch_size}, "
          f"sigma={args.sigma}px, pos_weight={args.pos_weight}")
    print("(CPU-bound — this will be slow; watch the per-epoch wall time below and Ctrl-C if "
          "it's clearly going to run too long, nothing is lost until the final save)\n")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = _run_epoch(train_ds, train=True)
        val_loss = _run_epoch(val_ds, train=False) if len(val_ds) else float("nan")
        dt = time.time() - t0
        print(f"  epoch {epoch + 1:2d}/{args.epochs}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  ({dt:.1f}s)")

    out_path = Path(args.out) if args.out else DATA_DIR / "models" / "tracknet_v2_finetuned.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print(f"\nsaved fine-tuned checkpoint -> {out_path}")
    print("This is a SEPARATE file from the pretrained checkpoint — nothing production-facing "
          "changed. Next: Stage 4 compares this against the pretrained weights on real serve "
          "contacts before anything gets promoted or pushed to S3.")


if __name__ == "__main__":
    main()
