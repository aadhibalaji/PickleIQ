"""Train the learned rally-end (bounce/hit) temporal model and REGISTER it.

Training uses stored Pass 2 feature traces + human rally labels (weak
supervision) via app.learning.dataset, with the SAME featurizer the model sees
at inference (no train/serve skew). The result is registered as a new 'shadow'
version in the model registry. It is validated (model vs heuristic on the
labeled clips) and promoted to 'live' only with --promote AND once there are
>= MIN_PROMOTE_CLIPS labeled clips — below that a model would just overfit one
clip, so it correctly stays shadow.

Once a version is 'live', Pass 2 loads it automatically via
model_registry.load_live('rally_end') and detect_events uses it (this replaces
the old dead load_bounce_model path). Rollback = promote a prior version.

Usage:
    .venv\\Scripts\\python backend/scripts/train_bounce.py [--epochs 40] [--promote]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cv.bounce import build_bounce_model  # noqa: E402
from app.learning import dataset, model_registry, validate  # noqa: E402

MIN_PROMOTE_CLIPS = 3


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--promote", action="store_true",
                    help="promote to live if validation passes (>= 3 clips)")
    args = ap.parse_args()

    X, Y, meta = dataset.build_end_dataset()
    if meta["n_windows"] < 8:
        print(f"only {meta['n_windows']} training windows across {meta['n_clips']} clip(s) — "
              "label + process more clips first (need human labels AND completed Pass 2).")
        return
    Xp, Yp = dataset.to_padded(X, Y)
    print(f"dataset: {meta['n_windows']} windows x {Xp.shape[1]} frames "
          f"from {meta['n_clips']} clip(s)  (pos={meta['n_pos']} neg={meta['n_neg']})")

    model = build_bounce_model()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor([0.1, 1.0, 0.7])  # class 0 (nothing) is dominant
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=dataset.IGNORE)
    Xt, Yt = torch.from_numpy(Xp), torch.from_numpy(Yp)

    model.train()
    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = model(Xt)  # (B, T, 3)
        loss = loss_fn(logits.reshape(-1, 3), Yt.reshape(-1))
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}")

    version, path = model_registry.register(
        "rally_end", metrics={}, trained_on=meta, origin="manual_cli", status="shadow")
    model.eval()
    torch.save(model.state_dict(), path)
    print(f"registered rally_end v{version} (SHADOW) -> {path}")

    if meta["n_clips"] < MIN_PROMOTE_CLIPS:
        print(f"cold-start guard: need >= {MIN_PROMOTE_CLIPS} labeled clips to validate & "
              f"promote, have {meta['n_clips']}. Model stays SHADOW. (Correct — a model trained "
              "on one clip overfits; collect more labels, then re-run.)")
        return

    print("validating (model vs heuristic, full re-run across labeled clips)…")
    res = validate.evaluate_model("rally_end", model, meta["video_ids"])
    model_registry.set_metrics(version, res)
    be, af = res["before"]["mean_abs_end_err"], res["after"]["mean_abs_end_err"]
    print(f"  end error: heuristic {be} -> model {af}   accept={res['accept']}")
    if res["accept"] and args.promote:
        model_registry.promote(version)
        print(f"PROMOTED rally_end v{version} to LIVE — Pass 2 uses it on the next run.")
    elif res["accept"]:
        print(f"validation PASSED; re-run with --promote to make v{version} live.")
    else:
        print(f"validation did not beat the heuristic — v{version} stays shadow.")


if __name__ == "__main__":
    main()
