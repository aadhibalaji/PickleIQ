"""Download official WASB-SBDT TrackNetV2 pretrained weights and install
them where the ball tracker picks them up automatically.

Source: NTT's WASB-SBDT benchmark (github.com/nttcom/WASB-SBDT, MODEL_ZOO.md)
— real TrackNetV2 checkpoints trained on public tennis / badminton ball
datasets. Tennis is the default: its broadcast behind-the-baseline view is
the closest match to this fixed pickleball camera.

Usage:
    .venv\\Scripts\\python backend/scripts/get_tracknet_weights.py [--sport tennis|badminton]

Writes data/models/tracknet_v2.pt. Re-run Pass 2 afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DATA_DIR  # noqa: E402
from app.cv.ball_tracknet import _build_model, load_state  # noqa: E402

GDRIVE_IDS = {
    # From WASB-SBDT MODEL_ZOO.md (TrackNetV2 rows)
    "tennis": "1BCnmvDX-LZpbkk4vlMEXMm-uzCoqJzDx",
    "badminton": "1lCVYzua7jJfuKqvWGypkYqr6PHA1EaPq",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=list(GDRIVE_IDS), default="tennis")
    args = ap.parse_args()

    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed — run: .venv\\Scripts\\pip install gdown")

    out_dir = DATA_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"wasb_tracknetv2_{args.sport}.pth.tar"
    dest = out_dir / "tracknet_v2.pt"

    if not raw.exists():
        print(f"downloading WASB TrackNetV2 ({args.sport}) from Google Drive…")
        gdown.download(id=GDRIVE_IDS[args.sport], output=str(raw), quiet=False)
    else:
        print(f"using cached download {raw}")

    print("verifying the checkpoint loads into our TrackNetV2…")
    import torch

    state = load_state(raw)
    model = _build_model()
    model.load_state_dict(state)  # strict — must match exactly
    model.eval()
    with torch.no_grad():
        y = torch.sigmoid(model(torch.zeros(1, 9, 288, 512)))
    assert y.shape == (1, 3, 288, 512), y.shape
    torch.save(state, dest)
    print(f"OK — installed {dest}")
    print("TrackNet will be used automatically on the next Pass 2 run "
          "(Regenerate video in the Report tab, or POST /reprocess?which=pass2).")


if __name__ == "__main__":
    main()
