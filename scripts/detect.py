"""Count and measure oysters in a tray image.

Runs the FastSAM-based pipeline, applies the stored px->mm calibration, and
writes a per-oyster CSV and an annotated overlay image.

Usage:
    python scripts/detect.py --image field-images/20260522_bag380_raw.jpeg
    python scripts/detect.py --image path/to/new.jpg --roi x0 y0 x1 y1
"""

from __future__ import annotations

import argparse
import os

import cv2

import oyster_vision as ov

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_WEIGHTS = os.path.join(ROOT, "models", "FastSAM-s.pt")
DEF_CALIB = os.path.join(ROOT, "outputs", "calibration.json")
OUT = os.path.join(ROOT, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=DEF_WEIGHTS)
    ap.add_argument("--calibration", default=DEF_CALIB)
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="override calibration px/mm")
    ap.add_argument("--roi", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="board bounding box; defaults to the calibration ROI")
    ap.add_argument("--no-roi", action="store_true",
                    help="ignore any board ROI and scan the whole image")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    ov.ensure_weights(args.weights)
    calib = ov.load_calibration(args.calibration) if os.path.exists(args.calibration) else {}
    px_per_mm = args.px_per_mm or calib.get("px_per_mm")
    if px_per_mm is None:
        raise SystemExit("No calibration found; run calibrate.py or pass --px-per-mm")

    roi = None if args.no_roi else (args.roi or calib.get("board_roi_xyxy"))

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    oysters, masks = ov.detect_and_measure(img, args.weights, px_per_mm, roi=roi)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{stem}_predictions.csv")
    ov.write_csv(oysters, csv_path, meta={"image": os.path.basename(args.image)})
    overlay = ov.draw_overlay(img, oysters, masks)
    ovl_path = os.path.join(args.outdir, f"{stem}_overlay.jpg")
    cv2.imwrite(ovl_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])

    lengths = [o.length_mm for o in oysters]
    widths = [o.width_mm for o in oysters]
    print(f"Counted {len(oysters)} oysters (px/mm={px_per_mm:.3f})")
    if oysters:
        print(f"  length mm: mean {sum(lengths)/len(lengths):.1f} "
              f"[{min(lengths):.1f}-{max(lengths):.1f}]")
        print(f"  width  mm: mean {sum(widths)/len(widths):.1f} "
              f"[{min(widths):.1f}-{max(widths):.1f}]")
    print(f"  wrote {csv_path}")
    print(f"  wrote {ovl_path}")


if __name__ == "__main__":
    main()
