"""Count (and, where scale is known, measure) oysters in a tray photo.

Differs from ``detect.py`` in three ways, each fixing a way the original
pipeline failed to transfer beyond its single reference image:

* **no absolute size assumption** - uses ``filter_oysters_adaptive``, which
  derives the size band from the image's own mask population and rejects
  vividly coloured, smooth regions (crate slats, clipboards, tarp, ear tags);
* **no truncation on dense trays** - ``--max-det`` defaults to 900 instead of
  FastSAM's 300, which silently capped dense seed photos;
* **no invented millimetres** - scale is resolved per image via ``scale.py``.
  When it cannot be established, counts are still reported and the mm columns
  are left empty rather than filled from another photo's constant.

Usage:
    python scripts/detect_adaptive.py --image field-images/x.jpeg
    python scripts/detect_adaptive.py --image x.jpeg --px-per-mm 3.10
    python scripts/detect_adaptive.py --image x.jpeg --ref-px 512 --ref-mm 150
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oyster_vision as ov
import scale as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_WEIGHTS = os.path.join(ROOT, "models", "FastSAM-s.pt")
DEF_SCALES = os.path.join(ROOT, "outputs", "scales.json")


def write_csv(oysters, path, image, px_per_mm):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Name", "Oyster", "cx_px", "cy_px", "length_px",
                    "width_px", "length_mm", "width_mm", "area_px"])
        for o in oysters:
            lmm = round(o.length_mm, 2) if px_per_mm else ""
            wmm = round(o.width_mm, 2) if px_per_mm else ""
            w.writerow([image, o.id, round(o.cx, 1), round(o.cy, 1),
                        round(o.length_px, 1), round(o.width_px, 1),
                        lmm, wmm, round(o.area_px, 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=DEF_WEIGHTS)
    ap.add_argument("--scales", default=DEF_SCALES)
    ap.add_argument("--px-per-mm", type=float, default=None)
    ap.add_argument("--ref-px", type=float, default=None,
                    help="a measured distance in the photo, in pixels")
    ap.add_argument("--ref-mm", type=float, default=None,
                    help="what that distance is, in millimetres")
    ap.add_argument("--max-det", type=int, default=900)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-mean-sat", type=float, default=80.0)
    ap.add_argument("--tex-pct", type=float, default=0.0,
                    help="optional texture percentile gate; 0 disables (default)")
    ap.add_argument("--roi", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    ppm, src = sc.resolve(args.image, args.px_per_mm,
                          sc.load_scales(args.scales), args.ref_px, args.ref_mm)

    ov.ensure_weights(args.weights)
    masks = ov.segment_everything(img, args.weights, imgsz=args.imgsz,
                                  max_det=args.max_det)
    if len(masks) >= args.max_det:
        print(f"  WARNING: hit max_det={args.max_det}; count may be truncated. "
              f"Re-run with a higher --max-det.")

    idx, info = ov.filter_oysters_adaptive(
        img, masks, max_mean_sat=args.max_mean_sat, tex_pct=args.tex_pct,
        roi=args.roi,
        return_info=True)
    kept = masks[idx] if len(idx) else masks[:0]

    oysters = []
    for k, m in enumerate(kept, start=1):
        lp, wp, ap_ = ov.measure_mask(m)
        p = ov.shape_props(m)
        oysters.append(ov.Oyster(
            id=k, cx=p["cx"], cy=p["cy"], length_px=lp, width_px=wp,
            length_mm=(lp / ppm) if ppm else 0.0,
            width_mm=(wp / ppm) if ppm else 0.0, area_px=ap_))

    stem = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{stem}_adaptive.csv")
    write_csv(oysters, csv_path, os.path.basename(args.image), ppm)
    ovl_path = os.path.join(args.outdir, f"{stem}_adaptive_overlay.jpg")
    cv2.imwrite(ovl_path, ov.draw_overlay(img, oysters, kept),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"Counted {len(oysters)} oysters")
    print(f"  masks {info['n_masks']} -> shape {info['shape']} -> "
          f"colour {info['colour']} -> texture {info['texture']} -> "
          f"size {info['area']} -> NMS {info['final']}")
    if ppm:
        L = [o.length_mm for o in oysters]
        W = [o.width_mm for o in oysters]
        print(f"  scale {ppm:.3f} px/mm ({src})")
        if L:
            print(f"  length mm: mean {sum(L)/len(L):.1f} [{min(L):.1f}-{max(L):.1f}]")
            print(f"  width  mm: mean {sum(W)/len(W):.1f} [{min(W):.1f}-{max(W):.1f}]")
    else:
        print("  scale UNKNOWN - mm columns left empty. Supply --px-per-mm, "
              "--ref-px/--ref-mm, or add an entry to outputs/scales.json.")
    print(f"  wrote {csv_path}")
    print(f"  wrote {ovl_path}")


if __name__ == "__main__":
    main()
