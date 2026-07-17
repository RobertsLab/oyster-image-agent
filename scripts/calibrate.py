"""Derive the pixel->mm calibration and validate the pipeline.

Uses the ImageJ-annotated reference image (POINT overlay = one crosshair per
oyster placed by a human) together with the human length/width measurements to:

  * run the detector,
  * match detections to the human-placed points,
  * fit a single px-per-mm scale (least squares through the origin over the
    pooled length and width measurements),
  * report detection precision/recall and measurement error.

Outputs ``outputs/calibration.json`` and ``outputs/metrics.json``.

Usage:
    python scripts/calibrate.py
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

import oyster_vision as ov

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_TIF = os.path.join(ROOT, "field-images", "20260522_bag380_annotated_imageJ.tif")
DEF_RAW = os.path.join(ROOT, "field-images", "20260522_bag380_raw.jpeg")
DEF_CSV = os.path.join(ROOT, "outputs", "20260522_bag380_data.csv")
DEF_WEIGHTS = os.path.join(ROOT, "models", "FastSAM-s.pt")
OUT = os.path.join(ROOT, "outputs")


def read_imagej_points(tif_path):
    """Return (N,2) array of human oyster-centroid points, ordered 1..N."""
    import roifile
    import tifffile
    t = tifffile.TiffFile(tif_path)
    roi = roifile.ImagejRoi.frombytes(t.imagej_metadata["ROI"])
    pts = roi.subpixel_coordinates
    if pts is None:
        pts = roi.integer_coordinates + np.array([roi.left, roi.top])
    return np.asarray(pts, float)


def greedy_match(det_xy, gt_xy, max_dist):
    """Greedy nearest-neighbour one-to-one matching. Returns list of (di, gi)."""
    D = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    used_d, used_g, pairs = set(), set(), []
    for di, gi in order:
        if D[di, gi] >= max_dist:
            break
        if di in used_d or gi in used_g:
            continue
        used_d.add(di); used_g.add(gi); pairs.append((int(di), int(gi)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", default=DEF_TIF)
    ap.add_argument("--raw", default=DEF_RAW)
    ap.add_argument("--csv", default=DEF_CSV)
    ap.add_argument("--weights", default=DEF_WEIGHTS)
    ap.add_argument("--match-dist", type=float, default=90.0,
                    help="max centroid distance (px) for a detection<->point match")
    args = ap.parse_args()

    ov.ensure_weights(args.weights)
    img = cv2.imread(args.raw)
    H, W = img.shape[:2]

    gt = read_imagej_points(args.tif)
    # board ROI = human-point extent + margin (keeps FastSAM off feet/grating/tools)
    margin = 180
    x0, y0 = gt.min(0) - margin
    x1, y1 = gt.max(0) + margin
    roi = [max(0, x0), max(0, y0), min(W, x1), min(H, y1)]

    oysters, masks = ov.detect_and_measure(img, args.weights, px_per_mm=1.0, roi=roi)
    det_xy = np.array([[o.cx, o.cy] for o in oysters])
    print(f"Detected {len(oysters)} oyster instances; {len(gt)} human points.")

    pairs = greedy_match(det_xy, gt, args.match_dist)
    TP = len(pairs); FP = len(oysters) - TP; FN = len(gt) - TP
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * TP / (2 * TP + FP + FN) if TP else 0

    piv = (pd.read_csv(args.csv)
           .pivot_table(index="Oyster", columns="Measurement", values="Value mm"))
    di = np.array([p[0] for p in pairs]); gi = np.array([p[1] for p in pairs])
    px_L = np.array([oysters[i].length_px for i in di])
    px_W = np.array([oysters[i].width_px for i in di])
    mm_L = piv.loc[gi + 1, "length"].values
    mm_W = piv.loc[gi + 1, "width"].values

    # single scale (px per mm) via least squares through origin over pooled L+W
    px_all = np.r_[px_L, px_W]; mm_all = np.r_[mm_L, mm_W]
    px_per_mm = float(np.sum(px_all * mm_all) / np.sum(mm_all ** 2))

    def err(px, mm):
        pred = px / px_per_mm
        return {
            "corr": float(np.corrcoef(px, mm)[0, 1]),
            "MAE_mm": float(np.abs(pred - mm).mean()),
            "bias_mm": float((pred - mm).mean()),
            "RMSE_mm": float(np.sqrt(((pred - mm) ** 2).mean())),
            "MAPE_pct": float((np.abs(pred - mm) / mm).mean() * 100),
        }

    calib = {
        "px_per_mm": px_per_mm,
        "mm_per_px": 1.0 / px_per_mm,
        "board_roi_xyxy": [float(v) for v in roi],
        "source_image": os.path.basename(args.raw),
        "n_calibration_oysters": int(TP),
        "method": "least-squares through origin over pooled length+width vs human mm",
    }
    metrics = {
        "detection": {"n_detected": len(oysters), "n_truth": len(gt),
                      "TP": TP, "FP": FP, "FN": FN,
                      "precision": round(precision, 3), "recall": round(recall, 3),
                      "f1": round(f1, 3)},
        "length_mm": {k: round(v, 2) for k, v in err(px_L, mm_L).items()},
        "width_mm": {k: round(v, 2) for k, v in err(px_W, mm_W).items()},
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"calibration": calib, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
