"""Oyster counting and measurement from field tray images.

Core library used by the CLI scripts in this directory. The pipeline is:

1. Instance segmentation with FastSAM (a pretrained "segment everything" model).
   No task-specific training is required, which suits a dataset of a single
   labelled image.
2. Filter the raw masks down to oyster instances using size, shape and
   (optionally) a board region-of-interest, then de-duplicate with mask-IoU NMS.
3. Measure each oyster: length = maximum Feret (caliper) diameter of the mask,
   width = short side of the minimum-area rectangle. Pixels are converted to
   millimetres with a single calibration constant (px per mm).

The calibration constant and default board ROI live in
``outputs/calibration.json`` and are produced by ``calibrate.py`` from the
ImageJ-annotated reference image.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

FASTSAM_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"


# --------------------------------------------------------------------------- #
# Model + segmentation
# --------------------------------------------------------------------------- #
def ensure_weights(path: str) -> str:
    """Download FastSAM-s weights to ``path`` if not already present."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading FastSAM weights -> {path}")
        urllib.request.urlretrieve(FASTSAM_URL, path)
    return path


def segment_everything(image_bgr, weights, imgsz=1024, conf=0.4, iou=0.9,
                       device="cpu"):
    """Run FastSAM in 'segment everything' mode. Returns an (N, H, W) uint8 array."""
    from ultralytics import FastSAM

    model = FastSAM(weights)
    res = model(image_bgr, device=device, retina_masks=True, imgsz=imgsz,
                conf=conf, iou=iou, verbose=False)[0]
    if res.masks is None:
        return np.zeros((0,) + image_bgr.shape[:2], np.uint8)
    return res.masks.data.cpu().numpy().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@dataclass
class Oyster:
    id: int
    cx: float          # centroid x (full-image pixels)
    cy: float          # centroid y
    length_px: float
    width_px: float
    length_mm: float
    width_mm: float
    area_px: float


def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def measure_mask(mask):
    """Return (length_px, width_px, area_px) for a single binary mask.

    length = maximum Feret diameter (largest distance across the convex hull),
    which matches the hinge-to-bill caliper measurement a person records.
    width  = short side of the minimum-area bounding rectangle.
    """
    c = _largest_contour(mask)
    if c is None:
        return None
    area = float(cv2.contourArea(c))
    hull = cv2.convexHull(c).reshape(-1, 2).astype(float)
    d = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)
    feret = float(d.max())
    (_, _), (w, h), _ = cv2.minAreaRect(c.astype(np.int32))
    width = float(min(w, h))
    return feret, width, area


def shape_props(mask):
    """Geometric properties used by the oyster filter."""
    c = _largest_contour(mask)
    if c is None:
        return None
    area = float(cv2.contourArea(c))
    if area < 1.0:
        return None
    hull_area = float(cv2.contourArea(cv2.convexHull(c)))
    (cx, cy), (w, h), _ = cv2.minAreaRect(c.astype(np.int32))
    L, W = max(w, h), min(w, h)
    return {
        "area": area,
        "cx": float(cx), "cy": float(cy),
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "extent": area / (L * W) if L * W > 0 else 0.0,
        "aspect": L / W if W > 0 else 99.0,
    }


# --------------------------------------------------------------------------- #
# Filtering + de-duplication
# --------------------------------------------------------------------------- #
def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def filter_oysters(masks, min_area=4000, max_area=45000, min_solidity=0.85,
                   min_extent=0.60, aspect_range=(1.05, 3.2), roi=None,
                   nms_iou=0.30):
    """Filter 'segment everything' masks down to oyster instances.

    ``roi`` is an optional (x0, y0, x1, y1) board bounding box in the same
    coordinate frame as ``masks``; detections whose centroid falls outside it
    are dropped. Returns indices into ``masks`` (sorted, largest first).
    """
    props = [shape_props(m) for m in masks]
    keep = []
    for i, p in enumerate(props):
        if p is None:
            continue
        if not (min_area < p["area"] < max_area):
            continue
        if p["solidity"] < min_solidity or p["extent"] < min_extent:
            continue
        if not (aspect_range[0] < p["aspect"] < aspect_range[1]):
            continue
        if roi is not None:
            x0, y0, x1, y1 = roi
            if not (x0 <= p["cx"] <= x1 and y0 <= p["cy"] <= y1):
                continue
        keep.append(i)

    keep.sort(key=lambda i: -props[i]["area"])
    final = []
    for i in keep:
        if all(_mask_iou(masks[i], masks[j]) < nms_iou for j in final):
            final.append(i)
    return final


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def detect_and_measure(image_bgr, weights, px_per_mm, roi=None, imgsz=1024,
                       conf=0.4, iou=0.9, device="cpu", **filter_kwargs):
    """Full pipeline on a BGR image. Returns (list[Oyster], kept_masks).

    ``roi`` may be given to restrict both segmentation and filtering to the
    tray/board; masks are returned in full-image coordinates.
    """
    if roi is not None:
        x0, y0, x1, y1 = [int(v) for v in roi]
        sub = image_bgr[y0:y1, x0:x1]
        masks_sub = segment_everything(sub, weights, imgsz, conf, iou, device)
        H, W = image_bgr.shape[:2]
        masks = np.zeros((len(masks_sub), H, W), np.uint8)
        masks[:, y0:y1, x0:x1] = masks_sub
    else:
        masks = segment_everything(image_bgr, weights, imgsz, conf, iou, device)

    idx = filter_oysters(masks, roi=roi, **filter_kwargs)
    kept = masks[idx] if len(idx) else masks[:0]

    oysters = []
    for k, m in enumerate(kept, start=1):
        length_px, width_px, area_px = measure_mask(m)
        p = shape_props(m)
        oysters.append(Oyster(
            id=k, cx=p["cx"], cy=p["cy"],
            length_px=length_px, width_px=width_px,
            length_mm=length_px / px_per_mm, width_mm=width_px / px_per_mm,
            area_px=area_px,
        ))
    return oysters, kept


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def draw_overlay(image_bgr, oysters, masks):
    out = image_bgr.copy()
    green = np.array([0, 200, 0], np.uint8)
    for o, m in zip(oysters, masks):
        out[m > 0] = (0.55 * out[m > 0] + 0.45 * green).astype(np.uint8)
        c = _largest_contour(m)
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
        cv2.drawContours(out, [box], 0, (0, 140, 255), 3)
        cv2.putText(out, str(o.id), (int(o.cx) - 15, int(o.cy) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3, cv2.LINE_AA)
    return out


def write_csv(oysters, path, meta=None):
    import csv
    meta = meta or {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Name", "Oyster", "cx_px", "cy_px",
                    "length_px", "width_px", "length_mm", "width_mm", "area_px"])
        img = meta.get("image", "")
        for o in oysters:
            w.writerow([img, o.id, round(o.cx, 1), round(o.cy, 1),
                        round(o.length_px, 1), round(o.width_px, 1),
                        round(o.length_mm, 2), round(o.width_mm, 2),
                        round(o.area_px, 1)])


def load_calibration(path):
    with open(path) as f:
        return json.load(f)
