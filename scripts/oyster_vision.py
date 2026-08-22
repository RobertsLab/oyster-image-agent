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
                       device="cpu", max_det=300):
    """Run FastSAM in 'segment everything' mode. Returns an (N, H, W) uint8 array.

    ``max_det`` is FastSAM's proposal cap. The default of 300 silently truncates
    dense seed/spat photos, so raise it when a tray holds more than ~300 objects.
    """
    from ultralytics import FastSAM

    model = FastSAM(weights)
    res = model(image_bgr, device=device, retina_masks=True, imgsz=imgsz,
                conf=conf, iou=iou, verbose=False, max_det=max_det)[0]
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


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_iou_fast(a, b, ba, bb):
    """IoU of two full-frame masks, evaluated only where their boxes overlap.

    A plain ``_mask_iou`` compares every pixel of two full-resolution frames,
    so an O(n^2) NMS over a few hundred masks becomes untenable once
    ``max_det`` is raised. Disjoint boxes are 0 by construction, and
    overlapping ones only need the intersecting window.
    """
    if ba is None or bb is None:
        return 0.0
    x0 = max(ba[0], bb[0]); y0 = max(ba[1], bb[1])
    x1 = min(ba[2], bb[2]); y1 = min(ba[3], bb[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    sa = a[y0:y1, x0:x1].astype(bool)
    sb = b[y0:y1, x0:x1].astype(bool)
    inter = int(np.logical_and(sa, sb).sum())
    if inter == 0:
        return 0.0
    area_a = int(a.sum()); area_b = int(b.sum())
    union = area_a + area_b - inter
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
# Scale-free adaptive filtering
# --------------------------------------------------------------------------- #
def _mask_texture(lap, m):
    v = lap[m > 0]
    return float(np.mean(np.abs(v))) if v.size else 0.0


def _mask_saturation(hsv, m):
    v = hsv[:, :, 1][m > 0]
    return float(np.mean(v)) if v.size else 0.0


def filter_oysters_adaptive(image_bgr, masks, min_solidity=0.85, min_extent=0.60,
                            aspect_range=(1.05, 3.2), area_lo=0.20, area_hi=5.0,
                            tex_pct=0, max_mean_sat=80.0, nms_iou=0.30,
                            min_abs_area=300, roi=None, return_info=False):
    """Filter masks to oysters without any absolute pixel-size assumption.

    ``filter_oysters`` gates on absolute pixel area (``min_area``/``max_area``),
    which silently encodes one camera height. This variant instead:

    * keeps the scale-free shape gates (solidity / extent / aspect);
    * offers a **texture gate** (``tex_pct``), off by default. It is a
      percentile, so any non-zero value discards that fraction of candidates
      whether or not they are spurious; on the reference image tex_pct=35 cost
      11 real oysters (75 -> 64) while gaining nothing, so it is opt-in only;
    * adds a **saturation gate** - oyster shells are near-neutral grey/brown/
      white, while the false positives are vividly coloured (yellow crates,
      blue clipboards, green tarp, yellow ear tags). Measured on the reference
      image real oysters have median mean-S 57 and 93%% fall below 80, whereas
      on a crate photo the spurious tray masks have median 115 and only 5%%
      fall below 80;
    * derives the size band from the **population median** of the surviving
      masks, so it adapts to whatever scale the photo was taken at.

    Defaults were chosen by sweeping against the reference image: it counts 75
    of 84 here, versus 80 for the original absolute-area filter. That ~5-oyster
    deficit is the cost of dropping the fitted constants, paid on the one image
    those constants were fitted to.

    **Known limitation.** This cannot rescue photos where the segmenter never
    proposes the oysters. On yellow-crate photos FastSAM segments the tray
    lattice rather than the shells, and raising ``imgsz`` to 1536 or 2048 does
    not help (counts plateau near 10 against ~90 visible oysters). Those need a
    trained detector, not better filtering.

    Returns indices into ``masks`` (largest first), or ``(indices, info)``
    when ``return_info`` is set.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    props = [shape_props(m) for m in masks]
    stage1 = []
    for i, p in enumerate(props):
        if p is None or p["area"] < min_abs_area:
            continue
        if p["solidity"] < min_solidity or p["extent"] < min_extent:
            continue
        if not (aspect_range[0] < p["aspect"] < aspect_range[1]):
            continue
        if roi is not None:
            x0, y0, x1, y1 = roi
            if not (x0 <= p["cx"] <= x1 and y0 <= p["cy"] <= y1):
                continue
        stage1.append(i)

    info = {"n_masks": len(masks), "shape": len(stage1), "colour": 0,
            "texture": 0, "area": 0, "final": 0, "median_area": 0.0}
    if not stage1:
        return ([], info) if return_info else []

    stage1c = [i for i in stage1
               if _mask_saturation(hsv, masks[i]) <= max_mean_sat]
    info["colour"] = len(stage1c)

    tex = {i: _mask_texture(lap, masks[i]) for i in stage1c}
    if not tex:
        return ([], info) if return_info else []
    tthr = float(np.percentile(list(tex.values()), tex_pct))
    stage2 = [i for i in stage1c if tex[i] >= tthr]
    info["texture"] = len(stage2)

    areas = np.array([props[i]["area"] for i in stage2], float)
    med = float(np.median(areas))
    stage3 = [i for i in stage2 if area_lo * med <= props[i]["area"] <= area_hi * med]
    info["area"] = len(stage3)
    info["median_area"] = med

    stage3.sort(key=lambda i: -props[i]["area"])
    boxes = {i: _bbox(masks[i]) for i in stage3}
    final = []
    for i in stage3:
        if all(_mask_iou_fast(masks[i], masks[j], boxes[i], boxes[j]) < nms_iou
               for j in final):
            final.append(i)
    info["final"] = len(final)
    return (final, info) if return_info else final


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
