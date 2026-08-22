"""Per-image pixel->millimetre scale for oyster tray photos.

The pipeline originally hardcoded a single ``px_per_mm`` fitted to one
reference image. That constant is only valid at the camera height and
resolution of that shot, so it cannot be reused across field photos taken
with different rigs.

This module resolves scale per image, in priority order:

1. ``--px-per-mm``          explicit value from the caller (always wins);
2. a per-image or per-set entry in a scales JSON file (see ``load_scales``);
3. ``--ref-px``/``--ref-mm`` a measured reference distance in the photo
   (e.g. two ends of a caliper beam, or a known board edge);
4. otherwise: **no scale**. Counting still works; measurements are withheld
   rather than computed from an invalid constant.

``propose_tick_scale`` offers an *assisted* estimate by finding a periodic
tick comb, but it is advisory only and must be confirmed - halftone screens,
deck gratings and mesh are often more periodic than the ruler itself.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Stored scales
# --------------------------------------------------------------------------- #
def load_scales(path):
    """Load a scales file: {"images": {name: px_per_mm}, "sets": {dir: px_per_mm}}."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def resolve(image_path, px_per_mm=None, scales=None, ref_px=None, ref_mm=None):
    """Return (px_per_mm, source) or (None, 'none') if scale is unknown."""
    if px_per_mm:
        return float(px_per_mm), "explicit"
    if ref_px and ref_mm:
        return float(ref_px) / float(ref_mm), f"reference {ref_px}px/{ref_mm}mm"
    scales = scales or {}
    name = os.path.basename(image_path)
    if name in scales.get("images", {}):
        return float(scales["images"][name]), "scales.json[image]"
    setname = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    if setname in scales.get("sets", {}):
        return float(scales["sets"][setname]), f"scales.json[set:{setname}]"
    return None, "none"


# --------------------------------------------------------------------------- #
# Assisted tick-comb proposal (advisory - always confirm)
# --------------------------------------------------------------------------- #
def _whitened_peaks(gray, win=192, stride=96, pmin=4.0, pmax=30.0, topk=40):
    H, W = gray.shape
    hann = np.outer(np.hanning(win), np.hanning(win))
    fy = np.fft.fftfreq(win)[:, None] * np.ones((1, win))
    fx = np.fft.fftfreq(win)[None, :] * np.ones((win, 1))
    rad = np.sqrt(fy ** 2 + fx ** 2)
    nb = win // 2 + 2
    ridx = np.minimum((rad * win).astype(int), nb - 1)
    band = (rad > 1.0 / pmax) & (rad < 1.0 / pmin)
    out = []
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            p = gray[y:y + win, x:x + win].astype(np.float32)
            if p.std() < 12:
                continue
            F = np.fft.fft2((p - p.mean()) * hann)
            P = F.real ** 2 + F.imag ** 2
            s = np.bincount(ridx.ravel(), P.ravel(), minlength=nb)
            c = np.bincount(ridx.ravel(), minlength=nb)
            prof = s / np.maximum(c, 1)
            Pw = np.where(band, P / np.maximum(prof[ridx], 1e-9), 0.0)
            i = int(np.argmax(Pw))
            iy, ix = divmod(i, win)
            out.append((float(Pw[iy, ix]), float(1.0 / rad[iy, ix]),
                        float(np.degrees(np.arctan2(fy[iy, ix], fx[iy, ix])) % 180),
                        x, y))
    out.sort(key=lambda r: -r[0])
    return out[:topk]


def propose_tick_scale(image_bgr, expect_mm=1.0):
    """Propose px_per_mm from the strongest periodic comb in the image.

    ADVISORY ONLY. Returns a dict with the candidate period, its location and
    a confidence, or None. The caller must verify against the actual ruler -
    halftone patches, gratings and mesh routinely outscore printed rulers.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    peaks = _whitened_peaks(gray)
    if not peaks:
        return None
    top = peaks[0]
    return {
        "period_px": top[1],
        "px_per_mm": top[1] / float(expect_mm),
        "angle_deg": top[2],
        "at_xy": (top[3], top[4]),
        "confidence": "low - verify against the ruler before use",
    }
