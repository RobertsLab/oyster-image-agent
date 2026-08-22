# Oyster image agent — count & measure oysters from tray photos

A vision pipeline that finds individual oysters in a field tray photograph,
counts them, and measures each one's **length** and **width** in millimetres —
reproducing the manual ImageJ workflow (place a point on each oyster, measure
with calipers) automatically.

![overlay example](outputs/20260522_bag380_raw_overlay.jpg)

## How it works

The system has no task-specific training (the dataset is a single labelled
image), so it uses a **pretrained "segment everything" model, FastSAM**, and
adds a domain-specific filter + measurement + calibration layer on top:

1. **Segment** — FastSAM proposes every object mask in the tray region.
2. **Filter to oysters** — keep masks in the oyster size range with high
   solidity/extent and a plausible aspect ratio, inside the board region of
   interest; de-duplicate overlapping masks with mask-IoU NMS.
3. **Measure** — for each surviving mask:
   - *length* = maximum Feret (caliper) diameter of the mask outline — the
     longest dimension, matching how a person measures hinge-to-bill;
   - *width* = short side of the minimum-area bounding rectangle.
4. **Calibrate** — convert pixels to millimetres with a single `px_per_mm`
   constant fit against the human measurements (see below).

## Results on the reference image (`bag380`, 84 oysters)

| Metric | Value |
|---|---|
| Oysters counted | **80** (truth 84) |
| Detection precision / recall / F1 | **0.90 / 0.86 / 0.88** |
| Length error (MAE) | **4.6 mm (4.8%)**, r = 0.82 |
| Width error (MAE) | **4.7 mm (7.9%)**, r = 0.73 |
| Calibration | **2.435 px/mm** (0.411 mm/px) |

Full numbers are written to [`outputs/metrics.json`](outputs/metrics.json) and
[`outputs/calibration.json`](outputs/calibration.json).

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # FastSAM weights auto-download on first run

# Re-derive px/mm + validation metrics from the annotated reference image
python scripts/calibrate.py

# Count & measure oysters in any tray image
python scripts/detect.py --image field-images/20260522_bag380_raw.jpeg
#   -> outputs/<name>_predictions.csv   (one row per oyster: px + mm L/W, centroid, area)
#   -> outputs/<name>_overlay.jpg       (masks + oriented boxes + IDs)
```

Key options for `detect.py`: `--roi X0 Y0 X1 Y1` to set the tray box for a new
image, `--no-roi` to scan the whole frame, `--px-per-mm` to override calibration.

## Repository layout

```
scripts/
  oyster_vision.py   # core library: segment, filter, measure, overlay, I/O
  calibrate.py       # derive px/mm + eval against ImageJ ground truth
  detect.py          # CLI: image -> predictions CSV + overlay
field-images/        # raw + ImageJ-annotated reference image, human data
outputs/             # predictions, overlay, calibration.json, metrics.json
models/              # FastSAM-s.pt (auto-downloaded, git-ignored)
METHODS.md           # detailed methodology, ground-truth extraction, limitations
```

## Important caveats

- **Calibration is scene-specific.** `px_per_mm` was fit for this camera height
  and tray. For accurate millimetres on new photos, either keep the camera
  geometry fixed **or** (recommended) place a ruler / scale card in frame and
  re-derive the scale. See [METHODS.md](METHODS.md).
- The detector misses oysters that are heavily mud-merged with the board
  (~14% here) and occasionally counts a non-oyster object inside the tray (the
  `380` cattle tag is one such false positive in the example).

See [METHODS.md](METHODS.md) for the ground-truth extraction, the accuracy
breakdown, and ideas for improving both count recall and measurement fidelity.
