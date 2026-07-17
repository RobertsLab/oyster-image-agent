# Methods & evaluation

## Data

One labelled scene, `20260522_bag380` (Goose Point):

- `20260522_bag380_raw.jpeg` — 4032×3024 overhead photo of ~84 oysters on a
  concrete tray, plus tools, a `380` cattle tag, feet and grating.
- `20260522_bag380_annotated_imageJ.tif` — same image carrying an **ImageJ ROI
  overlay**: a single `POINT` ROI with **84 sub-pixel points**, one crosshair
  per oyster, ordered to match the measurement table.
- `20260522_bag380_data.xlsx` / `outputs/20260522_bag380_data.csv` — human
  length & width in mm for each of the 84 oysters.

### Ground-truth extraction

`scripts/calibrate.py:read_imagej_points` reads the ImageJ metadata from the
TIFF with `tifffile`, parses the ROI blob with `roifile`, and returns the 84
sub-pixel point coordinates. These are the human oyster centroids and are used
both to score detection and to pair detections with the mm measurements
(point *i* ↔ oyster *i+1* in the table).

## Pipeline

Implemented in `scripts/oyster_vision.py`.

1. **Board ROI.** Segmentation and filtering are restricted to the tray, taken
   as the bounding box of the human points + 180 px margin. This keeps FastSAM
   away from feet, grating, the bucket and the notebook. For a new image the ROI
   can be passed on the command line.

2. **Segmentation — FastSAM-s** (`ultralytics`), "segment everything" mode,
   `imgsz=1024`, `retina_masks=True`, `conf=0.4`, `iou=0.9`, CPU. ~130 candidate
   masks, ~7 s on this image.

3. **Oyster filter + NMS** (`filter_oysters`):
   - area 4 000–45 000 px² (oyster-sized at this scale),
   - solidity > 0.85 and rectangular extent > 0.60 (compact, convex blobs —
     rejects mud swirls and thin fragments),
   - aspect ratio 1.05–3.2,
   - centroid inside the board ROI,
   - greedy mask-IoU NMS at 0.30 removes duplicate/overlapping masks.

4. **Measurement** (`measure_mask`): length = maximum pairwise distance across
   the convex hull (max Feret diameter); width = short side of `minAreaRect`.
   Feret length correlates better with the human caliper length (r 0.82) than
   the rectangle's long side (r 0.74), because people record the longest shell
   axis.

5. **Calibration.** A single `px_per_mm` is fit by least squares through the
   origin over the **pooled** length and width measurements of the matched
   oysters: `px_per_mm = Σ(px·mm) / Σ(mm²)` = **2.435**. One scalar is used for
   both axes because pixel↔mm is a single physical scale.

## Evaluation protocol

Detections are matched to the 84 human points by **greedy nearest-neighbour,
one-to-one**, accepting a pair when centroids are within 90 px (~37 mm, well
under one oyster). A global (Hungarian) assignment was rejected: minimising the
*sum* of distances produces pathological long-range pairings when two masks land
on one oyster, understating accuracy.

| | value |
|---|---|
| detected / truth | 80 / 84 |
| TP / FP / FN | 72 / 8 / 12 |
| precision / recall / F1 | 0.90 / 0.86 / 0.88 |
| length MAE / bias / RMSE | 4.6 / −1.0 / 7.1 mm |
| width MAE / bias / RMSE | 4.7 / +2.4 / 6.2 mm |
| length r / width r | 0.82 / 0.73 |

## Limitations & next steps

- **Recall (12 misses).** Oysters fully caked in mud blend into the board and
  FastSAM does not separate them. Raising `imgsz` did not help. A point-prompt
  grid (SAM/MobileSAM), or fine-tuning a YOLO-seg detector once a handful more
  images are annotated, would recover these.
- **False positives (8).** Mostly board texture inside the ROI plus the bright
  `380` tag. A tag-colour gate (very high saturation / specific hue) and a
  learned board mask would cut these.
- **Width fidelity (r 0.73).** The mask's short axis is a rougher proxy for the
  human width than Feret length is for length; a shape model fit per oyster
  (e.g. ellipse or principal-axis width at the mid-length) could tighten it.
- **Calibration transfer.** `px_per_mm` is specific to this camera geometry.
  Put a fiducial (ruler/scale card) in frame for robust per-image scale, or fix
  the mount height. The `380` tag or the caliper in-frame could also serve as a
  known-size reference for auto-scaling.
