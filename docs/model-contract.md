# Model contract

What the exported `.tflite` files actually take and return, measured from the
files themselves and checked against the `.pt` reference. Every decoder in the
app is written against this document, not against assumption.

Exported 2026-11-01 with ultralytics 8.4.197, litert-torch 0.10.0,
`format=litert`, `quantize=w8a32`, `nms=False`.

## The flag that decides the whole contract

`nms` has three settings and they produce different graphs:

| `--nms` | Ultralytics `nms=` | Output | Consumer must run NMS |
|---|---|---|---|
| `free` (what we ship) | `False` | `[1, 300, 6]` | no |
| `external` | `None` (the default) | `[1, 4+nc, anchors]` | **yes** |
| `embedded` | `True` | varies by format | no |

Exporting without `--nms free` gives the raw one2many head. Measured on POSM at
640, that output was `[1, 52, 8400]` (4 box + 48 classes over 80² + 40² + 20²
anchors), with cx,cy,w,h normalized 0 to 1, and it was duplicate-heavy: 11
anchors above 0.30 of which 9 duplicated a higher-scoring box at IoU above 0.6,
where the reference produced 2 detections. That is the head the app is **not**
built for.

One constraint to remember before Task 18. Ultralytics force-disables the
NMS-free head when `quantize` is `8` or `w8a16`, with the comment that static
activation quantization collapses the end2end class-index output
(`engine/exporter.py`). So **adopting static int8 means adopting the raw head
and writing NMS**. That is a cost of the static int8 decision, not a detail.

## POSM detection

`posm_640.tflite`, `posm_960.tflite`. Source `POSM_YOLO11X.pt`, which is a
yolo26x detection model despite the filename, 48 classes, trained at 960.

### Input

| Property | Value |
|---|---|
| Shape | `[1, 3, N, N]`, N = 640 or 960 |
| Layout | **NCHW**, channels before spatial dimensions |
| Type | `float32` |
| Range | 0.0 to 1.0, so divide byte values by 255 |
| Order | RGB |
| Letterbox | scale by `min(N/W, N/H)`, centred, **padded with 114 grey** |

NCHW matters: the LiteRT exporter traces PyTorch directly and keeps PyTorch's
layout. Do not assume the NHWC that the older onnx2tf route produced.

The 114 grey pad value is what Ultralytics uses, so it is what the model saw in
training. A different pad colour is a silent accuracy change.

### Output

One tensor, `[1, 300, 6]`, `float32`. 300 fixed slots, each row:

| Column | Meaning |
|---|---|
| 0 to 3 | `x1, y1, x2, y2` |
| 4 | score, 0.0 to 1.0, already through sigmoid |
| 5 | class id as a float, round to int, 0 to 47 |

Three properties the decoder relies on, all measured:

1. **Boxes are in letterboxed input pixels**, 0 to N. Not normalized. Reverse
   with `(coord - pad) / scale`.
2. **Rows are sorted by score, descending.** Verified: `all(diff(col4) <= 0)`.
3. **Unused slots are zero-filled.** On the probe image only 7 of 300 rows had
   a score above 0.01, so the decoder can stop at the first row below threshold
   instead of walking all 300.

### Verification against the .pt reference

One real POSM photo (1919x1080,
`CLEAR_MEN_CSM_5ML_BUNTING_JUL26_1_20261025_173711_350d5e03.jpg`) at 640,
threshold 0.30. Rows un-letterboxed into source pixels against
`YOLO(POSM_YOLO11X.pt).predict(imgsz=640, conf=0.30)`:

| | tflite (w8a32) | .pt reference |
|---|---|---|
| 1 | cls 16, 0.468, (784.9, 315.7, 1924.9, 462.0) | cls 16, 0.574, (785.9, 313.7, 1914.4, 461.8) |
| 2 | cls **13**, 0.333, (71.1, 244.1, 780.9, 435.6) | cls **16**, 0.350, (71.2, 243.0, 783.1, 436.1) |

Boxes agree to 1 or 2 pixels, which confirms the coordinate space.

Two differences are real and expected, and both belong in Task 16's analysis
rather than being treated as bugs now:

- **Scores sit lower under w8a32** (0.468 against 0.574). A fixed threshold
  therefore keeps fewer detections on device than on the server.
- **A marginal detection changed class**: 13 against 16 on the 0.33 box, while
  the box itself was identical. Dynamic int8 can flip a class where two
  classes are close. This is the signature of quantization drift, not of a
  preprocessing bug, and the two are distinguishable precisely because the
  box agreed.

### File sizes

| File | Size |
|---|---|
| `posm_640.tflite` | 56.8 MB |
| `posm_960.tflite` | 57.0 MB |

Note for the spec's asset-size risk: w8a32 did **not** quarter the 118 MB
checkpoint. It produced 57 MB, about half, because dynamic int8 quantizes
weights but leaves activations and some layers in float. Two POSM resolutions
are therefore 114 MB on their own, and the spec's "roughly 150 MB total"
estimate will not survive adding QPDS. Revisit at Task 12 with real numbers.

## Still to measure

QPDS-seg and QPDS-cls, at Task 12. Record there: both seg output tensors with
the proto grid shape and channel count, whether the mask coefficients are the
trailing columns of the row tensor, and the cls input range and output length.
