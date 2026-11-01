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

### Verification, and the resampling that decides it

One real POSM photo (1919x1080,
`CLEAR_MEN_CSM_5ML_BUNTING_JUL26_1_20261025_173711_350d5e03.jpg`) at 640,
threshold 0.30.

**Downscale resampling changes the answer, so it is part of the contract.**
Fitting a 1919 pixel frame into 640 is a 3x reduction. PIL's `resize` filters
as it downscales; `cv2.INTER_LINEAR` does not, and Ultralytics uses cv2. The
two preprocessings disagree about what the top detection even is:

| Preprocessing | Top row |
|---|---|
| PIL BILINEAR (filtered) | cls 16 @ 0.468, (261.8, 245.3, 642.0, 294.1) |
| cv2 INTER_LINEAR (unfiltered) | cls 13 @ 0.489, (28.7, 222.2, 369.3, 291.9) |
| **The app, on device** | **cls 13 @ 0.559, (28.8, 222.0, 370.5, 291.5)** |

The app matches cv2 to about a pixel on the boxes and to the class, which is
the agreement that matters: cv2 is what the reference pipeline does. An
earlier version of this document quoted the PIL numbers, which made the app
look wrong when it was the measurement that was unrepresentative.

So `InputTensor` resamples with plain bilinear and no antialiasing, on
purpose. It would be an easy and wrong "improvement" to add filtering here.

### Why not compare against `YOLO(...).predict()`

Running the `.pt` through Ultralytics on this image returns class 16 twice,
which looks like a third disagreement and is not a fair comparison. YOLO26
carries two heads. `predict()` uses the one2many head and runs NMS over it;
our export with `nms=False` ships the one2one NMS-free head. They are
different heads of the same network and are entitled to disagree on marginal
detections.

The honest reference for the app is therefore **the same exported `.tflite`,
run in Python with cv2 preprocessing**. That is the row the table above
compares against, and the app reproduces it.

Scores still sit a little higher on device than in the Python run of the same
file (0.559 against 0.489). Same class, same box, so this is arithmetic order
and the delegate, not preprocessing. Task 16 is where that gets characterised
properly.

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

## QPDS segmentation

`qpds_seg_640.tflite`, `qpds_seg_960.tflite`. Source `QPDS-seg.pt`, yolo26x-seg,
1 class (product), trained at 1280.

It exported without incident. Ultralytics issue #23557 is a YOLO26 **seg**
model failing TFLite export, so this was the task most likely to fail; the
modern LiteRT path handled it and the documented `end2end=False` fallback was
not needed.

### Input

Same as POSM: `[1, 3, N, N]` float32 NCHW, 0.0 to 1.0, RGB, letterboxed with
114 grey. N is 640 or 960.

### Outputs

Two tensors.

**Rows**, `[1, 300, 38]` float32. The detect layout with the mask coefficients
appended:

| Column | Meaning |
|---|---|
| 0 to 3 | `x1, y1, x2, y2` in letterboxed input pixels |
| 4 | score, 0.0 to 1.0 |
| 5 | class id, always `0`: this model has one class |
| 6 to 37 | 32 mask coefficients, roughly -7.4 to +7.7 |

Score-sorted descending and zero-tailed, same as POSM, so the decoder stops at
the first row below threshold.

**Protos**, `[1, 32, P, P]` float32, where **P is N/4**: 160 at imgsz 640, 240
at 960. Not the 64 the plan guessed, which is why this is measured rather than
assumed.

### Building a mask

`sigmoid(coeffs . protos)` over the 32 channels, giving a P by P grid in 0..1.
An instance's box maps into that grid by dividing by 4.

Verified on a real shelf photo (1440x1080, 24 instances at `seg_conf` 0.20):
for the top instance the mask averaged **0.948 inside its box against 0.374
over the whole grid**, which is the separation the crop builder depends on.

## QPDS classification

`qpds_cls_224.tflite`. Source `QPDS-cls.pt`, yolo26x-cls, 56 classes.

| Property | Value |
|---|---|
| Input | `[1, 3, 224, 224]` float32 NCHW, 0.0 to 1.0, RGB |
| Output | `[1, 56]` float32 |

**The output is already softmaxed**, summing to 1.0000. It is not logits, so
nothing downstream should apply softmax again.

Sanity check on the same photo: the top instance classified as
`dove_serum_bar` at 1.000, on a DOVE and LUX International shelf. Correct
product family, so the crop geometry and the class indices line up.

## File sizes, and the asset budget

| File | Size |
|---|---|
| `posm_640.tflite` | 56.8 MB |
| `posm_960.tflite` | 57.0 MB |
| `qpds_seg_640.tflite` | 63.9 MB |
| `qpds_seg_960.tflite` | 64.1 MB |
| `qpds_cls_224.tflite` | 28.9 MB |
| **All five** | **270.7 MB** |

The spec's section 11 estimated "roughly 150 MB". That was wrong, because
`w8a32` halves a checkpoint rather than quartering it: dynamic int8 quantizes
weights but leaves activations and some layers in float.

Shipping one resolution per model instead of two brings it to about 150 MB and
costs the in-app resolution switch. That is a decision for whoever ships this,
not a defect.

## Still to measure


Nothing. All five shipped models are recorded above.

Static int8 (Task 18) will need its own pass, because Ultralytics force-
disables the NMS-free head for `quantize=8`, which changes the row contract
and brings NMS back.
