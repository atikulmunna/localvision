# Validation

## What this answers, and what it does not

The app runs quantized models at a different resolution from the server, so
some disagreement with production is expected and is not a defect. Chasing it
would waste time.

The question worth answering is narrower: **does the app implement the
pipeline correctly?** A preprocessing mistake and a quantization loss both
show up as "the labels look a bit off", and from inside the app they are
indistinguishable. They are distinguishable by signature:

| | Preprocessing bug | Quantization drift |
|---|---|---|
| Boxes | land nowhere near, or nothing found | agree closely |
| Scores | meaningless | sit a little lower |
| Classes | wrong broadly | occasional flip on a marginal detection |

Both were seen for real while building this. A filtered downscale (PIL rather
than cv2) changed **which class won** on a POSM photo. Separately, the
quantized model called a box class 13 where the reference said 16, with the
box identical to a pixel.

## The reference, and why it is not the server

The reference is **the same exported `.tflite` files the app ships**, run in
Python with cv2 preprocessing (`tools/reference_labels.py`).

It is deliberately **not** `YOLO(...).predict()` on the `.pt`. That path runs
the one2many head through NMS, while our export ships the NMS-free one2one
head. They are different heads of the same network and are entitled to
disagree, so a difference there would prove nothing about the app.

For QPDS the reference reimplements the two-stage path in Python, including
the masked crop. That is the point of it: evidence only counts if the two
implementations were written separately.

## Result

Run with `tools/diff_labels.py`, matching detections positionally at IoU 0.5
so that a class flip on an agreeing box is visible rather than hidden.

### POSM, 640, allowlist off

| image | matched | ref only | app only | flips | mean IoU | mean score delta |
|---|---|---|---|---|---|---|
| posm_fixture.jpg | 1 | 0 | 0 | 0 | 0.990 | +0.070 |
| qpds_fixture.jpg | 4 | 0 | 0 | 0 | 0.997 | +0.001 |

### QPDS two-stage, 640, DOVE_LUX_INTERNATIONAL

| image | matched | ref only | app only | flips | mean IoU | mean score delta |
|---|---|---|---|---|---|---|
| posm_fixture.jpg | 0 | 0 | 0 | 0 | n/a | n/a |
| qpds_fixture.jpg | 24 | 0 | 0 | 0 | 0.987 | +0.000 |

**Verdict: agree.** Every detection matched, nothing was invented or lost, no
class flipped, and boxes agree to roughly 1% IoU. This is the "neither"
outcome: no preprocessing bug, and on these images not even visible drift.

It covers the whole chain, not just the easy half. The QPDS row exercises
`CropBuilder` and the mask decode, which POSM never touches, against a
separately written implementation of the server's crop geometry.

The positive score delta is worth noting: the device scored *higher* than the
Python run of the same file (+0.070 on one image). Same class, same box, so
this is arithmetic order and the delegate, not preprocessing.

Zero detections on `posm_fixture.jpg` under QPDS is correct. It is a POSM
poster photographed against a door, with no DOVE or LUX products in it.

## The gap in this evidence

**Two images, not the twenty the plan called for.**

The plan specified roughly 20 images spanning at least three QPDS slabs and
three POSM categories, drawn from `M:\TDI\UBL-Git\datapipe`. That corpus is
no longer on this machine: `datapipe/s3/downloaded_images` is empty, and what
remains under `datapipe` is SAM3 demo material (`groceries.jpg`, `truck.jpg`)
plus 270 frames from a single video clip, none of it UBL shelf imagery.

The two images used are the real UBL photos already committed as test
fixtures. They are genuine shop photographs and they do exercise both
pipelines end to end, but two images cannot show:

- behaviour across different slabs and categories
- rare classes, crowded shelves, poor lighting, odd angles
- whether the agreement here is typical or lucky

So the honest claim is: **the app's implementation matches the reference on
every image tested, and the sample is small.** Point
`tools/reference_labels.py` and `LabelDumpTest` at a larger set and the tables
above regenerate.

## Reproducing

The app side is an instrumented test rather than a debug activity, so it needs
no exported activity and no storage permissions. Note that
`connectedDebugAndroidTest` **uninstalls the app when it finishes**, which
deletes the output, so drive it with `am instrument` instead:

```bash
./gradlew :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w \
  -e class com.tdi.ubldetect.pipeline.LabelDumpTest \
  com.tdi.ubldetect.test/androidx.test.runner.AndroidJUnitRunner
adb shell run-as com.tdi.ubldetect cat files/labeldump/posm.json > validation/app_posm.json
adb shell run-as com.tdi.ubldetect cat files/labeldump/qpds.json > validation/app_qpds.json
```

Then, in WSL2:

```bash
python reference_labels.py --model ../exports/posm_640.tflite \
  --images ../validation/images --out ../validation/ref_posm.json
python diff_labels.py --reference ../validation/ref_posm.json --app ../validation/app_posm.json
```

Add `--cls-model ../exports/qpds_cls_224.tflite --catalog ... --entry <SLAB>`
for the QPDS two-stage reference.
