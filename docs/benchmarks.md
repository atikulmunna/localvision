# Benchmarks

Samsung Galaxy S23+ (SM-S916N), Snapdragon 8 Gen 2, Android 17. Models
exported `w8a32` (dynamic int8).

Measured by `BenchmarkTest` over fixture images, not the live camera. A live
frame's cost depends on what the phone is pointed at, because stage two costs
one inference per detected product, so live numbers are not comparable
between runs. Means over 20 runs for POSM and 10 for QPDS, after 3 warm-up
runs.

## Per frame

| Pipeline | imgsz | Delegate | Load | Frame | Breakdown |
|---|---|---|---|---|---|
| POSM | 640 | GPU | 4560 ms | **368 ms** | prep 165, infer 202 |
| POSM | 960 | GPU | 5406 ms | 705 ms | prep 306, infer 399 |
| QPDS | 640 | GPU | 7442 ms | 3086 ms | prep 212, seg 236, cls x24 2636 |
| POSM | 640 | CPU | **124 ms** | 1384 ms | prep 118, infer 1265 |
| POSM | 960 | CPU | 196 ms | 3232 ms | prep 262, infer 2969 |
| QPDS | 640 | CPU | 117 ms | 4456 ms | prep 130, seg 1880, cls x24 2445 |

YUV to upright RGB, paid per frame before any model runs and not included
above:

| Analysis stream | Cost |
|---|---|
| 720x1280 (current) | 84 ms |
| 480x640 (CameraX default) | 28 ms |

## What these say

**The GPU delegate costs seconds to start and pays it back in one second of
use.** Loading on GPU takes 4.6 s against 124 ms on CPU, a 37x difference,
but each frame is then 368 ms against 1384 ms, nearly 4x faster. The entire
startup wait is delegate initialisation, not reading the model: a separate
measurement put the memory map at 0 ms and interpreter construction at
3035 ms.

This corrects an earlier conclusion recorded during debugging, that GPU and
CPU loads were both about 3.5 s. That measurement was taken while the GPU
delegate was failing and falling back, so the "CPU" number included a failed
GPU attempt. CPU alone is 124 ms.

**Stage two dominates QPDS**, and by a long way. On a shelf with 24 products,
segmentation costs 236 ms and classification costs 2636 ms, about **110 ms
per product**. That is the fixed batch of 1 in the exported classifier: 24
products means 24 separate inferences. The instance cap is the only lever,
and at its default of 40 it is not really capping anything on a normal shelf.

**960 costs roughly double 640** and buys detail the letterbox mostly
discards. 640 on GPU at 368 ms is the only combination that feels live.

**The 720p analysis stream costs 84 ms per frame** against 28 ms at CameraX's
default. That was bought deliberately, to make the analysis field of view
match the preview's; a 4:3 analysis stream against a 16:9 preview misplaced
every box.

## Replacing the spec's estimates

Spec section 4 estimated 2 to 4 inferences per second for POSM and 0.5 to 1.5
for QPDS. Measured, including the YUV conversion:

| | Estimated | Measured |
|---|---|---|
| POSM 640 GPU | 2 to 4 /sec | **2.2 /sec** (452 ms) |
| QPDS 640 GPU | 0.5 to 1.5 /sec | **0.32 /sec** (3170 ms) |

POSM landed inside the estimate. QPDS is about three times slower than the
optimistic end, for a reason the spec did not anticipate: it assumed the
classifier could take a batch of 8, and the export only accepts a batch of 1.

## The startup latency, and what would fix it

Opening the camera shows a preview in 125 ms and the first boxes about 4.5 s
later, all of it GPU delegate initialisation.

The measurements point at a fix that was not obvious before them: **start on
CPU and move to GPU once it is ready.** CPU is usable in 124 ms, so boxes
would appear in well under a second, and the swap to GPU would land a few
seconds later and quadruple the frame rate. The machinery already exists,
since changing delegate at runtime is a supported control and the swap is
already serialised onto the analysis thread.

Not implemented here. It is a behaviour change rather than a measurement, so
it belongs to whoever decides this app's priorities, and static int8
(Task 18) may shrink the load enough to make it unnecessary.
