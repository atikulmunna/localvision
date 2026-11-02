# localVision

An Android app that runs the UBL POSM and QPDS models on the phone, fully
offline, and draws boxes with product names over the camera.

It is a viewer. It shows you what the models see. It does not score
compliance, and it is not a measurement of how good the models are.

## This is not an accuracy measurement

Worth saying first, because it is the easiest mistake to make with it.

The labels this app draws **will differ from the server's**, by design:

- it runs at **640 or 960**, where production runs QPDS segmentation at
  **1280**, a setting worth a measured +84.8% on images where 640 finds
  fewer than 10 products
- the models are **quantized** (`w8a32`), and quantization shifts scores and
  occasionally flips a marginal class
- there is **no compliance layer at all**: no planogram, no order checks, no
  shelftalker, exclusivity, SOS or sachet, no aggregation

So a product this app misses is not evidence that production missed it, and
a label it gets wrong is not a production defect. Use it to see detections
with your own eyes, not to judge a model. `docs/validation.md` has what was
actually verified.

## What it does

Pick **POSM** or **QPDS**, then a category or a slab, then point the camera.

Choosing matters: it restricts labelling to the classes that entry expects,
which is both faster to read and more accurate, since a classifier choosing
among a slab's own SKUs makes fewer mistakes than one choosing among all 56.

The picker shows what each entry can actually do, from the catalog's
reconciliation against the shipped model:

- **plain** means every class it expects is in the model
- **amber** means some are missing, which is normal: SKUs reach the standards
  before a model is trained to detect them
- **greyed out and not tappable** means the model has none of its classes, so
  it could never label anything. `Mini meals` is in this state today

### Controls

| | |
|---|---|
| slider | confidence, default 0.30, the server's `posm.confidence` |
| 640 / 960 | inference resolution; lower is faster, higher sees more |
| Filter | restrict to the entry's classes, or show everything the model finds |
| GPU | switch delegate; CPU starts far faster but runs about 4x slower |
| Hold | freeze the boxes so labels can be read |
| Save | write the frame and its labels to app storage |

**Filter off is the useful diagnostic.** With it on, "not detected" and
"detected as something this category does not expect" look identical. Turning
it off tells them apart.

Saved frames land in app-private storage, not the gallery, because they are
photographs of someone's shop:

```bash
adb shell run-as com.tdi.ubldetect ls files/frames
```

## Supported device

One: the Samsung Galaxy S23+ (SM-S916N), Snapdragon 8 Gen 2, Android 17.
`minSdk` is 29 and other devices may work, but nothing else has been tried.

## Speed, measured

POSM 640 on GPU is about **2.2 frames per second**; QPDS about **0.32**. The
preview stays smooth at its own rate and the boxes trail it. Full table in
`docs/benchmarks.md`.

Two numbers worth knowing before you use it:

**The camera appears in 125 ms; the first boxes take about 4.5 s.** That is
GPU delegate initialisation, not the app hanging. The status pill reads
"Loading model" while it happens. CPU loads in 124 ms but each frame costs
nearly 4x more.

**QPDS costs about 110 ms per product.** Stage two runs one inference per
detected item because the exported classifier takes a fixed batch of 1, so a
24 product shelf spends 2.6 s classifying.

## Privacy

The app **declares no network permission at all**, so the platform refuses
network access and no code added later can send a shelf photo anywhere. That
is structural rather than a promise, and
`app/src/androidTest/.../TfliteRunnerTest` asserts the shipped manifest
requests `CAMERA` and nothing else.

## Building

Needs Android Studio's bundled JDK 21; the `java` on `PATH` is likely too old
for AGP.

```bash
export JAVA_HOME="/c/Program Files/Android/Android Studio/jbr"
./gradlew :app:installDebug
```

Model files are **not in git** (they are 57 MB each). Export them first, then
copy into `app/src/main/assets/models/`. See `tools/README.md`; export runs
in WSL2, since Ultralytics LiteRT export is Linux and macOS only.

`UBL-standalone` is read only. Nothing here writes to it; the tooling
references its checkpoints and standards by absolute path.

### Tests

```bash
./gradlew :app:testDebugUnitTest          # 104 JVM tests
./gradlew :app:connectedDebugAndroidTest  # 28 on the device
cd tools && python -m pytest tests/ -q    # 46 for the export and catalog tooling
```

**`connectedDebugAndroidTest` uninstalls the app when it finishes.** Install
after running it, not before, or the phone is left with no app. Anything that
writes results to app storage (the label dump, the benchmark) must be driven
with `adb shell am instrument` instead, or the output is deleted with the app.

## How it is put together

```
tools/          export and catalog generation (Python, WSL2, never shipped)
app/            the app (Kotlin, CameraX, LiteRT)
  imaging/      letterbox, YUV to RGB, input tensor, masked crops
  infer/        interpreter, detect and segment decoders, classifier
  pipeline/     POSM one stage, QPDS two stage
  catalog/      the generated class catalog
  ui/           picker, camera, overlay, controls
docs/           model contract, validation, benchmarks
```

Two documents are worth reading before changing anything:

**`docs/model-contract.md`** records what the exported models actually take
and return, measured rather than assumed. It also records the two traps:
exporting without `nms=False` silently produces a different, duplicate-heavy
head, and resampling with a filter (PIL rather than cv2) changes which class
wins.

**`docs/validation.md`** records what was verified, what was not, and why
static int8 was rejected.
