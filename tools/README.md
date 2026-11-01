# Export and catalog tooling

Build tooling. None of it ships in the app, and the app has no Ultralytics
dependency: it gets the exported `.tflite` weight files only. That is what keeps
the Ultralytics AGPL licence out of the app.

## UBL-standalone is read only

`M:\TDI\UBL-Git\UBL-standalone` is reference material. Nothing here writes to
it, moves anything in it, or runs with it as the working directory.

This is not just a convention to remember. Ultralytics writes its export output
into the directory holding the source checkpoint, so exporting straight from
that repo would drop a `_saved_model` folder and `.tflite` files inside it.
`export_litert.py` therefore copies each checkpoint into `exports/_staging/`
first and exports from the copy.

After any export, confirm the repo is untouched:

```bash
git -C "M:/TDI/UBL-Git/UBL-standalone" status --short   # must print nothing
```

## Environment

LiteRT export runs on Linux x86_64 and macOS only, so it runs in WSL2 Ubuntu,
never in the Windows Python.

The virtualenv lives in the WSL2 filesystem rather than in this repo, because
pip writes thousands of small files and the `/mnt/m` DrvFs mount is far slower:

```bash
python3 -m venv ~/.venvs/ubl-detect-viewer
. ~/.venvs/ubl-detect-viewer/bin/activate
pip install -r /mnt/m/TDI/UBL-Git/ubl-detect-viewer/tools/requirements.txt
```

Resolved versions from the first install are recorded at the bottom of this
file. The one hard requirement is that this is **not** the 8.4.11 that
UBL-standalone pins: Ultralytics issue #23557 is a YOLO26 segmentation export
failing on exactly that version, inside the onnx2tf path that current releases
no longer use.

## Exporting

```bash
SA=/mnt/m/TDI/UBL-Git/UBL-standalone
cd /mnt/m/TDI/UBL-Git/ubl-detect-viewer/tools

python export_litert.py --checkpoint $SA/models/POSM_YOLO11X.pt --imgsz 960 --quantize w8a32 --out-dir ../exports
python export_litert.py --checkpoint $SA/models/POSM_YOLO11X.pt --names-only --out-dir ../exports
```

`--quantize` takes `w8a32` (dynamic int8, no calibration data needed), `8`
(static int8, needs `--data`), `w8a16`, or `32` for float.

**`--nms free` is not optional for this project.** It is the default here for
that reason. It asks Ultralytics for the NMS-free end2end head, which outputs
`[1, 300, 6]`: one row per detection, already sorted, nothing to suppress on
device. Without it you get the raw one2many head (`[1, 4+nc, anchors]`,
normalized cx,cy,w,h, heavily duplicated) and the app would have to implement
NMS. See `docs/model-contract.md` for both measured shapes.

Ultralytics force-disables the NMS-free head for `--quantize 8` and
`--quantize w8a16`, because static activation quantization collapses its
class-index output. Static int8 therefore implies the raw head plus NMS.

### Asset names are a contract

The app opens models by exact asset path, so the names are derived in one place
(`asset_name`, `tflite_filename`) and pinned by `tests/test_export_naming.py`:

| Checkpoint | Asset name | Exported as |
|---|---|---|
| `POSM_YOLO11X.pt` | `posm` | `posm_640.tflite`, `posm_960.tflite` |
| `QPDS-seg.pt` | `qpds_seg` | `qpds_seg_640.tflite`, `qpds_seg_960.tflite` |
| `QPDS-cls.pt` | `qpds_cls` | `qpds_cls_224.tflite` |

Class names dump to `<CheckpointStem>.names.json`, keyed by checkpoint stem
rather than asset name, so the file is traceable to the exact weights it came
from.

Copy the chosen `.tflite` files into `app/src/main/assets/models/`. They are
gitignored and never committed.

## Tests

The naming contract is testable without torch, because `export_litert.py`
imports Ultralytics lazily inside the functions that need it:

```bash
python -m pytest tests/ -q
```

## Resolved versions

Recorded after the first install, so a future export can reproduce this one.

First install, 2026-11-01, WSL2 Ubuntu 26.04.1 LTS:

| Package | Version |
|---|---|
| ultralytics | 8.4.197 |
| litert-torch | 0.10.0 |
| litert-converter | 0.5.0 |
| ai-edge-quantizer | 0.10.0 |
| ai-edge-litert | 2.3.0 |
| torch | 2.14.1 |
| torchvision | 0.29.2 |
| numpy | 2.5.4 |
| pyyaml | 6.0.3 |
| pytest | 9.2.0 |
| Python | 3.14.4 |

8.4.197 is well clear of the 8.4.11 that issue #23557 was filed against.

**`litert-torch` is a required dependency that Ultralytics does not install.**
Ultralytics 8.4.197 exports LiteRT by tracing PyTorch directly through
`litert_torch`, and without it every export dies with
`ModuleNotFoundError: No module named 'litert_torch'` after the model has
already loaded, which reads like a model problem but is not one.

Installing it downgrades torch from 2.15.0 to 2.14.1, which is why torch is
listed above at 2.14.1. Install `litert-torch` first, or accept the downgrade.

### One gotcha when driving this from outside WSL

A shell variable set inside `wsl.exe -d Ubuntu -- bash -lc '...'` does not
survive into the command, so `SA=/mnt/...` followed by `$SA/models/x.pt` silently
becomes `/models/x.pt`. Inside an interactive WSL shell the variable is fine.
When scripting from Windows, write the checkpoint path out in full.
