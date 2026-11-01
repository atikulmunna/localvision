"""Export UBL YOLO checkpoints to LiteRT (.tflite) and dump their class names.

Build tooling. Nothing here ships in the app: the app gets the exported weight
files only, which is what keeps the Ultralytics AGPL licence out of it.

Two rules this module exists to enforce:

1. `M:\\TDI\\UBL-Git\\UBL-standalone` is READ ONLY. Ultralytics writes its
   export next to the source checkpoint, so exporting straight from that repo
   would create files inside it. Every export therefore runs from a staged
   copy of the checkpoint under `exports/_staging/`.
2. Asset names are a contract. The Android app opens `models/posm_960.tflite`
   by that exact name, so the name is derived here, in one place, and tested.

LiteRT export runs on Linux x86_64 and macOS only, so run this in WSL2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Checkpoint stem to app asset name. Explicit, because the filenames are
# historical: POSM_YOLO11X.pt is a YOLO26 model.
ASSET_NAMES = {
    "POSM_YOLO11X": "posm",
    "POSM_COMP": "posm_comp",
    "QPDS-seg": "qpds_seg",
    "QPDS-cls": "qpds_cls",
    "Shelftalker": "shelftalker",
    "EXCLUSIVITY": "exclusivity",
}


def asset_name(checkpoint: Path) -> str:
    """App-facing short name for a checkpoint, e.g. POSM_YOLO11X.pt -> posm."""
    stem = checkpoint.stem
    if stem in ASSET_NAMES:
        return ASSET_NAMES[stem]
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def tflite_filename(checkpoint: Path, imgsz: int) -> str:
    """Asset filename for an exported model at one resolution."""
    return f"{asset_name(checkpoint)}_{imgsz}.tflite"


def names_filename(checkpoint: Path) -> str:
    """Filename for the dumped class names, keyed by checkpoint stem."""
    return f"{checkpoint.stem}.names.json"


def export_target(checkpoint: Path, imgsz: int, out_dir: Path) -> Path:
    """Absolute path the exported asset is moved to.

    Absolute on purpose. A relative path is resolved against the working
    directory at the moment it is used, and Ultralytics runs a conversion
    lasting tens of seconds between the moment this path is computed and the
    moment it is read back. Keeping it absolute removes a whole class of
    "the file is right there but stat says it is not" failure.
    """
    return (out_dir.resolve() / tflite_filename(checkpoint, imgsz)).resolve()


def size_of(path: Path) -> int:
    """Size of `path` in bytes, 0 if it is not there.

    Asks the same two ways as `_present`, for the same measured reason: on
    /mnt/m a freshly moved large file can be listed by its directory and
    still raise ENOENT from `stat` in the process that moved it.
    """
    try:
        return path.stat().st_size
    except OSError:
        pass
    try:
        with os.scandir(path.parent) as entries:
            for entry in entries:
                if entry.name == path.name:
                    try:
                        return entry.stat().st_size
                    except OSError:
                        return 0
    except OSError:
        pass
    return 0


def _present(path: Path) -> bool:
    """Whether `path` is really there, asking two independent ways.

    `stat` is the obvious way and the one that lies. Exporting a 57 MB model
    to /mnt/m produced a process in which, at one moment, the directory
    listing contained the file while `stat` raised FileNotFoundError for
    that exact path, from both `Path.stat` and `DirEntry.stat`, and a fresh
    process stat'd it without complaint. The root cause was not isolated: it
    survives a 10 second wait, and neither a 1 KB move nor a 57 MB move
    reproduces it in isolation. So the check rests on the mechanism that was
    observed to work, which is reading the directory.

    Note what this deliberately does NOT do: it does not go through
    `size_of`. A listed file whose size cannot be read is present, but its
    size is unknown, and conflating those two meant an export with correct
    output reported failure.

    A stattable zero-byte file still counts as absent, because a truncated
    model is worse than a missing one: it fails later and less clearly.
    """
    try:
        return path.stat().st_size > 0
    except OSError:
        pass
    try:
        with os.scandir(path.parent) as entries:
            for entry in entries:
                if entry.name != path.name:
                    continue
                try:
                    return entry.stat().st_size > 0
                except OSError:
                    return True  # listed and unstattable: listing wins
    except OSError:
        pass
    return False


def wait_for_file(path: Path, timeout: float = 10.0) -> bool:
    """True once `path` is a real non-empty file, False if it never shows up.

    Bounded, so a move that genuinely failed still reports failure rather
    than hanging. See `_present` for why this is not a plain stat.
    """
    deadline = time.monotonic() + timeout
    while True:
        if _present(path):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _parse_quantize(value: str | None):
    """LiteRT takes 8 and 32 as ints, w8a16 and w8a32 as strings."""
    if value is None:
        return None
    return int(value) if value in {"8", "32"} else value


def _stage(checkpoint: Path, out_dir: Path) -> Path:
    """Copy the checkpoint somewhere writable, so the source repo is untouched."""
    staging = out_dir.resolve() / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged = (staging / checkpoint.name).resolve()
    if not staged.exists() or staged.stat().st_size != checkpoint.stat().st_size:
        shutil.copy2(checkpoint, staged)
    return staged


def _find_tflite(result) -> Path:
    """Ultralytics returns a file or a directory depending on format and version."""
    path = Path(str(result))
    if path.is_file() and path.suffix == ".tflite":
        return path
    candidates = sorted(path.glob("**/*.tflite")) if path.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no .tflite produced under {path}")
    # Prefer the smallest, which is the quantized one when several are written.
    return min(candidates, key=lambda p: p.stat().st_size)


# How the detection head leaves the graph.
#   "free"     -> nms=False: the NMS-free end2end head. One row per detection,
#                 no duplicates, nothing to suppress on device. Available for
#                 float and w8a32, but Ultralytics force-disables it for
#                 quantize in {8, "w8a16"} because static activation
#                 quantization collapses the class-index output.
#   "external" -> nms=None: the raw one2many head, [1, 4+nc, anchors].
#                 Duplicate-heavy, so the consumer must run NMS.
#   "embedded" -> nms=True: ask the exporter to bake NMS into the graph.
NMS_MODES = {"free": False, "external": None, "embedded": True}


def export(
    checkpoint: Path,
    imgsz: int,
    quantize: str | None,
    out_dir: Path,
    data: str | None = None,
    nms: str = "free",
    fraction: float = 1.0,
) -> Path:
    """Export one checkpoint at one resolution. Returns the written asset path."""
    from ultralytics import YOLO  # lazy: keeps this module importable without torch

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, out_dir = checkpoint.resolve(), out_dir.resolve()
    staged = _stage(checkpoint, out_dir)

    kwargs = {"format": "litert", "imgsz": imgsz, "device": "cpu"}
    if quantize is not None:
        kwargs["quantize"] = _parse_quantize(quantize)
    if data is not None:
        kwargs["data"] = data
        kwargs["fraction"] = fraction
    if nms not in NMS_MODES:
        raise ValueError(f"nms must be one of {sorted(NMS_MODES)}, got {nms!r}")
    kwargs["nms"] = NMS_MODES[nms]

    produced = _find_tflite(YOLO(str(staged)).export(**kwargs)).resolve()
    target = export_target(checkpoint, imgsz, out_dir)
    moved = Path(shutil.move(str(produced), str(target))).resolve()
    if not wait_for_file(moved):
        try:
            listing = sorted(p.name for p in moved.parent.iterdir())
        except OSError as e:
            listing = f"<cannot list {moved.parent}: {e}>"
        try:
            moved.stat()
            why = "stat now succeeds, so the wait was the problem"
        except OSError as e:
            why = f"stat says: {e!r}"
        raise FileNotFoundError(
            f"export moved to {moved} but nothing is there. {why}. "
            f"Parent holds {listing}"
        )
    return moved


def dump_names(checkpoint: Path, out_path: Path) -> dict:
    """Write {"0": "class_name", ...} for a checkpoint. Returns the mapping."""
    from ultralytics import YOLO  # lazy

    names = {str(k): v for k, v in YOLO(str(checkpoint)).names.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(names, indent=2, sort_keys=False), encoding="utf-8")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export UBL checkpoints to LiteRT")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--quantize", help="8, w8a16, w8a32 or 32")
    p.add_argument("--data", help="calibration dataset for static quantization")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--nms", default="free", choices=sorted(NMS_MODES),
                   help="free: NMS-free head (default); external: raw head, caller runs NMS; "
                        "embedded: NMS inside the graph")
    p.add_argument("--names-only", action="store_true")
    a = p.parse_args(argv)

    a.checkpoint = a.checkpoint.resolve()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.out_dir = a.out_dir.resolve()

    if a.names_only:
        out = a.out_dir / names_filename(a.checkpoint)
        names = dump_names(a.checkpoint, out)
        print(f"{len(names)} classes -> {out}")
        return 0

    if a.imgsz is None:
        p.error("--imgsz is required unless --names-only is given")

    target = export(
        a.checkpoint, a.imgsz, a.quantize, a.out_dir,
        data=a.data, nms=a.nms, fraction=a.fraction,
    )
    size = size_of(target)
    # size_of can legitimately fail on /mnt: see _present. Do not print 0.0 MB,
    # which reads like a broken export when the file is in fact correct.
    print(f"{target} ({size / 1e6:.1f} MB)" if size else f"{target} (written; size not readable from this process)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
