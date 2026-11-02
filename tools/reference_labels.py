"""Run the exported models over images in Python, to compare against the app.

This is the reference the app is judged against, and the choice matters.

It runs **the same `.tflite` files the app ships**, with cv2 preprocessing,
which is what Ultralytics and the server use. It deliberately does not run
`YOLO(...).predict()` on the `.pt`: that path uses the one2many head with NMS
while our export ships the NMS-free one2one head, so the two disagree for
reasons that have nothing to do with the app's correctness.

Resampling is plain `cv2.INTER_LINEAR` with no antialiasing, matching
`InputTensor` in the app. Using PIL here instead would filter on downscale
and, measured on a real POSM photo, change which class wins.

Output matches the app's sidecar format so tools/diff_labels.py can read
either side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

PAD = 114


def letterbox(rgb: np.ndarray, n: int):
    """Scale to fit, centre, pad 114 grey. The app's geometry exactly."""
    h, w = rgb.shape[:2]
    s = min(n / w, n / h)
    nw, nh = round(w * s), round(h * s)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((n, n, 3), PAD, np.uint8)
    px, py = (n - nw) // 2, (n - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, px, py


def to_nchw(canvas: np.ndarray) -> np.ndarray:
    return np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]


def detect(interp: Interpreter, image: Path, imgsz: int, threshold: float,
           allowed: set[int] | None) -> list[dict]:
    bgr = cv2.imread(str(image))
    if bgr is None:
        raise FileNotFoundError(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas, scale, px, py = letterbox(rgb, imgsz)

    interp.set_tensor(interp.get_input_details()[0]["index"], to_nchw(canvas))
    interp.invoke()
    rows = interp.get_tensor(interp.get_output_details()[0]["index"])[0]

    out = []
    for r in rows:
        score = float(r[4])
        # Rows are score-sorted, so the first below threshold ends them.
        if score < threshold:
            break
        cls = int(round(float(r[5])))
        if allowed is not None and cls not in allowed:
            continue
        out.append({
            "classId": cls,
            "score": score,
            "left": (float(r[0]) - px) / scale,
            "top": (float(r[1]) - py) / scale,
            "right": (float(r[2]) - px) / scale,
            "bottom": (float(r[3]) - py) / scale,
        })
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def qpds(seg: Interpreter, cls: Interpreter, image: Path, imgsz: int,
         seg_conf: float, cls_conf: float, cap: int,
         allowed: set[int] | None) -> list[dict]:
    """The two-stage path, mirroring QpdsPipeline and CropBuilder.

    Independent of the app's Kotlin, which is the point: it is only evidence
    if it was written separately. The geometry it mirrors is the server's
    (`core/detection.py`): mask out the background, crop the integer box,
    letterbox onto black, classify.

    The mask is deliberately built the server's way, whole and upsampled to
    the image, rather than the app's way, cell by cell within the box. Two
    routes to the same geometry is what makes a disagreement meaningful; two
    copies of one route only proves they were copied.
    """
    bgr = cv2.imread(str(image))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    canvas, scale, px, py = letterbox(rgb, imgsz)

    seg.set_tensor(seg.get_input_details()[0]["index"], to_nchw(canvas))
    seg.invoke()
    od = seg.get_output_details()
    rows = seg.get_tensor(od[0]["index"])[0]        # 300 x 38
    protos = seg.get_tensor(od[1]["index"])[0]      # 32 x P x P

    # The frame's extent inside the square, computed the way letterbox() did
    # it, so the mask can be un-letterboxed by slicing. Deriving it from the
    # padding instead would be off by one whenever the remainder is odd.
    nw, nh = round(W * scale), round(H * scale)

    out = []
    for r in rows:
        if len(out) >= cap or float(r[4]) < seg_conf:
            break
        # The server's chain, not the app's: build the whole mask, bring it to
        # full image size, then crop. The app instead computes only the cells
        # its box touches and indexes them per pixel. Same geometry by two
        # routes, which is the only reason this file is evidence. Doing it the
        # app's way here, as this once did, meant a shared misplacement of the
        # mask could not show up as a disagreement.
        mask = sigmoid(np.tensordot(r[6:], protos, axes=(0, 0)))
        in_square = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
        in_source = cv2.resize(
            in_square[py:py + nh, px:px + nw], (W, H), interpolation=cv2.INTER_NEAREST,
        )

        # box back to source pixels, truncated then clamped like the app
        sx0 = max(0, min(W, int((float(r[0]) - px) / scale)))
        sy0 = max(0, min(H, int((float(r[1]) - py) / scale)))
        sx1 = max(0, min(W, int((float(r[2]) - px) / scale)))
        sy1 = max(0, min(H, int((float(r[3]) - py) / scale)))
        if sx1 - sx0 <= 0 or sy1 - sy0 <= 0:
            continue

        crop = rgb[sy0:sy1, sx0:sx1].copy()
        ch, cw = crop.shape[:2]
        crop[in_source[sy0:sy1, sx0:sx1] < 0.5] = 0

        s2 = min(224 / cw, 224 / ch)
        rc = cv2.resize(crop, (max(1, round(cw * s2)), max(1, round(ch * s2))),
                        interpolation=cv2.INTER_LINEAR)
        square = np.zeros((224, 224, 3), np.uint8)   # black pad, not 114
        oy, ox = (224 - rc.shape[0]) // 2, (224 - rc.shape[1]) // 2
        square[oy:oy + rc.shape[0], ox:ox + rc.shape[1]] = rc

        cls.set_tensor(cls.get_input_details()[0]["index"], to_nchw(square))
        cls.invoke()
        scores = cls.get_tensor(cls.get_output_details()[0]["index"])[0]

        ids = range(len(scores)) if allowed is None else [
            i for i in allowed if i < len(scores)
        ]
        if not ids:
            continue
        best = max(ids, key=lambda i: scores[i])
        if float(scores[best]) < cls_conf:
            continue
        out.append({
            "classId": int(best),
            "score": float(scores[best]),
            "left": float(sx0), "top": float(sy0),
            "right": float(sx1), "bottom": float(sy1),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reference labels from the exported model")
    p.add_argument("--model", required=True, type=Path, help="exported .tflite")
    p.add_argument("--images", required=True, type=Path, help="directory of images")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--catalog", type=Path, help="catalog.json, to restrict classes")
    p.add_argument("--entry", help="slab or category key within the catalog")
    p.add_argument("--cls-model", type=Path, help="qpds_cls .tflite, enables two-stage")
    p.add_argument("--seg-conf", type=float, default=0.20)
    p.add_argument("--cap", type=int, default=40)
    a = p.parse_args(argv)

    allowed = None
    if a.catalog and a.entry:
        cat = json.loads(a.catalog.read_text(encoding="utf-8"))
        for group in ("posm_categories", "qpds_slabs"):
            for e in cat.get(group, []):
                if e["key"] == a.entry:
                    allowed = set(e["class_ids"])
        if allowed is None:
            print(f"entry not found in catalog: {a.entry}", file=sys.stderr)
            return 1

    interp = Interpreter(model_path=str(a.model))
    interp.allocate_tensors()

    images = sorted(
        q for q in a.images.iterdir()
        if q.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        print(f"no images in {a.images}", file=sys.stderr)
        return 1

    cls_interp = None
    if a.cls_model:
        cls_interp = Interpreter(model_path=str(a.cls_model))
        cls_interp.allocate_tensors()

    results = {}
    for image in images:
        if cls_interp is not None:
            dets = qpds(interp, cls_interp, image, a.imgsz,
                        a.seg_conf, a.threshold, a.cap, allowed)
        else:
            dets = detect(interp, image, a.imgsz, a.threshold, allowed)
        results[image.name] = {"detections": dets}
        print(f"{image.name}: {len(dets)}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
