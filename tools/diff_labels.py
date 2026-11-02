"""Compare the app's labels against a reference run, and say which kind of
difference it is.

The point is not parity. The app runs a quantized model at a different
resolution from the server, so some disagreement is expected and fine. What
matters is telling two things apart, because from inside the app they look
the same:

- **Preprocessing broken.** The letterbox, the rotation, the channel order or
  the input range is wrong. Boxes land nowhere near the reference, or nothing
  is found at all. Wrong everywhere and wrong structurally.
- **Quantization drift.** Boxes agree closely, scores sit a little lower, and
  a marginal detection occasionally changes class. Mostly right, fraying at
  the margins.

Both were seen for real while building this. Preprocessing: an antialiased
downscale changed which class won on a POSM photo. Drift: the quantized model
called a box class 13 where the reference said 16, with the box identical to
a pixel.

The reference is the same exported `.tflite` run in Python with cv2
preprocessing. Deliberately not `YOLO(...).predict()` on the `.pt`, which
runs a different head (one2many plus NMS against our NMS-free one2one) and so
disagrees for reasons that have nothing to do with the app.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Verdict(Enum):
    AGREE = "agree"
    DRIFT = "quantization drift"
    PREPROCESSING = "preprocessing bug"


@dataclass
class ImageReport:
    name: str
    matched: int
    only_reference: int
    only_app: int
    class_flips: int
    mean_iou: float
    mean_score_delta: float
    verdict: Verdict


def box_iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    iy = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    inter = ix * iy
    area_a = (a["right"] - a["left"]) * (a["bottom"] - a["top"])
    area_b = (b["right"] - b["left"]) * (b["bottom"] - b["top"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# Positional match only. Class is deliberately NOT part of matching, because a
# class flip on an agreeing box is the signal we most want to see.
MATCH_IOU = 0.5

# Below this share of the reference's detections matching positionally, the
# geometry is not merely drifting, it is wrong.
PREPROCESSING_MATCH_RATE = 0.5


def compare_image(reference: list[dict], app: list[dict]) -> ImageReport:
    """Match the app's detections to the reference by position, then judge."""
    unmatched_app = list(app)
    matched_pairs: list[tuple[dict, dict]] = []

    for ref in reference:
        best, best_iou = None, 0.0
        for cand in unmatched_app:
            iou = box_iou(ref, cand)
            if iou > best_iou:
                best, best_iou = cand, iou
        if best is not None and best_iou >= MATCH_IOU:
            matched_pairs.append((ref, best))
            unmatched_app.remove(best)

    matched = len(matched_pairs)
    only_reference = len(reference) - matched
    only_app = len(unmatched_app)
    class_flips = sum(1 for r, a in matched_pairs if r["classId"] != a["classId"])
    mean_iou = (
        sum(box_iou(r, a) for r, a in matched_pairs) / matched if matched else 0.0
    )
    mean_score_delta = (
        sum(a["score"] - r["score"] for r, a in matched_pairs) / matched
        if matched
        else 0.0
    )

    verdict = _judge(reference, app, matched, class_flips, only_app)
    return ImageReport(
        name="",
        matched=matched,
        only_reference=only_reference,
        only_app=only_app,
        class_flips=class_flips,
        mean_iou=mean_iou,
        mean_score_delta=mean_score_delta,
        verdict=verdict,
    )


def _judge(
    reference: list[dict],
    app: list[dict],
    matched: int,
    class_flips: int,
    only_app: int,
) -> Verdict:
    if not reference and not app:
        # Nothing in the picture is not evidence of anything.
        return Verdict.AGREE
    if not reference:
        # The app found things the reference did not. Not a geometry failure,
        # but not agreement either.
        return Verdict.DRIFT
    if matched / len(reference) < PREPROCESSING_MATCH_RATE:
        # Most of the reference's detections have no counterpart anywhere near
        # where they should be. A threshold cannot do that; broken geometry can.
        return Verdict.PREPROCESSING
    if matched == len(reference) and matched == len(app) and class_flips == 0:
        return Verdict.AGREE
    return Verdict.DRIFT


def _load(path: Path) -> dict[str, list[dict]]:
    """Accepts either {image: [detections]} or a single sidecar file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "detections" in data:
        return {path.stem: data["detections"]}
    return {k: v.get("detections", v) for k, v in data.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diff app labels against a reference")
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--app", required=True, type=Path)
    a = p.parse_args(argv)

    reference = _load(a.reference)
    app = _load(a.app)

    names = sorted(set(reference) | set(app))
    if not names:
        print("no images in either file")
        return 1

    reports = []
    for name in names:
        r = compare_image(reference.get(name, []), app.get(name, []))
        r.name = name
        reports.append(r)

    print(f"{'image':<42} {'match':>6} {'refOnly':>8} {'appOnly':>8} "
          f"{'flips':>6} {'IoU':>6} {'dScore':>7}  verdict")
    for r in reports:
        print(f"{r.name[:42]:<42} {r.matched:>6} {r.only_reference:>8} "
              f"{r.only_app:>8} {r.class_flips:>6} {r.mean_iou:>6.3f} "
              f"{r.mean_score_delta:>+7.3f}  {r.verdict.value}")

    worst = max(reports, key=lambda r: list(Verdict).index(r.verdict))
    print()
    print(f"images: {len(reports)}")
    for v in Verdict:
        n = sum(1 for r in reports if r.verdict is v)
        if n:
            print(f"  {v.value}: {n}")
    print(f"overall: {worst.verdict.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
