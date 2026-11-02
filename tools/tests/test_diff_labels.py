"""The diff's job is to tell two kinds of difference apart.

Per the spec, the point of comparing the app against a reference is not
parity. It is that a preprocessing bug and a quantization loss have different
signatures, and from inside the app they look identical:

- **Preprocessing broken**: boxes land in the wrong places, or nothing matches
  at all. Wrong everywhere and wrong structurally.
- **Quantization drift**: boxes agree, scores wobble, and a marginal
  detection occasionally changes class. Mostly right, fraying at the edges.

These tests pin that classification, because getting it backwards sends
someone hunting a letterbox bug that does not exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_labels import Verdict, box_iou, compare_image


def det(left, top, right, bottom, cls, score=0.9):
    return {"left": left, "top": top, "right": right, "bottom": bottom,
            "classId": cls, "score": score}


def test_identical_boxes_have_iou_one():
    assert box_iou(det(0, 0, 10, 10, 1), det(0, 0, 10, 10, 1)) == 1.0


def test_disjoint_boxes_have_iou_zero():
    assert box_iou(det(0, 0, 10, 10, 1), det(50, 50, 60, 60, 1)) == 0.0


def test_half_overlap_iou():
    # two 10x10 boxes sharing a 5x10 strip: 50 / 150
    assert abs(box_iou(det(0, 0, 10, 10, 1), det(5, 0, 15, 10, 1)) - (50 / 150)) < 1e-6


def test_exact_agreement_is_reported_as_agreeing():
    r = compare_image([det(0, 0, 10, 10, 3)], [det(0, 0, 10, 10, 3)])
    assert r.verdict is Verdict.AGREE
    assert r.matched == 1
    assert r.only_reference == 0 and r.only_app == 0


def test_same_box_different_class_is_quantization_drift_not_a_bug():
    # The measured case: the quantized model called one box class 13 where the
    # reference said 16, with the box identical to a pixel. That is drift.
    r = compare_image([det(0, 0, 10, 10, 16)], [det(0, 0, 10, 10, 13)])
    assert r.verdict is Verdict.DRIFT
    assert r.class_flips == 1


def test_a_lower_score_on_an_identical_detection_is_reported_but_still_agreement():
    """Scores always sit lower under quantization. That is not a difference
    in labels.

    The contract records 0.468 against 0.574 on the same box. If any score
    wobble counted as a disagreement the verdict would never read AGREE for a
    quantized model, which would make it useless for spotting the real
    failures. The shift is reported as a number instead, so it stays visible
    without crying wolf.
    """
    r = compare_image([det(0, 0, 10, 10, 3, 0.80)], [det(0, 0, 10, 10, 3, 0.55)])
    assert r.verdict is Verdict.AGREE
    assert abs(r.mean_score_delta + 0.25) < 1e-6


def test_boxes_in_the_wrong_place_are_a_preprocessing_bug():
    # Nothing matches positionally. A score threshold cannot produce this;
    # a letterbox, rotation or channel-order error can.
    r = compare_image([det(0, 0, 10, 10, 3)], [det(200, 200, 210, 210, 3)])
    assert r.verdict is Verdict.PREPROCESSING


def test_finding_nothing_at_all_is_a_preprocessing_bug():
    r = compare_image([det(0, 0, 10, 10, 3), det(20, 20, 30, 30, 4)], [])
    assert r.verdict is Verdict.PREPROCESSING


def test_missing_only_a_marginal_detection_is_drift_not_a_bug():
    # Three of four found, the lost one the weakest. That is a threshold
    # sitting slightly differently, not broken geometry.
    reference = [det(0, 0, 10, 10, 1, 0.9), det(20, 20, 30, 30, 2, 0.9),
                 det(40, 40, 50, 50, 3, 0.9), det(60, 60, 70, 70, 4, 0.31)]
    app = reference[:3]
    r = compare_image(reference, app)
    assert r.verdict is Verdict.DRIFT
    assert r.only_reference == 1


def test_both_empty_agrees():
    # A frame with nothing in it is not evidence of a bug.
    assert compare_image([], []).verdict is Verdict.AGREE


def test_app_inventing_detections_the_reference_never_saw_is_flagged():
    r = compare_image([], [det(0, 0, 10, 10, 3), det(20, 20, 30, 30, 4)])
    assert r.only_app == 2
    assert r.verdict is not Verdict.AGREE


def test_matching_prefers_the_best_overlapping_box():
    # Two candidates overlap the reference box; the closer one must win, or
    # the report blames the wrong detection for a class flip.
    reference = [det(0, 0, 10, 10, 5)]
    app = [det(6, 0, 16, 10, 9), det(0, 0, 10, 10, 5)]
    r = compare_image(reference, app)
    assert r.matched == 1
    assert r.class_flips == 0
