"""The catalog is the only thing the app knows about UBL's standards.

Everything the device needs about slabs, categories, expected classes and
display names is baked into one JSON at build time, so no compliance logic and
no YAML parsing ships. These tests pin what that JSON means.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_catalog import build_catalog

F = Path(__file__).resolve().parent / "fixtures"
QPDS_MIN = F / "qpds_standards_min.yaml"
POSM_MIN = F / "posm_standards_min.yaml"
NAMES_MIN = F / "names_min.json"
QPDS_CLS_NAMES_MIN = F / "qpds_cls_names_min.json"


def test_posm_category_collapses_shared_ai_class():
    # two standard names sharing one ai_class produce one class id
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, None)
    oral = next(c for c in cat["posm_categories"] if c["key"] == "oral_care")
    assert oral["class_ids"] == [12]


def test_posm_display_name_is_stable_when_names_share_a_class():
    # The model has one class, so one label is the honest answer: showing both
    # names would imply a distinction it cannot make. Alphabetically first wins
    # so the label does not change between builds.
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, None)
    oral = next(c for c in cat["posm_categories"] if c["key"] == "oral_care")
    assert oral["display"] == {"12": "Closeup LS Poster"}


def test_posm_standard_mapping_to_several_classes_contributes_all_of_them():
    """One standard, several model classes.

    A shelf talker is detected piece by piece, so `ai_class` is a list in 34 of
    the real file's 238 entries. Every listed class has to reach the allowlist,
    or the pieces the model does detect get filtered out as unexpected. All of
    them carry the one standard's name, because that is what the thing is
    called.
    """
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, None)
    skin = next(c for c in cat["posm_categories"] if c["key"] == "skin_care")
    assert 40 in skin["class_ids"] and 41 in skin["class_ids"]
    assert skin["display"]["40"] == "PS GAL Rolling Sticker Feb26"
    assert skin["display"]["41"] == "PS GAL Rolling Sticker Feb26"


def test_qpds_slab_resolves_display_names_to_ai_class():
    # shelf_types lists display names; product_mappings maps ai_class -> display name
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    slab = next(s for s in cat["qpds_slabs"] if s["key"] == "DOUBLE_SHELF")
    assert slab["display"] == {"3": "Standard Horlicks Jar", "7": "Chocolate Horlicks Jar"}


def test_size_suffixed_config_key_resolves_to_its_base_model_class():
    """`gl_mltvit_crm:50g` is model class `gl_mltvit_crm` plus a size.

    The real standards carry keys of the form `<ai_class>:<size>`, and the
    server takes the part before the colon as the class
    (`qpds_compliance.py:578`), resolving size separately by geometry. Reading
    the whole key as a class name made two real products permanently
    unlabelable while the model had the class all along.
    """
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    slab = next(s for s in cat["qpds_slabs"] if s["key"] == "SIZE_VARIANTS")
    assert slab["class_ids"] == [11]
    assert slab["missing_from_model"] == []


def test_products_collapsing_onto_one_class_are_labelled_without_the_difference():
    """Two sizes, one class, so the label must not claim a size.

    The classifier cannot tell the 50g from the 100g. Picking either name
    would put a specific weight on screen that is right half the time, so the
    label is the common part of the names.
    """
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    slab = next(s for s in cat["qpds_slabs"] if s["key"] == "SIZE_VARIANTS")
    assert slab["display"] == {"11": "Glow & Lovely Multi-Vitamin Cream"}


def test_class_expected_by_config_but_absent_from_model_is_reported():
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    slab = next(s for s in cat["qpds_slabs"] if s["key"] == "LITE")
    assert slab["missing_from_model"] == ["horlicks_lite"]
    assert slab["class_ids"] == [3]


def test_model_class_no_config_entry_mentions_is_reported():
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    assert "orphan_class" in cat["posm_categories"][0]["unmapped_model_classes"]


def test_commented_out_posm_entries_are_ignored():
    # posm_standards.yaml carries 128 commented-out ai_class lines beside the active ones
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, None)
    assert all("commented_only" not in str(c) for c in cat["posm_categories"])


def test_slab_with_every_class_missing_yields_empty_class_ids_not_an_error():
    # Review Focus 4: an empty allowlist is a reported state, not a crash
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    slab = next(s for s in cat["qpds_slabs"] if s["key"] == "ALL_MISSING")
    assert slab["class_ids"] == []
    assert len(slab["missing_from_model"]) > 0


def test_qpds_slabs_are_empty_until_the_classifier_is_exported():
    # Task 2 runs before Task 12, so the first real catalog is built with no
    # QPDS classifier names. Claiming every class is "missing from the model"
    # would be a lie about a model that does not exist yet, so the honest
    # answer is to publish no slabs at all and let the picker offer POSM only.
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, None)
    assert cat["qpds_slabs"] == []


def test_posm_subtitle_counts_the_materials_in_the_category():
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, None)
    oral = next(c for c in cat["posm_categories"] if c["key"] == "oral_care")
    assert oral["subtitle"] == "2 materials"


def test_qpds_subtitle_is_the_shelf_category():
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    slab = next(s for s in cat["qpds_slabs"] if s["key"] == "DOUBLE_SHELF")
    assert slab["subtitle"] == "Nutrition"


def test_catalog_is_json_serialisable_with_string_keyed_display():
    # the app parses this with kotlinx-serialization, which needs string keys
    cat = build_catalog(QPDS_MIN, POSM_MIN, NAMES_MIN, QPDS_CLS_NAMES_MIN)
    round_tripped = json.loads(json.dumps(cat))
    assert round_tripped == cat
    for entry in round_tripped["qpds_slabs"] + round_tripped["posm_categories"]:
        assert all(isinstance(k, str) for k in entry["display"])
        assert all(isinstance(i, int) for i in entry["class_ids"])
