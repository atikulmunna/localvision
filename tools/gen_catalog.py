"""Build the app's `catalog.json` from UBL's production standards files.

The app needs to know, for one slab or one POSM category, which model classes
to expect and what to call them. That knowledge lives in UBL-standalone's
standards YAML, tangled up with compliance rules the app has no business
carrying. So it is resolved here, at build time, into one flat JSON.

What crosses to the device: class ids, display names, and the reconciliation
warnings. What does not: YAML, thresholds, planogram order, quantities,
adherence rules, any of the compliance machinery.

Class names are resolved to model class **indices** here, where both the
config and the exported model's names are in hand. The app never maps a name
to an index, so it cannot disagree with the model about what class 12 is.

UBL-standalone is read only. This module only ever reads from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _as_list(value) -> list[str]:
    """A YAML field that is sometimes a name and sometimes a list of names."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _base_class(key: str) -> str:
    """Model class name from a config key, dropping any `:size` suffix.

    The standards carry keys like `gl_mltvit_crm:50g`, where the model class is
    `gl_mltvit_crm` and the size is resolved separately by geometry on the
    server (`qpds_compliance.py:578`). Reading the whole key as a class name
    leaves real products permanently unlabelable.
    """
    return key.split(":", 1)[0]


def _shared_label(names: list[str]) -> str:
    """One label for several product names that share a single model class.

    The model cannot distinguish them, so the label must not claim a
    distinction. The common prefix of the names is what is actually known:
    "...Cream 50g" and "...Cream 100g" become "...Cream". If the names share
    nothing useful, the first alphabetically is used, so the label at least
    stays stable between builds.
    """
    names = sorted(set(names))
    if len(names) == 1:
        return names[0]
    prefix = names[0]
    for other in names[1:]:
        while prefix and not other.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix.rstrip(" -,:/(")
    return prefix if len(prefix) >= 4 else names[0]


def _load_names(path: Path) -> dict[str, int]:
    """Read a dumped names JSON and invert it to class_name -> index."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: int(idx) for idx, name in raw.items()}


def _entry(
    key: str,
    subtitle: str,
    wanted: dict[str, list[str]],
    name_to_id: dict[str, int],
    unmapped: list[str],
) -> dict:
    """Assemble one catalog entry.

    `wanted` maps a model class name to every display name that resolves to
    it, which can be more than one: two sizes of one cream, or a January and
    a February version of one poster. Classes the model does not have are
    reported rather than dropped silently, because a label that can never
    appear otherwise looks like a detection failure.
    """
    labels: dict[int, str] = {}
    missing: list[str] = []
    for ai_class, displays in sorted(wanted.items()):
        if ai_class in name_to_id:
            labels[name_to_id[ai_class]] = _shared_label(displays)
        else:
            missing.append(ai_class)
    return {
        "key": key,
        "subtitle": subtitle,
        "class_ids": sorted(labels),
        "display": {str(i): labels[i] for i in sorted(labels)},
        "missing_from_model": missing,
        "unmapped_model_classes": unmapped,
    }


def _posm_categories(posm_yaml: Path, posm_names: Path) -> list[dict]:
    doc = yaml.safe_load(posm_yaml.read_text(encoding="utf-8")) or {}
    items = doc.get("item_mappings") or {}
    name_to_id = _load_names(posm_names)

    # category -> {ai_class: display name}. Several standard names can share
    # one ai_class, e.g. a Jan and a Feb version of the same poster. The model
    # has a single class for them, so one label is the honest answer: showing
    # both would imply a distinction it cannot make. Alphabetically first wins,
    # so the label does not drift between builds.
    by_category: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, int] = {}
    for posm_name, spec in sorted(items.items()):
        if not isinstance(spec, dict):
            continue
        # Skip DUMMY entries, exactly as the server does
        # (utils/posm_compliance.py:184, which also counts only non-DUMMY
        # names as active).
        #
        # Half the real file is these: 119 of 238 entries are a "DUMMY "
        # prefixed duplicate of a real entry, carrying the same ai_class.
        # Keeping them cost nothing in the allowlist, since every class is
        # reachable through a real name too, but it decided labels: picking
        # the alphabetically first name among those sharing a class handed
        # the placeholder the label whenever the real name sorted after D,
        # which is how a Sunsilk poster came to read "DUMMY SUNSILK RELAUNCH
        # BLACK 5PLUS1 POSTER PHASE1 AUG26" on screen.
        if posm_name.upper().startswith("DUMMY "):
            continue
        ai_class, category = spec.get("ai_class"), spec.get("category")
        if not ai_class or not category:
            continue
        counts[category] = counts.get(category, 0) + 1
        # ai_class is a single name or a list of them. A shelf talker is
        # detected piece by piece (top, left, right, bottom), so 34 of the
        # real file's 238 entries list several classes for one standard.
        # Every one has to reach the allowlist, or the pieces the model does
        # detect get filtered out as unexpected. They all carry the standard's
        # own name, because that is what the thing is called.
        for name in _as_list(ai_class):
            by_category.setdefault(category, {}).setdefault(_base_class(name), []).append(posm_name)

    claimed = {c for wanted in by_category.values() for c in wanted}
    unmapped = sorted(n for n in name_to_id if n not in claimed)

    return [
        _entry(
            category,
            f"{counts[category]} material" if counts[category] == 1 else f"{counts[category]} materials",
            wanted,
            name_to_id,
            unmapped,
        )
        for category, wanted in sorted(by_category.items())
    ]


def _qpds_slabs(qpds_yaml: Path, qpds_cls_names: Path | None) -> list[dict]:
    # No classifier export yet means no QPDS data. Reporting every class as
    # "missing from the model" would be a claim about a model that does not
    # exist, so publish nothing and let the picker offer POSM only.
    if qpds_cls_names is None:
        return []

    doc = yaml.safe_load(qpds_yaml.read_text(encoding="utf-8")) or {}
    shelf_types = doc.get("shelf_types") or {}
    categories = doc.get("shelf_categories") or {}
    # The real file maps ai_class -> display name, while a slab lists display
    # names, so this has to be inverted to go the way the slabs need.
    # Note the direction: the real file is ai_class -> display name, while a
    # slab lists display names, so this is inverted. Keys keep their `:size`
    # suffix here; _base_class strips it at the point of use.
    display_to_class = {v: k for k, v in (doc.get("product_mappings") or {}).items()}
    name_to_id = _load_names(qpds_cls_names)

    wanted_by_slab: dict[str, dict[str, list[str]]] = {}
    for slab, spec in shelf_types.items():
        wanted: dict[str, list[str]] = {}
        for product in (spec or {}).get("products") or []:
            display = (product or {}).get("product")
            if not display:
                continue
            config_key = display_to_class.get(display)
            # A product with no product_mappings entry cannot be named by any
            # class, so it is reported under its display name.
            ai_class = _base_class(config_key) if config_key else f"<unmapped product: {display}>"
            wanted.setdefault(ai_class, []).append(display)
        wanted_by_slab[slab] = wanted

    claimed = {c for wanted in wanted_by_slab.values() for c in wanted}
    unmapped = sorted(n for n in name_to_id if n not in claimed)

    return [
        _entry(slab, categories.get(slab, "Uncategorised"), wanted, name_to_id, unmapped)
        for slab, wanted in sorted(wanted_by_slab.items())
    ]


def build_catalog(
    qpds_yaml: Path,
    posm_yaml: Path,
    posm_names: Path,
    qpds_cls_names: Path | None,
) -> dict:
    """The whole catalog: every slab, every POSM category, every class name.

    `model_classes` carries the exported models' own class names, for classes
    no entry expects. With the allowlist off the app deliberately shows those
    classes, and without their names it printed "class 12" and "class 54" at
    100% confidence: the model being right and the app unable to say so.
    """
    model_classes = {
        "posm": json.loads(posm_names.read_text(encoding="utf-8")),
    }
    if qpds_cls_names is not None:
        model_classes["qpds_cls"] = json.loads(
            qpds_cls_names.read_text(encoding="utf-8"),
        )

    return {
        "qpds_slabs": _qpds_slabs(qpds_yaml, qpds_cls_names),
        "posm_categories": _posm_categories(posm_yaml, posm_names),
        "model_classes": model_classes,
    }


def describe(catalog: dict) -> str:
    """Human-readable reconciliation summary.

    Printed at generation time because a mismatch is expected, not alarming:
    new SKUs ship in config before a model can detect them. It needs to be
    visible and recorded, not chased.
    """
    lines: list[str] = []
    for group in ("qpds_slabs", "posm_categories"):
        entries = catalog[group]
        lines.append(f"{group}: {len(entries)} entries")
        for e in entries:
            flags = []
            if not e["class_ids"]:
                flags.append("NO RESOLVABLE CLASSES")
            if e["missing_from_model"]:
                flags.append(f"missing: {', '.join(e['missing_from_model'])}")
            if flags:
                lines.append(f"  {e['key']}: {' | '.join(flags)}")
        if entries and entries[0]["unmapped_model_classes"]:
            unmapped = entries[0]["unmapped_model_classes"]
            lines.append(f"  model classes no entry mentions ({len(unmapped)}): "
                         f"{', '.join(unmapped)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build catalog.json from UBL standards")
    p.add_argument("--qpds", required=True, type=Path, help="qpds_standards.yaml")
    p.add_argument("--posm", required=True, type=Path, help="posm_standards.yaml")
    p.add_argument("--posm-names", required=True, type=Path)
    p.add_argument("--qpds-cls-names", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)

    catalog = build_catalog(a.qpds, a.posm, a.posm_names, a.qpds_cls_names)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(describe(catalog))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
