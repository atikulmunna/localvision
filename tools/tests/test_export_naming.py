"""Asset naming is a contract, not a convenience.

The Android app loads models by exact asset path (`models/posm_960.tflite`),
so the name the exporter writes and the name the app opens have to be derived
from one place. These tests pin that derivation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from export_litert import (
    asset_name,
    export_target,
    names_filename,
    size_of,
    tflite_filename,
    wait_for_file,
)


def test_posm_checkpoint_maps_to_posm_asset_name():
    assert asset_name(Path("/x/models/POSM_YOLO11X.pt")) == "posm"


def test_qpds_seg_checkpoint_maps_to_qpds_seg():
    assert asset_name(Path("/x/models/QPDS-seg.pt")) == "qpds_seg"


def test_qpds_cls_checkpoint_maps_to_qpds_cls():
    assert asset_name(Path("/x/models/QPDS-cls.pt")) == "qpds_cls"


def test_unmapped_checkpoint_falls_back_to_a_sanitised_stem():
    assert asset_name(Path("/x/models/SOS-Detection.pt")) == "sos_detection"


def test_tflite_filename_carries_the_resolution():
    assert tflite_filename(Path("/x/models/POSM_YOLO11X.pt"), 960) == "posm_960.tflite"
    assert tflite_filename(Path("/x/models/QPDS-seg.pt"), 640) == "qpds_seg_640.tflite"


def test_names_filename_uses_the_checkpoint_stem_not_the_asset_name():
    # Task 1 Step 4 verifies exports/POSM_YOLO11X.names.json by that exact path
    assert names_filename(Path("/x/models/POSM_YOLO11X.pt")) == "POSM_YOLO11X.names.json"


def test_export_target_is_absolute_even_from_a_relative_out_dir():
    # A relative target is resolved against the working directory at the moment
    # it is used. Ultralytics runs a long conversion between our computing the
    # path and our reading it back, so every path crossing that call is absolute.
    t = export_target(Path("POSM_YOLO11X.pt"), 640, Path("../exports"))
    assert t.is_absolute()
    assert t.name == "posm_640.tflite"


def test_export_target_keeps_an_already_absolute_out_dir():
    t = export_target(Path("POSM_YOLO11X.pt"), 960, Path("/tmp/out"))
    assert t == Path("/tmp/out/posm_960.tflite").resolve()


def test_wait_for_file_sees_a_file_that_is_already_there(tmp_path):
    f = tmp_path / "there.bin"
    f.write_bytes(b"x")
    assert wait_for_file(f, timeout=0.5) is True


def test_wait_for_file_gives_up_on_a_file_that_never_appears(tmp_path):
    # On the /mnt DrvFs mount a large file can be invisible to stat for a moment
    # after shutil.move returns. A bounded wait absorbs that without hiding a
    # genuinely failed move, which still returns False.
    assert wait_for_file(tmp_path / "never.bin", timeout=0.2) is False


def test_wait_for_file_trusts_the_directory_listing_when_stat_lies(tmp_path, monkeypatch):
    """The measured DrvFs failure: stat says ENOENT while iterdir shows the file.

    Observed exporting a 57 MB model to /mnt/m. In one process, at one moment,
    `moved.parent.iterdir()` listed `posm_640.tflite` and `moved.stat()` raised
    FileNotFoundError for that exact path. A fresh process stat'd it fine. So
    the check cannot rest on stat alone.
    """
    f = tmp_path / "moved.bin"
    f.write_bytes(b"x" * 4096)

    real_stat = Path.stat

    def blind_stat(self, *a, **kw):
        if self.name == "moved.bin":
            raise FileNotFoundError(2, "No such file or directory")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", blind_stat)
    assert wait_for_file(f, timeout=0.3) is True


def test_wait_for_file_accepts_a_listed_file_whose_size_is_unknowable(tmp_path, monkeypatch):
    """The real DrvFs case: the directory entry exists and nothing can stat it.

    Both `Path.stat` and `DirEntry.stat` fail. Being listed by the directory
    is then the only evidence available, and it is enough: the move reported
    success and the entry is there. Treating this as absent fails an export
    whose output is in fact correct, which is what happened before this test.
    """
    f = tmp_path / "listed.bin"
    f.write_bytes(b"x" * 4096)

    class BlindEntry:
        name = "listed.bin"

        def stat(self, *a, **kw):
            raise FileNotFoundError(2, "No such file or directory")

    class BlindScandir:
        def __enter__(self):
            return [BlindEntry()]

        def __exit__(self, *a):
            return False

    real_stat = Path.stat

    def blind_stat(self, *a, **kw):
        if self.name == "listed.bin":
            raise FileNotFoundError(2, "No such file or directory")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", blind_stat)
    monkeypatch.setattr("export_litert.os.scandir", lambda p: BlindScandir())

    assert wait_for_file(f, timeout=0.3) is True


def test_size_of_reports_the_size(tmp_path):
    f = tmp_path / "sized.bin"
    f.write_bytes(b"x" * 1234)
    assert size_of(f) == 1234


def test_size_of_falls_back_to_the_listing_when_stat_lies(tmp_path, monkeypatch):
    # same DrvFs lie as above, hit on the way to printing the exported size
    f = tmp_path / "sized.bin"
    f.write_bytes(b"x" * 1234)
    real_stat = Path.stat

    def blind_stat(self, *a, **kw):
        if self.name == "sized.bin":
            raise FileNotFoundError(2, "No such file or directory")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", blind_stat)
    assert size_of(f) == 1234


def test_size_of_returns_zero_for_a_missing_file(tmp_path):
    assert size_of(tmp_path / "absent.bin") == 0


def test_wait_for_file_rejects_an_empty_file(tmp_path):
    # a zero-byte file means the move or write did not really land
    f = tmp_path / "empty.bin"
    f.touch()
    assert wait_for_file(f, timeout=0.2) is False


def test_importing_the_module_does_not_require_ultralytics():
    # the tooling must be importable on a machine with no torch, so the naming
    # contract stays testable without a multi-gigabyte install
    assert "ultralytics" not in sys.modules
