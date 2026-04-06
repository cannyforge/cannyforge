"""Tests for skill export/import portability helpers."""

import json
import zipfile

from cannyforge.core import CannyForge
from cannyforge.corrections import Correction


def test_export_skill_writes_bundle_with_exportable_corrections(tmp_path):
    forge = CannyForge(data_dir=tmp_path / "source")
    forge.reset()

    exportable = Correction(
        id="corr_keep",
        skill_name="test_skill",
        error_type="WrongToolError",
        content="Keep this correction.",
        source_errors=["e1"],
        created_at=1.0,
        times_injected=5,
        times_effective=3,
    )
    filtered = Correction(
        id="corr_drop",
        skill_name="test_skill",
        error_type="WrongToolError",
        content="Drop this correction.",
        source_errors=["e2"],
        created_at=1.0,
        times_injected=10,
        times_effective=2,
    )
    forge.knowledge_base.add_correction("test_skill", exportable)
    forge.knowledge_base.add_correction("test_skill", filtered)

    bundle_path = tmp_path / "exports" / "test_skill.cannyforge"
    forge.export_skill("test_skill", str(bundle_path))

    assert bundle_path.exists()
    with zipfile.ZipFile(bundle_path, "r") as bundle:
        assert {"manifest.json", "corrections.json"}.issubset(set(bundle.namelist()))
        manifest = json.loads(bundle.read("manifest.json"))
        corrections = json.loads(bundle.read("corrections.json"))

    assert manifest["skill_name"] == "test_skill"
    assert manifest["correction_count"] == 1
    assert [entry["id"] for entry in corrections] == ["corr_keep"]


def test_import_skill_loads_corrections_and_resets_usage(tmp_path):
    source = CannyForge(data_dir=tmp_path / "source")
    source.reset()

    correction = Correction(
        id="corr_keep",
        skill_name="test_skill",
        error_type="WrongToolError",
        content="Keep this correction.",
        source_errors=["e1"],
        created_at=1.0,
        times_injected=5,
        times_effective=4,
    )
    source.knowledge_base.add_correction("test_skill", correction)

    bundle_path = tmp_path / "exports" / "test_skill.cannyforge"
    source.export_skill("test_skill", str(bundle_path))

    target = CannyForge(data_dir=tmp_path / "target")
    target.reset()
    imported = target.import_skill(str(bundle_path))

    imported_corrections = target.knowledge_base.get_corrections("test_skill")
    assert imported == 1
    assert len(imported_corrections) == 1
    assert imported_corrections[0].times_injected == 0
    assert imported_corrections[0].times_effective == 0