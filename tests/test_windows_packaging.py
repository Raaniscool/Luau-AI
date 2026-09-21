"""Static tests for the reproducible DukeOTR Windows portable-release path.

These tests do not claim to run a Windows executable.  They keep source resource lookup and the
post-build layout verifier honest on every platform where the repository test suite runs.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop_app.core import builder_adapter
from desktop_app.resources import application_icon_path, application_resource_path
from scripts.verify_windows_release import validate_release


_ROOT = Path(__file__).resolve().parents[1]


class BundledResourceTests(unittest.TestCase):
    def test_checked_in_icon_is_a_multi_resolution_windows_ico(self) -> None:
        icon = application_icon_path()
        raw = icon.read_bytes()
        reserved, icon_type, count = struct.unpack("<HHH", raw[:6])
        self.assertEqual((reserved, icon_type), (0, 1))
        self.assertGreaterEqual(count, 6)
        dimensions: set[int] = set()
        for index in range(count):
            start = 6 + index * 16
            width, height, _colors, _reserved, planes, bit_count, byte_count, offset = struct.unpack("<BBBBHHII", raw[start : start + 16])
            dimensions.add(width or 256)
            self.assertEqual(width or 256, height or 256)
            self.assertEqual((planes, bit_count), (1, 32))
            self.assertGreater(byte_count, 8)
            self.assertEqual(raw[offset : offset + 8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue({16, 32, 48, 256}.issubset(dimensions))

    def test_source_resource_path_does_not_depend_on_current_working_directory(self) -> None:
        original_directory = Path.cwd()
        expected = application_icon_path()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                self.assertEqual(application_icon_path(), expected)
            finally:
                os.chdir(original_directory)

    def test_frozen_resource_path_uses_pyinstaller_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = Path(temporary)
            expected = bundle_root / "desktop_app" / "assets" / "dukeotr.ico"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"icon")
            with patch.object(sys, "_MEIPASS", str(bundle_root), create=True):
                self.assertEqual(application_resource_path("assets", "dukeotr.ico"), expected)

    def test_builder_mode_reads_packaged_routing_config_from_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_root = Path(temporary)
            config_destination = bundle_root / "configs" / "builder_verifier_reviewer.json"
            config_destination.parent.mkdir(parents=True)
            config_destination.write_text(
                (_ROOT / "configs" / "builder_verifier_reviewer.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with patch.object(builder_adapter.sys, "_MEIPASS", str(bundle_root), create=True):
                context = builder_adapter.plan_builder_request("Explain a simple Luau local variable.")
            self.assertNotEqual(context.level, "unavailable")
            self.assertTrue(context.selected_checks)


class PortableReleaseVerifierTests(unittest.TestCase):
    def _release_layout(self, directory: Path) -> Path:
        package = directory / "DukeOTR"
        (package / "_internal" / "desktop_app" / "assets").mkdir(parents=True)
        (package / "_internal" / "configs").mkdir(parents=True)
        (package / "DukeOTR.exe").write_bytes(b"MZ-not-a-real-executable-for-static-layout-testing")
        (package / "_internal" / "desktop_app" / "assets" / "dukeotr.ico").write_bytes(application_icon_path().read_bytes())
        (package / "_internal" / "configs" / "builder_verifier_reviewer.json").write_text("{}", encoding="utf-8")
        return package

    def test_release_layout_requires_exe_icon_and_builder_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._release_layout(Path(temporary))
            self.assertEqual(validate_release(package), [])

    def test_release_layout_refuses_accidental_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._release_layout(Path(temporary))
            (package / "_internal" / "qwen3-4b.gguf").write_bytes(b"not-a-model")
            errors = validate_release(package)
            self.assertTrue(any("Model/checkpoint artifact" in error for error in errors))

    def test_release_layout_reports_missing_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            errors = validate_release(Path(temporary) / "missing")
            self.assertEqual(len(errors), 1)
            self.assertIn("does not exist", errors[0])


class ReleaseRecipeTests(unittest.TestCase):
    def test_recipe_keeps_the_existing_tkinter_app_and_required_resources(self) -> None:
        spec = (_ROOT / "packaging" / "dukeotr.spec").read_text(encoding="utf-8")
        build_script = (_ROOT / "scripts" / "build_windows_release.ps1").read_text(encoding="utf-8")
        shortcut_script = (_ROOT / "scripts" / "create_desktop_shortcut.ps1").read_text(encoding="utf-8")

        self.assertIn("desktop_app", spec)
        self.assertIn("console=False", spec)
        self.assertIn("dukeotr.ico", spec)
        self.assertIn('"configs"', spec)
        self.assertIn("verify_windows_release.py", build_script)
        self.assertIn("import tkinter", build_script)
        self.assertIn("CreateDesktopShortcut", build_script)
        self.assertIn("WScript.Shell", shortcut_script)
        self.assertIn("DukeOTR.lnk", shortcut_script)

    def test_spec_resolves_from_its_own_directory_and_collects_only_required_app_resources(self) -> None:
        captured: dict[str, object] = {}

        class FakeAnalysis:
            def __init__(self, scripts: list[str], **kwargs: object) -> None:
                captured["analysis_scripts"] = scripts
                captured["analysis_kwargs"] = kwargs
                self.pure = "pure"
                self.scripts = "scripts"
                self.binaries = "binaries"
                self.zipfiles = "zipfiles"
                self.datas = "datas"

        def fake_pyz(*args: object, **kwargs: object) -> str:
            captured["pyz"] = (args, kwargs)
            return "pyz"

        def fake_exe(*args: object, **kwargs: object) -> str:
            captured["exe"] = (args, kwargs)
            return "exe"

        def fake_collect(*args: object, **kwargs: object) -> str:
            captured["collect"] = (args, kwargs)
            return "collect"

        spec_path = _ROOT / "packaging" / "dukeotr.spec"
        namespace = {
            "SPECPATH": str(spec_path.parent),
            "Analysis": FakeAnalysis,
            "PYZ": fake_pyz,
            "EXE": fake_exe,
            "COLLECT": fake_collect,
        }
        exec(compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec"), namespace)

        scripts = captured["analysis_scripts"]
        kwargs = captured["analysis_kwargs"]
        self.assertEqual(scripts, [str(_ROOT / "desktop_app" / "__main__.py")])
        self.assertEqual(kwargs["pathex"], [str(_ROOT)])
        self.assertIn((str(_ROOT / "configs"), "configs"), kwargs["datas"])
        self.assertIn((str(_ROOT / "desktop_app" / "assets"), "desktop_app/assets"), kwargs["datas"])
        self.assertEqual(captured["exe"][1]["name"], "DukeOTR")
        self.assertFalse(captured["exe"][1]["console"])
        self.assertEqual(captured["collect"][1]["name"], "DukeOTR")


if __name__ == "__main__":
    unittest.main()
