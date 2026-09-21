# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the portable Windows DukeOTR desktop release.

Run this spec only on the target Windows architecture.  PyInstaller does not cross-compile a
Windows executable from Linux/macOS; the accompanying PowerShell script creates a clean local
build virtual environment and invokes this file reproducibly.
"""

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve().parent
ICON = PROJECT_ROOT / "desktop_app" / "assets" / "dukeotr.ico"
VERSION_INFO = PROJECT_ROOT / "packaging" / "dukeotr_version_info.txt"


a = Analysis(
    [str(PROJECT_ROOT / "desktop_app" / "__main__.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        # Builder mode reads only this existing deterministic routing configuration.  It does
        # not package training data, evaluation material, reports, or any Ollama model files.
        (str(PROJECT_ROOT / "configs"), "configs"),
        # The runtime window icon is a resource, separate from the executable icon below.
        (str(PROJECT_ROOT / "desktop_app" / "assets"), "desktop_app/assets"),
    ],
    hiddenimports=[
        "tkinter",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "tkinter.ttk",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DukeOTR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ICON),
    version=str(VERSION_INFO),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DukeOTR",
)
