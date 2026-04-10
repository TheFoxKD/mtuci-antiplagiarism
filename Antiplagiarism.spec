# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller: один исполняемый файл GUI (для демонстрации / тестирования).

Сборка из корня репозитория:
  uv sync --group build
  uv run pyinstaller Antiplagiarism.spec
"""
import sys
from pathlib import Path

# Каталог, где лежит этот .spec (корень проекта).
root = Path(SPECPATH)
src = root / "src"

block_cipher = None

# Явно подтягиваем локальные модули из src/ (импорты без пакетного префикса).
hiddenimports = [
    "document_io",
    "highlight",
    "report_pdf",
    "similarity",
    "text_normalize",
    "docx",
    "docx.opc",
    "docx.oxml",
    "pypdf",
    "reportlab",
    "rapidfuzz",
    "sklearn",
    "sklearn.utils",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
]

a = Analysis(
    [str(src / "gui_app.py")],
    pathex=[str(src)],
    binaries=[],
    # TTF для кириллицы в PDF (report_pdf._font_search_paths → sys._MEIPASS/fonts).
    datas=[
        (str(src / "fonts" / "DejaVuSans.ttf"), "fonts"),
        (str(src / "fonts" / "DejaVuSans-Bold.ttf"), "fonts"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Antiplagiarism",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    # Только macOS: передача argv в GUI onefile; на Windows должно быть False.
    argv_emulation=(sys.platform == "darwin"),
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
