"""Точки входа для [project.scripts]: добавляет каталог src в sys.path."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


def main_gui() -> None:
    _ensure_src_on_path()
    import gui_app

    gui_app.main()


def main_cli() -> None:
    _ensure_src_on_path()
    import text_similarity_prototype

    text_similarity_prototype.main()
