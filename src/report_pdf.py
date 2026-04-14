"""Экспорт отчёта в PDF (reportlab)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from similarity import SimilarityMethod, SimilarityResult

# Внутренние имена после registerFont (Helvetica не содержит кириллицы — в PDF будут «квадраты»).
_REPORT_FONT = "AntiplagDejaVu"
_REPORT_FONT_BOLD = "AntiplagDejaVuBd"

_fonts_registered = False


def _font_search_paths() -> list[Path]:
    """Каталоги, где ищем DejaVu: рядом с модулем, в PyInstaller onefile — _MEIPASS/fonts."""
    out: list[Path] = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        out.append(Path(sys._MEIPASS) / "fonts")
    out.append(Path(__file__).resolve().parent / "fonts")
    return out


def _resolve_dejavu_pair() -> tuple[Path, Path]:
    """
    Пара regular + bold TTF с кириллицей.
    Сначала встроенные файлы в fonts/, затем типичные пути ОС (Linux/macOS/Windows).
    """
    names = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    for base in _font_search_paths():
        reg = base / names[0]
        bd = base / names[1]
        if reg.is_file() and bd.is_file():
            return reg, bd

    candidates_regular = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/Library/Fonts/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    candidates_bold = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    windir = Path(sys.environ.get("WINDIR", r"C:\Windows"))
    candidates_regular.extend(
        [
            windir / "Fonts" / "arial.ttf",
            windir / "Fonts" / "arialuni.ttf",
        ]
    )
    candidates_bold.append(windir / "Fonts" / "arialbd.ttf")

    for reg in candidates_regular:
        if not reg.is_file():
            continue
        for bd in candidates_bold:
            if bd.is_file():
                return reg, bd
        return reg, reg

    raise FileNotFoundError(
        "Не найдены TTF-шрифты с кириллицей (ожидались DejaVu Sans в каталоге fonts/ рядом с report_pdf.py). "
        "Проверьте, что файлы DejaVuSans.ttf и DejaVuSans-Bold.ttf на месте."
    )


def _ensure_report_fonts() -> tuple[str, str]:
    """Регистрирует шрифты в pdfmetrics один раз на процесс."""
    global _fonts_registered
    if _fonts_registered:
        return _REPORT_FONT, _REPORT_FONT_BOLD
    regular, bold = _resolve_dejavu_pair()
    pdfmetrics.registerFont(TTFont(_REPORT_FONT, str(regular)))
    pdfmetrics.registerFont(TTFont(_REPORT_FONT_BOLD, str(bold)))
    _fonts_registered = True
    return _REPORT_FONT, _REPORT_FONT_BOLD


def _paragraph_styles() -> dict:
    """Стили как в sample, но с Unicode-шрифтом для всего русскоязычного текста."""
    body, body_bd = _ensure_report_fonts()
    base = getSampleStyleSheet()
    styles: dict = {}

    def clone(name: str, **kw) -> ParagraphStyle:
        p = base[name]
        return ParagraphStyle(
            name=f"Report{name}",
            parent=p,
            fontName=kw.get("fontName", body),
            fontSize=kw.get("fontSize", p.fontSize),
            leading=kw.get("leading", p.leading),
            spaceAfter=kw.get("spaceAfter", p.spaceAfter),
            spaceBefore=kw.get("spaceBefore", p.spaceBefore),
        )

    styles["Title"] = clone("Title", fontName=body_bd, fontSize=18, leading=22, spaceAfter=12)
    styles["Normal"] = clone("Normal", fontName=body, fontSize=10, leading=14)
    styles["Heading2"] = clone("Heading2", fontName=body_bd, fontSize=14, leading=18, spaceBefore=12, spaceAfter=8)
    return styles


def _method_ru(m: SimilarityMethod) -> str:
    return {
        SimilarityMethod.JACCARD_SHINGLE: "Жаккар (фразы из слов)",
        SimilarityMethod.TFIDF_COSINE: "TF-IDF и косинус",
        SimilarityMethod.LEVENSHTEIN: "Левенштейн",
    }.get(m, m.value)


def write_report_pdf(
    output_path: Path,
    folder: Path,
    method: SimilarityMethod,
    result: SimilarityResult,
    warnings: list[tuple[str, str]],
    shingle_k: int,
    extra_note: str = "",
) -> None:
    """Пишет PDF с метаданными, таблицей и предупреждениями."""
    body, body_bd = _ensure_report_fonts()
    styles = _paragraph_stylesheet_dict()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story: list = []

    story.append(Paragraph("Отчёт: анализ сходства текстов", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Папка: {html_escape_light(str(folder))}", styles["Normal"]))
    story.append(Paragraph(f"Способ сравнения: {_method_ru(method)}", styles["Normal"]))
    story.append(Paragraph(f"Длина фразы для Жаккара (число слов подряд): {shingle_k}", styles["Normal"]))
    if extra_note:
        story.append(Paragraph(html_escape_light(extra_note), styles["Normal"]))
    story.append(Spacer(1, 16))

    headers = [
        "Документ",
        "Оригинальность, %",
        "Наибольшая близость, %",
        "Самые похожие файлы",
    ]
    data = [headers]
    n = len(result.names)
    for i in range(n):
        significant = result.significant_neighbor_indices[i]
        if significant:
            parts = [f"{result.names[j]} ({result.matrix_percent[i, j]:.2f}%)" for j in significant]
            neighbor = "; ".join(parts)
        else:
            neighbor = "нет значимых совпадений"
        data.append(
            [
                result.names[i],
                f"{result.uniqueness_percent[i]:.2f}",
                f"{result.max_similarity[i]:.2f}",
                neighbor,
            ]
        )
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), body_bd),
                ("FONTNAME", (0, 1), (-1, -1), body),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))

    if warnings:
        story.append(Paragraph("Замечания при открытии файлов", styles["Heading2"]))
        max_w = 40
        for name, reason in warnings[:max_w]:
            story.append(Paragraph(f"{html_escape_light(name)}: {html_escape_light(reason)}", styles["Normal"]))
        if len(warnings) > max_w:
            story.append(Paragraph(f"... ещё {len(warnings) - max_w} записей", styles["Normal"]))

    doc.build(story)


def _paragraph_stylesheet_dict() -> dict:
    """Имя совпадает с ключами, как у getSampleStyleSheet() — удобно для Paragraph(..., styles['Normal'])."""
    st = _paragraph_styles()
    return {"Title": st["Title"], "Normal": st["Normal"], "Heading2": st["Heading2"]}


def html_escape_light(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
