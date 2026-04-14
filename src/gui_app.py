"""Десктопный GUI: папка, три метода, таблица, подсветка, PDF."""

from __future__ import annotations

import html as html_module
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from document_io import CorpusEntry, load_corpus
from highlight import (
    highlight_jaccard_multi_with_legend,
    highlight_levenshtein_multi_with_legend,
    highlight_tfidf_multi_with_legend,
    html_document_body,
)
from report_pdf import write_report_pdf
from similarity import SimilarityMethod, SimilarityResult, compute_similarity

_LOG = logging.getLogger("antiplagiarism.gui")


def _debug_logging_enabled() -> bool:
    return os.environ.get("ANTIPLAGIAT_DEBUG", "").strip() in ("1", "true", "yes", "on")


def _configure_debug_log() -> None:
    if not _debug_logging_enabled():
        return
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(message)s",
    )


# Подписи методов для интерфейса: короткое имя в списке + всплывающая подсказка.
METHOD_OPTIONS: list[tuple[SimilarityMethod, str, str]] = [
    (
        SimilarityMethod.JACCARD_SHINGLE,
        "Жаккар (фразы из слов)",
        "Одинаковые короткие цепочки слов.",
    ),
    (
        SimilarityMethod.TFIDF_COSINE,
        "TF-IDF и косинус",
        "Близость по значимым словам.",
    ),
    (
        SimilarityMethod.LEVENSHTEIN,
        "Левенштейн (весь текст)",
        "Сравнение по полному нормализованному тексту как по одной длинной строке символов.",
    ),
]

# Только поле ввода и список: без ::drop-down / ::up-button / ::*-arrow.
# На macOS + Fusion кастом этих подконтролов ломает отрисовку (пусто или полоска).
_COMBO_FIELD_STYLE = """
QComboBox {
    padding: 6px 32px 6px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background-color: #ffffff;
    color: #0f172a;
    min-height: 28px;
}
QComboBox:hover { border-color: #94a3b8; }
QComboBox:focus { border-color: #0d9488; }
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #94a3b8;
    outline: none;
    selection-background-color: #99f6e4;
    selection-color: #042f2e;
    padding: 2px;
}
QComboBox QAbstractItemView::item {
    min-height: 26px;
    padding: 6px 10px;
    color: #0f172a;
    background-color: #ffffff;
}
QComboBox QAbstractItemView::item:hover { background-color: #ecfeff; }
QComboBox QAbstractItemView::item:selected {
    background-color: #5eead4;
    color: #042f2e;
}
"""

_SPIN_FIELD_STYLE = """
QSpinBox {
    padding: 4px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background-color: #ffffff;
    color: #0f172a;
    min-height: 28px;
    font-size: 13px;
}
QSpinBox:hover { border-color: #94a3b8; }
QSpinBox:focus { border-color: #0d9488; }
"""


def _application_stylesheet() -> str:
    return """
    QMainWindow { background-color: #eef2f7; }
    QWidget#centralRoot { background-color: #eef2f7; }
    QFrame#headerBar {
        background-color: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 12px;
    }
    QLabel#appTitle {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
    }
    QLabel#appSubtitle {
        font-size: 13px;
        color: #64748b;
    }
    QPushButton#primaryBtn {
        background-color: #0d9488;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        min-height: 20px;
    }
    QPushButton#primaryBtn:hover { background-color: #14b8a6; }
    QPushButton#primaryBtn:pressed { background-color: #0f766e; }
    QPushButton#primaryBtn:disabled { background-color: #94a3b8; color: #f1f5f9; }
    QPushButton#secondaryBtn {
        background-color: #ffffff;
        color: #334155;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }
    QPushButton#secondaryBtn:hover { background-color: #f8fafc; border-color: #94a3b8; }
    QPushButton#secondaryBtn:disabled { color: #94a3b8; border-color: #e2e8f0; background: #f8fafc; }
    QPushButton#ghostBtn {
        background-color: transparent;
        color: #64748b;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 16px;
    }
    QPushButton#ghostBtn:hover { background-color: #f1f5f9; color: #475569; }
    QLabel { color: #334155; }
    QProgressBar {
        border: 2px solid #dbe4f0;
        border-radius: 8px;
        background: #ffffff;
        text-align: center;
        height: 22px;
    }
    QProgressBar::chunk {
        background-color: #0d9488;
        border-radius: 5px;
    }
    QTableWidget {
        background-color: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 10px;
        gridline-color: #e8eef4;
        selection-background-color: #ccfbf1;
        selection-color: #0f172a;
    }
    QTableWidget::item:alternate {
        background-color: #f8fafc;
    }
    QHeaderView::section {
        background-color: #e0f2f1;
        color: #134e4a;
        padding: 10px 8px;
        border: none;
        border-bottom: 2px solid #99f6e4;
        font-weight: 600;
        font-size: 12px;
    }
    QCheckBox { spacing: 8px; color: #334155; }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid #cbd5e1;
        background: #ffffff;
    }
    QCheckBox::indicator:checked {
        background-color: #0d9488;
        border-color: #0f766e;
    }
    QTextEdit {
        border: 1px solid #dbe4f0;
        border-radius: 8px;
        background: #ffffff;
    }
    QDialog { background-color: #eef2f7; }
    """


def _similarity_method_from_combo(combo: QComboBox) -> SimilarityMethod | None:
    """
    QComboBox.currentData() для Enum иногда отдаёт str (как у SimilarityMethod(str, Enum)),
    а не сам enum — приводим к SimilarityMethod.
    """
    data = combo.currentData()
    if isinstance(data, SimilarityMethod):
        return data
    if isinstance(data, str):
        try:
            return SimilarityMethod(data)
        except ValueError:
            return None
    return None


def _highlight_neighbor_indices(result: SimilarityResult, row: int, k: int) -> list[int]:
    """
    Соседи для окна подсветки: значимые источники (>= порога) + при пустом списке fallback.
    Всегда возвращаем как минимум одного соседа, чтобы окно не было пустым.
    """
    n = len(result.names)
    pairs: list[tuple[float, int]] = []
    for j in range(n):
        if j == row:
            continue
        pairs.append((result.matrix_percent[row, j], j))
    pairs.sort(key=lambda x: -x[0])
    significant_n = len(result.significant_neighbor_indices[row])
    # Если значимых нет — показываем хотя бы одного лучшего для наглядности.
    need = max(1, k if significant_n == 0 else max(k, significant_n))
    return [j for _, j in pairs[:need]]


class SimilarityWorker(QThread):
    progress = Signal(int, int)
    status_text = Signal(str)
    finished_ok = Signal(object)
    finished_err = Signal(str)

    def __init__(
        self, folder: Path, method: SimilarityMethod, shingle_k: int, recursive: bool
    ) -> None:
        super().__init__()
        self.folder = folder
        self.method = method
        self.shingle_k = shingle_k
        self.recursive = recursive
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            _LOG.debug("worker start folder=%s method=%s recursive=%s", self.folder, self.method, self.recursive)
            self.status_text.emit("Чтение файлов из папки…")
            loaded = load_corpus(self.folder, recursive=self.recursive)
            _LOG.debug(
                "load_corpus: entries=%s warnings=%s",
                len(loaded.entries),
                len(loaded.warnings),
            )
            if self._cancel:
                self.finished_err.emit("Остановлено до начала сравнения.")
                return
            if len(loaded.entries) < 2:
                self.finished_err.emit(
                    "Нужно не меньше двух подходящих документов. "
                    "Проверьте формат (.txt, .md, .docx, .pdf с текстом) и папку."
                )
                return
            names = [e.name for e in loaded.entries]
            texts = [e.normalized for e in loaded.entries]

            self.status_text.emit("Сравнение документов… Это может занять время.")

            def prog(done: int, total: int) -> None:
                self.progress.emit(done, total)

            sim = compute_similarity(
                names,
                texts,
                self.method,
                shingle_size=self.shingle_k,
                is_cancelled=self._is_cancelled,
                progress=prog,
            )
            if self._cancel:
                self.finished_err.emit("Остановлено по кнопке «Стоп».")
                return
            self.finished_ok.emit(
                {
                    "entries": loaded.entries,
                    "warnings": loaded.warnings,
                    "result": sim,
                    "method": self.method,
                    "shingle_k": self.shingle_k,
                    "folder": self.folder,
                }
            )
        except InterruptedError:
            self.finished_err.emit("Остановлено.")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _LOG.exception("worker failed")
            self.finished_err.emit(
                f"Не удалось выполнить сравнение.\n\n{exc}\n\n"
                "Подробный отчёт об ошибке выводится в терминал, если программа запущена из него."
            )


class HighlightDialog(QDialog):
    def __init__(
        self,
        entries: list[CorpusEntry],
        corpus_norm: list[str],
        result: SimilarityResult,
        row: int,
        method: SimilarityMethod,
        shingle_k: int,
        neighbor_k: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Совпадающие фрагменты")
        self.resize(920, 540)
        neigh_idx = _highlight_neighbor_indices(result, row, neighbor_k)
        f = entries[row]
        neigh_names = [result.names[j] for j in neigh_idx]
        names_line = ", ".join(neigh_names)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        cap = QLabel(f"<b>{f.name}</b> — сравнение с: {names_line}")
        cap.setWordWrap(True)
        cap.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(cap)

        neighbors_norm = [entries[j].normalized for j in neigh_idx]

        if method == SimilarityMethod.JACCARD_SHINGLE:
            hl = highlight_jaccard_multi_with_legend(
                f.raw_text, f.normalized, neighbors_norm, shingle_k, neigh_names
            )
        elif method == SimilarityMethod.TFIDF_COSINE:
            hl = highlight_tfidf_multi_with_legend(
                f.raw_text, corpus_norm, row, neigh_idx, neigh_names, top_n=30
            )
        else:
            hl = highlight_levenshtein_multi_with_legend(
                f.normalized, neighbors_norm, neigh_names
            )

        te_f = QTextEdit()
        te_f.setReadOnly(True)
        te_f.setHtml(html_document_body(hl.inner_html))

        leg_bits = " ".join(
            f'<span style="background-color:{col}; padding:2px 8px; margin-right:6px; border-radius:3px">'
            f"{html_module.escape(name)}</span>"
            for name, col in hl.legend
        )
        leg = QLabel(f"<b>Цвета:</b> {leg_bits}")
        leg.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(leg)

        te_g = QTextEdit()
        te_g.setReadOnly(True)
        # показываем первого соседа целиком
        g = entries[neigh_idx[0]]
        te_g.setPlainText(g.raw_text)

        row_h = QHBoxLayout()
        lh = QLabel("<b>Этот файл</b>")
        lh.setTextFormat(Qt.TextFormat.RichText)
        rh = QLabel(f"<b>Файл для сравнения:</b> {html_module.escape(g.name)}")
        rh.setTextFormat(Qt.TextFormat.RichText)
        row_h.addWidget(lh)
        row_h.addStretch()
        row_h.addWidget(rh)
        layout.addLayout(row_h)
        panes = QHBoxLayout()
        panes.setSpacing(12)
        panes.addWidget(te_f)
        panes.addWidget(te_g)
        layout.addLayout(panes)
        if method == SimilarityMethod.LEVENSHTEIN:
            layout.addWidget(
                QLabel(
                    "Для метода «Левенштейн» текст показан в едином виде: "
                    "регистр и знаки приведены к одному формату, как при расчёте."
                )
            )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Сходство текстов — анализ документов")
        self.resize(1000, 620)

        _configure_debug_log()

        self._folder: Path | None = None
        self._bundle: dict | None = None
        self._worker: SimilarityWorker | None = None

        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("headerBar")
        head_layout = QVBoxLayout(header)
        head_layout.setContentsMargins(18, 16, 18, 16)
        head_layout.setSpacing(6)
        title_lbl = QLabel("Сходство текстов")
        title_lbl.setObjectName("appTitle")
        sub_lbl = QLabel("Папка → «Сравнить». Двойной щелчок по строке — выделение совпадений.")
        sub_lbl.setObjectName("appSubtitle")
        sub_lbl.setWordWrap(True)
        head_layout.addWidget(title_lbl)
        head_layout.addWidget(sub_lbl)
        root.addWidget(header)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        self.btn_folder = QPushButton("Выбрать папку…")
        self.btn_folder.setObjectName("secondaryBtn")
        self.btn_folder.clicked.connect(self._pick_folder)
        self.lbl_folder = QLabel("Папка не выбрана")
        self.lbl_folder.setStyleSheet("color: #64748b;")
        row1.addWidget(self.btn_folder)
        row1.addWidget(self.lbl_folder, stretch=1)
        self.chk_recursive = QCheckBox("Включая подпапки")
        self.chk_recursive.setToolTip("Обход вложенных папок; в таблице — путь от выбранной папки.")
        row1.addWidget(self.chk_recursive)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(12)
        row2.addWidget(QLabel("Способ:"))
        self.combo_method = QComboBox()
        self.combo_method.setMinimumWidth(260)
        self.combo_method.setStyleSheet(_COMBO_FIELD_STYLE)
        tip_role = Qt.ItemDataRole.ToolTipRole
        for method, label, tip in METHOD_OPTIONS:
            self.combo_method.addItem(label, method.value)
            idx = self.combo_method.count() - 1
            self.combo_method.setItemData(idx, tip, tip_role)
        row2.addWidget(self.combo_method)
        row2.addWidget(QLabel("Фраза, слов:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 8)
        self.spin_k.setValue(2)
        self.spin_k.setToolTip("Для Жаккара: сколько слов подряд в одной фразе.")
        self.spin_k.setStyleSheet(_SPIN_FIELD_STYLE)
        row2.addWidget(self.spin_k)
        row2.addWidget(QLabel("Файлов в подсветке:"))
        self.spin_K = QSpinBox()
        self.spin_K.setRange(1, 10)
        self.spin_K.setValue(1)
        self.spin_K.setToolTip(
            "Минимум столько соседей в окне подсветки. Если есть несколько значимых источников (>= 20%), "
            "в подсветку попадут все они, даже при значении 1."
        )
        self.spin_K.setStyleSheet(_SPIN_FIELD_STYLE)
        row2.addWidget(self.spin_K)
        row2.addStretch()
        root.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(10)
        self.btn_run = QPushButton("Сравнить документы")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("Стоп")
        self.btn_cancel.setObjectName("ghostBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_pdf = QPushButton("Сохранить отчёт PDF…")
        self.btn_pdf.setObjectName("secondaryBtn")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self._export_pdf)
        row3.addWidget(self.btn_run)
        row3.addWidget(self.btn_cancel)
        row3.addWidget(self.btn_pdf)
        row3.addStretch()
        root.addLayout(row3)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [
                "Документ",
                "Оригинальность, %",
                "Наибольшая близость, %",
                "Самые похожие файлы",
            ]
        )
        _hdr = self.table.horizontalHeader()
        _hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        _tips = [
            "Имя файла (путь от выбранной папки, если есть вложенность).",
            "100% минус максимальную близость к другому файлу.",
            "Самая высокая похожесть с любым другим документом.",
            "Все соседи с похожестью не ниже 20%; для каждого указан его процент.",
        ]
        for _c, _t in enumerate(_tips):
            _hi = self.table.horizontalHeaderItem(_c)
            if _hi is not None:
                _hi.setToolTip(_t)
        self.table.doubleClicked.connect(self._on_table_double_click)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table)

        # Строка «подозрительные»: у документа несколько значимых источников (>= 20%).
        self._suspicious = QLabel("")
        self._suspicious.setWordWrap(True)
        self._suspicious.setStyleSheet("color: #92400e; padding: 2px 0 0 0; font-size: 12px;")
        root.addWidget(self._suspicious)

        self._status = QLabel("Форматы: txt, md, docx, pdf с текстом.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #475569; padding-top: 4px; font-size: 13px;")
        root.addWidget(self._status)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Папка с документами для сравнения")
        if d:
            self._folder = Path(d)
            self.lbl_folder.setText(str(self._folder))
            self.lbl_folder.setStyleSheet("color: #334155;")

    def _run(self) -> None:
        if not self._folder or not self._folder.is_dir():
            QMessageBox.warning(
                self,
                "Нужна папка",
                "Выберите папку, в которой лежат файлы для сравнения.",
            )
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Подождите",
                "Сравнение уже выполняется. Дождитесь окончания или нажмите «Стоп».",
            )
            return
        method = _similarity_method_from_combo(self.combo_method)
        if method is None:
            QMessageBox.warning(
                self,
                "Сбой настроек",
                "Не удалось определить способ сравнения. Перезапустите программу.",
            )
            _LOG.error("combo_method.currentData()=%r", self.combo_method.currentData())
            return
        k = self.spin_k.value()
        try:
            self._worker = SimilarityWorker(
                self._folder, method, k, self.chk_recursive.isChecked()
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.status_text.connect(self._on_worker_status)
            self._worker.finished_ok.connect(self._on_done)
            self._worker.finished_err.connect(self._on_fail)
            self.btn_run.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.progress.setValue(0)
            self._status.setText("Подождите, идёт подготовка…")
            self._worker.start()
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("failed to start worker")
            QMessageBox.critical(
                self,
                "Не удалось запустить",
                f"{type(exc).__name__}: {exc}\n\n"
                "Запустите программу из терминала — там появится полный текст ошибки.",
            )
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(False)

    def _on_worker_status(self, text: str) -> None:
        self._status.setText(text)

    def _cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            return
        self.progress.setValue(int(100 * done / total))

    def _on_done(self, bundle: dict) -> None:
        self._bundle = bundle
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_pdf.setEnabled(True)
        self.progress.setValue(100)
        res: SimilarityResult = bundle["result"]
        n = len(res.names)
        self._status.setText(f"Готово. Сравнено документов: {n}.")
        self.table.setRowCount(n)
        flagged: list[str] = []
        for i in range(n):
            significant = res.significant_neighbor_indices[i]
            if significant:
                parts = [f"{res.names[j]} ({res.matrix_percent[i, j]:.2f}%)" for j in significant]
                neighbor = "; ".join(parts)
            else:
                neighbor = "нет значимых совпадений"
            if len(significant) > 1:
                flagged.append(res.names[i])
            self.table.setItem(i, 0, QTableWidgetItem(res.names[i]))
            self.table.setItem(i, 1, QTableWidgetItem(f"{res.uniqueness_percent[i]:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{res.max_similarity[i]:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(neighbor))
        if flagged:
            self._suspicious.setText(
                "Несколько значимых источников (>= 20%) у: " + "; ".join(flagged)
            )
        else:
            self._suspicious.setText("")
        w = bundle["warnings"]
        if w:
            lines = "\n".join(f"{a}: {b}" for a, b in w[:30])
            more = f"\n... и ещё {len(w) - 30}" if len(w) > 30 else ""
            QMessageBox.information(
                self,
                "Замечания при открытии файлов",
                lines + more,
            )

    def _on_fail(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self._suspicious.setText("")
        self._status.setText("Сравнение не завершено — см. сообщение выше.")
        QMessageBox.warning(self, "Сообщение", msg)

    def _on_table_double_click(self, index: QModelIndex) -> None:
        if not self._bundle:
            return
        row = index.row()
        if row < 0:
            return
        entries: list[CorpusEntry] = self._bundle["entries"]
        res: SimilarityResult = self._bundle["result"]
        method: SimilarityMethod = self._bundle["method"]
        shingle_k = self._bundle["shingle_k"]
        corpus_norm = [e.normalized for e in entries]
        dlg = HighlightDialog(
            entries,
            corpus_norm,
            res,
            row,
            method,
            shingle_k,
            self.spin_K.value(),
            self,
        )
        dlg.exec()

    def _export_pdf(self) -> None:
        if not self._bundle:
            return
        bundle = self._bundle
        # Имя по умолчанию с расширением .pdf — в папке анализа, чтобы не искать каталог вручную.
        default_name = f"otchet_{datetime.now().strftime('%Y-%m-%d_%H%M')}.pdf"
        default_path = str(Path(bundle["folder"]) / default_name)
        dlg = QFileDialog(self)
        dlg.setWindowTitle("Сохранить отчёт")
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dlg.setNameFilters(["Документ PDF (*.pdf)"])
        dlg.setDefaultSuffix("pdf")
        dlg.selectFile(default_path)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dlg.selectedFiles()
        if not chosen:
            return
        path = chosen[0]
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".pdf":
            p = p.with_suffix(".pdf")
        try:
            write_report_pdf(
                p,
                bundle["folder"],
                bundle["method"],
                bundle["result"],
                bundle["warnings"],
                bundle["shingle_k"],
            )
        except FileNotFoundError as exc:
            QMessageBox.warning(
                self,
                "Не удалось сохранить PDF",
                str(exc),
            )
            return
        QMessageBox.information(self, "Отчёт сохранён", f"Файл:\n{p}")


def main() -> None:
    _orig_excepthook = sys.excepthook

    def _gui_excepthook(exc_type, exc, tb) -> None:
        traceback.print_exception(exc_type, exc, tb)
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Критическая ошибка",
                f"{exc_type.__name__}: {exc}\n\nПодробности — в окне терминала.",
            )

    sys.excepthook = _gui_excepthook
    try:
        app = QApplication([])
        # Fusion: Combo/Spin без QSS на подконтролах — стрелки рисует стиль; рамка задаётся на самих виджетах.
        app.setStyle("Fusion")
        app.setStyleSheet(_application_stylesheet())
        w = MainWindow()
        w.show()
        app.exec()
    finally:
        sys.excepthook = _orig_excepthook


if __name__ == "__main__":
    main()
