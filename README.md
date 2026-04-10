# MTUCI Antiplagiarism

Приложение для сравнения текстов и подсчёта процента похожести.

## Окружение (uv)

Зависимости задаются в `pyproject.toml`, lock — в `uv.lock`.

```bash
uv sync
```

Добавить пакет: `uv add <имя>`. Запуск в venv проекта: `uv run python ...`.

## Запуск

**Путь к файлу:** везде удобнее писать **`src/gui_app.py`** (через `/`). Так работает и Windows, и macOS.  
В **zsh / bash** (macOS, Linux) не используйте обратный слэш `src\gui_app.py` — `\` там экранирует символ, и путь превратится в несуществующий `srcgui_app.py`.

### Windows (PowerShell или «Командная строка»)

1. Установите **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (официальный установщик для Windows подойдёт).
2. Откройте терминал и перейдите в **корень проекта** (папка, где лежат `pyproject.toml` и `src`):

   ```bat
   cd "C:\путь\к\проекту\MTUCI-IT-Project"
   ```

3. Подтяните зависимости:

   ```bat
   uv sync
   ```

4. Запуск **окна программы (GUI)** — любой из вариантов:

   ```bat
   uv run python src/gui_app.py
   ```

   или

   ```bat
   uv run antiplagiarism-gui
   ```

5. Запуск **консоли** (топ похожих пар, Жаккар):

   ```bat
   uv run python src/text_similarity_prototype.py
   ```

   или

   ```bat
   uv run antiplagiarism-cli
   ```

Если Windows спрашивает разрешение для Python или «неизвестное приложение» — это нормально для скриптов; при скачанном **готовом `.exe`** (см. ниже) может всплыть **Защитник Windows / SmartScreen** для неподписанной программы: «Подробнее» → «Выполнить в любом случае» (если доверяете источнику).

### macOS / Linux

После `uv sync`:

```bash
uv run antiplagiarism-gui
uv run antiplagiarism-cli
```

Напрямую из `src/`:

```bash
uv run python src/gui_app.py
uv run python src/text_similarity_prototype.py
```

PDF извлекается только как текстовый слой (без OCR). Сканированные страницы без текста будут пропущены с предупреждением.

**Быстрая самопроверка:** папка [`data/quick-test/`](data/quick-test/) — три коротких `.txt`; в GUI выберите её как папку для скана и смотрите инструкцию в `data/quick-test/README.txt`. Расширенный набор примеров — в `data/texts/`.

## Сборка бинарника для демонстрации / тестирования

Один исполняемый файл GUI (всё внутри, Python на машине не нужен) собирается через [PyInstaller](https://pyinstaller.org/) и файл [`Antiplagiarism.spec`](Antiplagiarism.spec).

**macOS / Linux:**

```bash
uv sync --group build
./scripts/build_gui_binary.sh
# или: uv run pyinstaller --clean --noconfirm Antiplagiarism.spec
```

**Windows** (из корня проекта в PowerShell/cmd):

```bat
uv sync --group build
scripts\build_gui_binary.bat
```

или одной командой: `uv run pyinstaller --clean --noconfirm Antiplagiarism.spec`

**Результат:** в папке `dist/` — **`Antiplagiarism.exe`** (Windows) или файл **`Antiplagiarism`** (macOS/Linux). Запуск двойным щелчком по `.exe` или из терминала.

Каталоги `build/` и `dist/` в git не входят (см. `.gitignore`). На macOS неподписанный бинарник может блокировать Gatekeeper (ПКМ → «Открыть»). На Windows возможны предупреждения SmartScreen для неподписанного `.exe`.

Если при запуске сборки или бинарника не хватает модуля, добавьте его в `hiddenimports` в `Antiplagiarism.spec`.
