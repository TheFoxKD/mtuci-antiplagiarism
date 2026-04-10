#!/usr/bin/env bash
# Собрать один исполняемый файл GUI в dist/Antiplagiarism (macOS/Linux).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --group build
uv run pyinstaller --clean --noconfirm Antiplagiarism.spec
echo ""
echo "Готово. Запуск:"
echo "  ./dist/Antiplagiarism"
