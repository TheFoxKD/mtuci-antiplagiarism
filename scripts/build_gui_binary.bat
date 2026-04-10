@echo off
REM Сборка одного exe GUI (Windows). Запуск: из проводника двойной щелчок ИЛИ из корня репо:
REM   scripts\build_gui_binary.bat
REM Нужны: uv (https://docs.astral.sh/uv/), интернет при первом uv sync.
cd /d "%~dp0\.."
where uv >nul 2>nul
if errorlevel 1 (
  echo Ошибка: uv не найден в PATH. Установите: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)
uv sync --group build
if errorlevel 1 exit /b 1
uv run pyinstaller --clean --noconfirm Antiplagiarism.spec
if errorlevel 1 exit /b 1
echo.
echo Готово: dist\Antiplagiarism.exe
