@echo off
setlocal enabledelayedexpansion
title NoSlop - RVQ/Codec Artifact Remover
cd /d "%~dp0"

for /f "delims=" %%v in ('python -m app --version 2^>nul') do set "NOSLOP_VERSION=%%v"
if not defined NOSLOP_VERSION set "NOSLOP_VERSION=NoSlop"

set "PORT=8737"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:%PORT%"

:menu
cls
echo ============================================
echo   %NOSLOP_VERSION% - RVQ/Codec Artifact Remover
echo   https://github.com/leckminartor/noslop
echo ============================================
echo.
echo   [1] Web-App starten        (KI-Modi: quality/deep/dering - GPU)
echo   [2] Web-App starten        (nur DSP-Modus, kein GPU-Torch noetig)
echo   [3] CLI - Audio analysieren
echo   [4] CLI - Audio wiederherstellen (DSP-Preset)
echo   [5] CLI - KI-Enhance (quality/deep/dering)
echo   [6] Version anzeigen
echo   [7] Tests ausfuehren
echo   [8] Beenden
echo.
set /p choice="Auswahl: "

if "%choice%"=="1" goto start_web_ai
if "%choice%"=="2" goto start_web_dsp
if "%choice%"=="3" goto cli_analyze
if "%choice%"=="4" goto cli_restore
if "%choice%"=="5" goto cli_enhance
if "%choice%"=="6" goto show_version
if "%choice%"=="7" goto run_tests
if "%choice%"=="8" goto end
goto menu

:start_web_ai
if not exist "%VENV_PY%" (
    echo [!] Keine .venv gefunden. AI-Modi benoetigen .venv mit torch+demucs.
    echo     Siehe README: AI stage installieren oder Option 2 nutzen.
    pause
    goto menu
)
echo Starte Web-App mit AI-Unterstuetzung auf %URL% ...
"%VENV_PY%" -m uvicorn app.server:app --host 127.0.0.1 --port %PORT%
echo.
echo Server beendet.
pause
goto menu

:start_web_dsp
echo Starte Web-App (nur DSP-Modi) auf %URL% ...
python -m uvicorn app.server:app --host 127.0.0.1 --port %PORT%
echo.
echo Server beendet.
pause
goto menu

:cli_analyze
set /p filepath="Pfad zur Audiodatei: "
if not exist "%filepath:"=%" (
    echo [!] Datei nicht gefunden.
    pause
    goto menu
)
python -m app.cli analyze "%filepath:"=%"
echo.
pause
goto menu

:cli_restore
set /p filepath="Pfad zur Audiodatei: "
if not exist "%filepath:"=%" (
    echo [!] Datei nicht gefunden.
    pause
    goto menu
)
echo Presets: gentle / standard / strong
set /p preset="Preset [standard]: "
if "%preset%"=="" set preset=standard
python -m app.cli restore "%filepath:"=%" --preset %preset%
echo.
pause
goto menu

:cli_enhance
if not exist "%VENV_PY%" (
    echo [!] Keine .venv gefunden. KI-Enhance benoetigt .venv mit torch+demucs.
    pause
    goto menu
)
set /p filepath="Pfad zur Audiodatei: "
if not exist "%filepath:"=%" (
    echo [!] Datei nicht gefunden.
    pause
    goto menu
)
echo Modi: fast (nur DSP) / quality (AI-Stems) / deep (volle Re-Renderung) / dering (Metal-Entfernung)
set /p mode="Modus [quality]: "
if "%mode%"=="" set mode=quality
"%VENV_PY%" -m app.cli enhance "%filepath:"=%" --mode %mode%
echo.
pause
goto menu

:show_version
python -m app --version
echo.
pause
goto menu

:run_tests
python -m pytest tests/ -q
echo.
pause
goto menu

:end
endlocal