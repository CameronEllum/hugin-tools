@echo off
rem ============================================================================
rem  pano.bat - run Pano (pano.py) on Windows via uv.
rem
rem  Run it from the folder that holds your panorama jpgs, e.g.:
rem      pano.bat                  (results in this folder)
rem      pano.bat -i "D:\path\row" -o "D:\path\result"
rem      pano.bat -d               (verbose logging)
rem
rem  All arguments are passed straight through to pano.py.
rem ============================================================================

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install it from https://docs.astral.sh/uv/  ^(e.g. pip install uv^)
    exit /b 1
)

uv run "%~dp0pano.py" %*
exit /b %errorlevel%
