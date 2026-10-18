@echo off
rem ============================================================================
rem  pano.bat - shim: run `hugin_tools pano`.  Kept so existing habits keep
rem  working; see hugin_tools.bat for the full command.
rem
rem  Options go after the command, e.g.:  pano.bat -d
rem ============================================================================

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install it from https://docs.astral.sh/uv/  ^(e.g. pip install uv^)
    exit /b 1
)

uv run "%~dp0hugin_tools.py" pano %*
exit /b %errorlevel%
