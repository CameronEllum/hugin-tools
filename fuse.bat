@echo off
rem ============================================================================
rem  fuse.bat - shim: run `hugin fuse`.  Kept so existing habits keep working;
rem  see hugin.bat for the full command.
rem
rem  Options go after the command, e.g.:  fuse.bat --all
rem ============================================================================

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install it from https://docs.astral.sh/uv/  ^(e.g. pip install uv^)
    exit /b 1
)

uv run "%~dp0hugin.py" fuse %*
exit /b %errorlevel%
