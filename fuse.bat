@echo off
rem ============================================================================
rem  fuse.bat - run Fuse (fuse.py) on Windows via uv.
rem
rem  Run it from the folder that holds your images, e.g.:
rem      fuse.bat                  (default ais aligner, results in this folder)
rem      fuse.bat --all            (every output type, motion included)
rem      fuse.bat --aligner all    (run manual and ais into sub-folders)
rem      fuse.bat -d               (keep everything + verbose logging)
rem      fuse.bat -i "D:\path\export" -o "D:\path\out"
rem
rem  All arguments are passed straight through to fuse.py.
rem ============================================================================

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install it from https://docs.astral.sh/uv/  ^(e.g. pip install uv^)
    exit /b 1
)

uv run "%~dp0fuse.py" %*
exit /b %errorlevel%
