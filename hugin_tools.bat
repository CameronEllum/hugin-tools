@echo off
rem ============================================================================
rem  hugin_tools.bat - run hugin_tools (hugin_tools.py) on Windows via uv.
rem
rem  Run it from the folder that holds your images, e.g.:
rem      hugin_tools.bat fuse --all            (fuse: every output type, motion included)
rem      hugin_tools.bat fuse --aligner all    (run manual and ais into sub-folders)
rem      hugin_tools.bat fuse -d               (keep everything + verbose logging)
rem      hugin_tools.bat pano                  (stitch a panorama)
rem      hugin_tools.bat pano -d               (keep the numbered .pto projects)
rem      hugin_tools.bat fuse -i "D:\path\export" -o "D:\path\out"
rem
rem  The command (fuse or pano) comes first; all later arguments pass through.
rem ============================================================================

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install it from https://docs.astral.sh/uv/  ^(e.g. pip install uv^)
    exit /b 1
)

uv run "%~dp0hugin_tools.py" %*
exit /b %errorlevel%
