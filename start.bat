@echo off
REM Start DVR Restorer GUI on Windows.
REM Double-click or run from any terminal.
cd /d "%~dp0"
python -m dvrrestorer.gui.app %*
