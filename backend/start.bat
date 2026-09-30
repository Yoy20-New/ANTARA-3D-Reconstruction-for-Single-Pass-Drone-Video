@echo off
echo.
echo  ============================================================
echo   PROJECT ANTARA -- Web Upload Portal
echo  ============================================================
echo.
echo  Starting server on http://127.0.0.1:8765/
echo  Press Ctrl+C to stop.
echo.
call conda activate antara
cd /d %~dp0..
python -m backend.server %*
pause
