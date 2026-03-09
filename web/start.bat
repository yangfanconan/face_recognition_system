@echo off
chcp 65001 >nul
echo ============================================
echo DDFD-FaceRec Web Demo
echo ============================================
echo.

cd /d "%~dp0"

echo Starting web service...
echo Access: http://localhost:7860
echo Press Ctrl+C to stop
echo.

..\venv\Scripts\python -u app.py

pause
