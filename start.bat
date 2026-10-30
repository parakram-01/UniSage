@echo off
title UniSage - University Query Management System
color 0A

echo.
echo  ============================================================
echo    UniSage UQMS ^| University Query Management System
echo  ============================================================
echo.

cd /d "%~dp0"

echo  Starting UniSage Flask Application at http://127.0.0.1:5000
echo  Press Ctrl+C to stop the server.
echo.

python app.py

pause
