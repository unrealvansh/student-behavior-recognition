@echo off
echo ============================================================
echo   Student Behavior Recognition System
echo   Starting server...
echo ============================================================
echo.

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Navigate to the project directory
cd /d "%~dp0"

REM Run the Flask app
python app.py

pause
