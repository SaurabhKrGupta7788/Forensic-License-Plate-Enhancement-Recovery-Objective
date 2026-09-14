@echo off
echo Activating paddle_env_clone...
call conda activate paddle_env_clone 2>nul
if %errorlevel% neq 0 (
    call "paddle_env_clone\Scripts\activate.bat" 2>nul
)
echo Running Flask app...
python flask_app.py
pause
