@echo off
echo Activating paddle_env_clone...
call conda activate paddle_env_clone 2>nul
if %errorlevel% neq 0 (
    echo Conda environment not found. Attempting to activate as standard venv...
    call "paddle_env_clone\Scripts\activate.bat" 2>nul
)
echo Running pipeline...
python run_pipeline.py
echo Done.
pause
