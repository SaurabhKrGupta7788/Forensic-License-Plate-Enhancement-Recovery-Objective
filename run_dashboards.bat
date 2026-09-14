@echo off
call conda activate paddle_env_clone 2>nul
if %errorlevel% neq 0 (
    call "paddle_env_clone\Scripts\activate.bat" 2>nul
)
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python generate_dashboards.py
