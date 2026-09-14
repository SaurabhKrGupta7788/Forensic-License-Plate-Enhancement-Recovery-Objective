@echo off
echo Activating paddle_env_clone...
call conda activate paddle_env_clone 2>nul
if %errorlevel% neq 0 (
    call "paddle_env_clone\Scripts\activate.bat" 2>nul
)

echo Fixing protobuf environment for Streamlit...
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo Running Streamlit app...
streamlit run app.py
pause
