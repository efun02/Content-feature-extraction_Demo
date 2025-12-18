@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat >nul
streamlit run app.py --server.port=8501
pause