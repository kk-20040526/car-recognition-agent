@echo off
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
.venv\Scripts\pythonw.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000 >> storage\server.log 2>> storage\server.err.log

