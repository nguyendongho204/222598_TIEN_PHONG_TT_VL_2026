@echo off
echo Killing old process on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 >nul
echo Starting EBM-SVM API...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
