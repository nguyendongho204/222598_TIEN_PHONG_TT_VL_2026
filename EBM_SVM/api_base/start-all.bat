@echo off
REM Start Backend and Frontend

echo ========================================
echo Starting EBM-SVM Full Stack...
echo ========================================

REM Start Backend API
echo.
echo [1/2] Starting Backend API (Port 8000)...
cd /d D:\Nam4\ThucTap\EBM_SVM\api_base
start "Backend API" cmd /k python run_api.py

REM Wait a bit for backend to start
timeout /t 3 /nobreak

REM Start Frontend
echo [2/2] Starting Frontend (Port 3000)...
cd /d D:\Nam4\ThucTap\EBM_SVM\api_base\frontend
start "Frontend React" cmd /k npm start

echo.
echo ========================================
echo ✓ Both servers are starting!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend:    http://localhost:3000
echo Swagger Docs: http://localhost:8000/docs
echo.
echo Press any key to close this window...
pause
