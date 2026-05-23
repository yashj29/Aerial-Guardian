@echo off
echo Starting Aerial Guardian...
echo.

:: Start FastAPI backend in a new window
start "Aerial Guardian API" cmd /k "cd /d %~dp0 && python -m uvicorn api.server:app --host 0.0.0.0 --port 8000"

:: Wait a moment for the API to boot
timeout /t 3 /nobreak > nul

:: Start React frontend in a new window
start "Aerial Guardian Web" cmd /k "cd /d %~dp0\web && npm run dev"

echo.
echo API  -> http://localhost:8000
echo Web  -> http://localhost:5173
echo.
echo Both servers are starting in separate windows.
pause
