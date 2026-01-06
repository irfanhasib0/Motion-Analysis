@echo off
REM NVR Server Startup Script for Windows

echo Starting NVR Server...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is required but not installed.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Check if Node.js is installed
node --version >nul 2>&1
if not errorlevel 1 (
    echo Building React frontend...
    cd frontend
    
    REM Install npm dependencies if node_modules doesn't exist
    if not exist "node_modules" (
        echo Installing npm dependencies...
        npm install
    )
    
    REM Build frontend
    echo Building frontend...
    npm run build
    cd ..
) else (
    echo Node.js not found. Skipping frontend build.
    echo You can build the frontend manually with: cd frontend && npm install && npm run build
)

REM Create recordings directory
echo Creating recordings directory...
if not exist "recordings" mkdir recordings

REM Start the server
echo Starting FastAPI server...
python main.py

pause