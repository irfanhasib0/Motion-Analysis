#!/bin/bash

# NVR Server Startup Script

echo "Starting NVR Server..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
#echo "Activating virtual environment..."
#source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install OpenCV if it's not already installed or if it was commented out
#echo "Ensuring OpenCV is installed..."
#pip install opencv-python==4.8.1.78

# Check if Node.js is installed for frontend build
#curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
if command -v node &> /dev/null; then
    echo "Building React frontend..."
    cd frontend
    
    # Install npm dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        echo "Installing npm dependencies..."
        npm install
    fi
    
    # Build frontend
    echo "Building frontend..."
    npm run build
    cd ..
else
    echo "Node.js not found. Skipping frontend build."
    echo "You can build the frontend manually with: cd frontend && npm install && npm run build"
fi

# Create recordings directory
echo "Creating recordings directory..."
mkdir -p recordings

# Start the server
echo "Starting FastAPI server..."
python3 main.py