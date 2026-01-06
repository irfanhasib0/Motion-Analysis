#!/bin/bash

# Frontend Build Script for NVR Server

echo "Building React frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is required but not installed."
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is required but not installed."
    exit 1
fi

# Navigate to frontend directory
if [ ! -d "frontend" ]; then
    echo "Frontend directory not found. Make sure you're in the api directory."
    exit 1
fi

cd frontend

# Install npm dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
else
    echo "npm dependencies already installed."
fi

# Build frontend
echo "Building React frontend..."
npm run build

# Check if build was successful
if [ -d "build" ] && [ -f "build/index.html" ]; then
    echo "✅ Frontend build completed successfully!"
    echo "The NVR server can now serve the React application."
else
    echo "❌ Frontend build failed!"
    exit 1
fi

cd ..
echo "Frontend build script completed."