#!/usr/bin/env python3
"""
Simple NVR Server - API Only
Run this script to start the NVR API server without the frontend.
Access the API documentation at http://localhost:8000/docs
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import main app components
try:
    from main import app
    print("✅ NVR Server components loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import main app: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "NVR Server API is running",
        "frontend_available": os.path.exists("frontend/build/index.html")
    }

if __name__ == "__main__":
    print("🎬 Starting NVR Server (API Only)")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/health")
    
    if not os.path.exists("frontend/build/index.html"):
        print("⚠️  Frontend not built. To build frontend:")
        print("   ./build_frontend.sh")
        print("   or manually: cd frontend && npm install && npm run build")
    
    try:
        uvicorn.run(
            "run_api:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)