from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import json
import os
from datetime import datetime
import logging

from services.recording_service import RecordingService
from services.streaming_service import StreamingService
from services.processing_service import ProcessingService
from models.camera import Camera, CameraCreate, CameraUpdate
from models.recording import Recording, RecordingCreate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NVR Server", description="Network Video Recorder with RTSP and Camera Support")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
recording_service = RecordingService()
streaming_service = StreamingService()
processing_service = ProcessingService()

# Mount static files for React app only if build directory exists
frontend_build_path = "frontend/build"
frontend_static_path = "frontend/build/static"
if os.path.exists(frontend_static_path):
    app.mount("/static", StaticFiles(directory=frontend_static_path), name="static")
    logger.info("Frontend static files mounted")
else:
    logger.warning("Frontend build directory not found. Run 'cd frontend && npm install && npm run build' to build the frontend.")

# WebSocket connections for real-time updates
active_connections: Dict[str, WebSocket] = {}
last_camera_status: Dict[str, str] = {}  # Track last known camera status

async def check_camera_status_changes():
    """Background task to check for camera status changes and broadcast updates"""
    global last_camera_status
    while True:
        try:
            cameras = recording_service.get_cameras()
            for camera in cameras:
                current_status = camera.status.value
                last_status = last_camera_status.get(camera.id)
                
                if last_status != current_status:
                    last_camera_status[camera.id] = current_status
                    await broadcast_message({
                        "type": "camera_status_updated",
                        "camera_id": camera.id,
                        "status": current_status,
                        "camera": camera.dict()
                    })
                    logger.info(f"Camera {camera.id} status changed to {current_status}")
                    
        except Exception as e:
            logger.error(f"Error checking camera status: {e}")
        
        # Check every 2 seconds
        await asyncio.sleep(2)

# Start the background task
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks"""
    # Initialize camera status tracking
    cameras = recording_service.get_cameras()
    for camera in cameras:
        last_camera_status[camera.id] = camera.status.value
    
    # Start status checking task
    asyncio.create_task(check_camera_status_changes())
    logger.info("Started camera status monitoring")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            # Handle WebSocket messages if needed
    except WebSocketDisconnect:
        del active_connections[client_id]

async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    for connection in active_connections.values():
        try:
            await connection.send_text(json.dumps(message))
        except:
            pass

# Camera management endpoints
@app.get("/api/cameras", response_model=List[Camera])
async def get_cameras():
    """Get all cameras"""
    return recording_service.get_cameras()

@app.post("/api/cameras", response_model=Camera)
async def create_camera(camera: CameraCreate):
    """Add a new camera"""
    try:
        new_camera = recording_service.add_camera(camera)
        await broadcast_message({"type": "camera_added", "camera": new_camera.dict()})
        return new_camera
    except Exception as e:
        logger.error(f"Failed to add camera: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/cameras/{camera_id}", response_model=Camera)
async def update_camera(camera_id: str, camera_update: CameraUpdate):
    """Update camera settings"""
    try:
        updated_camera = recording_service.update_camera(camera_id, camera_update)
        await broadcast_message({"type": "camera_updated", "camera": updated_camera.dict()})
        return updated_camera
    except Exception as e:
        logger.error(f"Failed to update camera: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """Delete a camera"""
    try:
        recording_service.remove_camera(camera_id)
        await broadcast_message({"type": "camera_deleted", "camera_id": camera_id})
        return {"message": "Camera deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete camera: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start/Connect to a camera"""
    try:
        success = recording_service.start_camera(camera_id)
        if success:
            await broadcast_message({"type": "camera_started", "camera_id": camera_id})
            return {"message": "Camera started successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start camera - camera may be unavailable or in use")
    except ValueError as e:
        logger.error(f"Camera not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stop/Disconnect from a camera"""
    try:
        recording_service.stop_camera(camera_id)
        await broadcast_message({"type": "camera_stopped", "camera_id": camera_id})
        return {"message": "Camera stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop camera: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Recording management endpoints
@app.post("/api/cameras/{camera_id}/start-recording")
async def start_recording(camera_id: str, background_tasks: BackgroundTasks):
    """Start recording from a camera"""
    logger.info(f"Start recording request for camera: {camera_id}")
    try:
        # Check if camera exists and get its status
        if camera_id not in recording_service.cameras:
            logger.error(f"Camera not found: {camera_id}")
            raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")
        
        camera = recording_service.cameras[camera_id]
        logger.info(f"Camera status: {camera.status}, name: {camera.name}")
        
        recording_id = recording_service.start_recording(camera_id)
        logger.info(f"Recording started successfully: {recording_id}")
        
        await broadcast_message({
            "type": "recording_started", 
            "camera_id": camera_id,
            "recording_id": recording_id
        })
        return {"message": "Recording started", "recording_id": recording_id}
    except ValueError as e:
        logger.error(f"Validation error starting recording: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cameras/{camera_id}/stop-recording")
async def stop_recording(camera_id: str):
    """Stop recording from a camera"""
    try:
        recording_service.stop_recording(camera_id)
        await broadcast_message({"type": "recording_stopped", "camera_id": camera_id})
        return {"message": "Recording stopped"}
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/recordings", response_model=List[Recording])
async def get_recordings(camera_id: Optional[str] = None):
    """Get all recordings, optionally filtered by camera"""
    return recording_service.get_recordings(camera_id)

@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording"""
    try:
        recording_service.delete_recording(recording_id)
        return {"message": "Recording deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete recording: {e}")
        raise HTTPException(status_code=404, detail=str(e))

# Video streaming endpoints
@app.get("/api/cameras/{camera_id}/stream")
async def get_camera_stream(camera_id: str):
    """Get live video stream from camera"""
    try:
        return StreamingResponse(
            streaming_service.generate_camera_stream(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        logger.error(f"Failed to get camera stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/cameras/{camera_id}/stream/close")
async def close_camera_stream(camera_id: str):
    """Close camera stream to free resources"""
    try:
        streaming_service.close_camera_stream(camera_id)
        return {"message": "Camera stream closed successfully"}
    except Exception as e:
        logger.error(f"Failed to close camera stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recordings/{recording_id}/stream")
async def get_recording_stream(recording_id: str):
    """Stream a recorded video"""
    try:
        return StreamingResponse(
            streaming_service.generate_recording_stream(recording_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        logger.error(f"Failed to get recording stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/recordings/{recording_id}/download")
async def download_recording(recording_id: str):
    """Download a recorded video file"""
    try:
        file_path = recording_service.get_recording_path(recording_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Recording file not found")
        
        return FileResponse(
            file_path,
            media_type='video/mp4',
            filename=f"recording_{recording_id}.mp4"
        )
    except Exception as e:
        logger.error(f"Failed to download recording: {e}")
        raise HTTPException(status_code=404, detail=str(e))

# Processing endpoints
@app.get("/api/processing/types")
async def get_processing_types():
    """Get available video processing types"""
    return processing_service.get_available_processors()

@app.post("/api/cameras/{camera_id}/processing/{processor_type}/start")
async def start_processing(camera_id: str, processor_type: str, params: dict = None):
    """Start custom processing on camera stream"""
    try:
        await processing_service.start_processing(camera_id, processor_type, params)
        await broadcast_message({
            "type": "processing_started",
            "camera_id": camera_id,
            "processor_type": processor_type
        })
        return {"message": f"Started {processor_type} processing on camera {camera_id}"}
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/cameras/{camera_id}/processing/stop")
async def stop_processing(camera_id: str):
    """Stop custom processing on camera stream"""
    try:
        processing_service.stop_processing(camera_id)
        await broadcast_message({
            "type": "processing_stopped",
            "camera_id": camera_id
        })
        return {"message": f"Stopped processing on camera {camera_id}"}
    except Exception as e:
        logger.error(f"Failed to stop processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/system/info")
async def get_system_info():
    """Get system information and camera status"""
    return {
        "cameras": recording_service.get_camera_status(),
        "recordings": len(recording_service.get_recordings()),
        "processing_active": processing_service.get_active_processors(),
        "disk_usage": recording_service.get_disk_usage(),
        "uptime": recording_service.get_uptime()
    }

# Serve React app
@app.get("/")
async def serve_react_app():
    if os.path.exists("frontend/build/index.html"):
        return FileResponse("frontend/build/index.html")
    else:
        return {"message": "NVR Server API is running. Frontend not built. Access the API at /docs"}

@app.get("/{path:path}")
async def serve_react_routes(path: str):
    file_path = f"frontend/build/{path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    elif os.path.exists("frontend/build/index.html"):
        return FileResponse("frontend/build/index.html")
    else:
        return {"message": "NVR Server API is running. Frontend not built. Access the API at /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=True)