from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import json
import os
from datetime import datetime
import logging
import base64
import hashlib
import hmac
import secrets
import time

from services.camera_service import CameraService
from services.dashboard_service import DashboardService
from models.camera import Camera, CameraCreate, CameraUpdate
from models.recording import Recording, RecordingCreate

user = os.popen('uname -n').read().strip()
if user == 'irfan-linux':
    configs = 'pc'
elif user == 'raspberrypi':
    configs = 'pi'
else:
    configs = 'default'

print("Running with config:", configs)

# Initialize services
camera_service = CameraService(configs=configs)
dashboard_service = DashboardService(camera_service=camera_service)
    
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
AUTH_PASSWORD = os.getenv("API_PASSWORD", "admin123")
AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))
AUTH_SECRET = os.getenv("AUTH_SECRET", secrets.token_hex(32))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NVR Server", description="Network Video Recorder with RTSP and Camera Support")

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)
api_key_scheme = APIKeyHeader(name="x-api-password", auto_error=False)

class LoginRequest(BaseModel):
    password: str

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token() -> str:
    payload = {
        "exp": int(time.time()) + AUTH_TOKEN_TTL_SECONDS,
        "scope": "api",
    }
    payload_str = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_part = _b64url(payload_str.encode("utf-8"))
    signature = hmac.new(AUTH_SECRET.encode("utf-8"), payload_part.encode("utf-8"), hashlib.sha256).digest()
    signature_part = _b64url(signature)
    return f"{payload_part}.{signature_part}"


def verify_access_token(token: str) -> bool:
    try:
        payload_part, signature_part = token.split(".", 1)
        expected_signature = hmac.new(
            AUTH_SECRET.encode("utf-8"), payload_part.encode("utf-8"), hashlib.sha256
        ).digest()
        provided_signature = _b64url_decode(signature_part)
        if not hmac.compare_digest(expected_signature, provided_signature):
            return False

        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return False
        return payload.get("scope") == "api"
    except Exception:
        return False


async def legacy_auth_header_valid(request: Request) -> bool:
    password_header = await api_key_scheme(request)
    return bool(password_header and password_header == AUTH_PASSWORD)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not AUTH_ENABLED:
        return await call_next(request)

    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    public_paths = {"/api/health", "/api/auth/token", "/api/auth/login"}
    if path in public_paths:
        return await call_next(request)
    
    # Optional: Do not remove this commented block
    #auth_header = request.headers.get("Authorization", "")
    #if auth_header.startswith("Bearer "):
    #    token = auth_header.replace("Bearer ", "", 1).strip()
    #    if verify_access_token(token):
    #        return await call_next(request)
    
    bearer_token = await oauth2_scheme(request)
    if bearer_token and verify_access_token(bearer_token):
        return await call_next(request)

    query_token = request.query_params.get("access_token")
    if query_token:
        query_token = query_token.split("?", 1)[0]
    if query_token and verify_access_token(query_token):
        return await call_next(request)
    
    # Optional: Do not remove this commented block
    #password_header = request.headers.get("x-api-password")
    #if password_header and password_header == AUTH_PASSWORD:
    #    return await call_next(request)
    if await legacy_auth_header_valid(request):
        return await call_next(request)

    return Response(
        content=json.dumps({"detail": "Unauthorized"}),
        status_code=401,
        media_type="application/json",
    )

async def check_camera_status_changes():
    """Background task to check for camera status changes and broadcast updates"""
    global last_camera_status
    while True:
        try:
            cameras = camera_service.get_cameras()
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
    cameras = camera_service.get_cameras()
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
@app.post("/api/auth/token")
async def token_login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not AUTH_ENABLED:
        return {
            "access_token": "",
            "token_type": "bearer",
            "expires_in": 0,
            "auth_enabled": False,
        }

    if form_data.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    return {
        "access_token": create_access_token(),
        "token_type": "bearer",
        "expires_in": AUTH_TOKEN_TTL_SECONDS,
        "auth_enabled": True,
    }


@app.post("/api/auth/login")
async def login(payload: LoginRequest):
    if not AUTH_ENABLED:
        return {
            "access_token": "",
            "token_type": "bearer",
            "expires_in": 0,
            "auth_enabled": False,
        }

    if payload.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    return {
        "access_token": create_access_token(),
        "token_type": "bearer",
        "expires_in": AUTH_TOKEN_TTL_SECONDS,
        "auth_enabled": True,
    }


@app.get("/api/cameras", response_model=List[Camera])
async def get_cameras():
    """Get all cameras"""
    return camera_service.get_cameras()

@app.post("/api/cameras", response_model=Camera)
async def create_camera(camera: CameraCreate):
    """Add a new camera"""
    try:
        new_camera = camera_service.add_camera(camera)
        await broadcast_message({"type": "camera_added", "camera": new_camera.dict()})
        return new_camera
    except Exception as e:
        logger.error(f"Failed to add camera: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/cameras/{camera_id}", response_model=Camera)
async def update_camera(camera_id: str, camera_update: CameraUpdate):
    """Update camera settings"""
    try:
        updated_camera = camera_service.update_camera(camera_id, camera_update)
        await broadcast_message({"type": "camera_updated", "camera": updated_camera.dict()})
        return updated_camera
    except Exception as e:
        logger.error(f"Failed to update camera: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """Delete a camera"""
    try:
        camera_service.remove_camera(camera_id)
        await broadcast_message({"type": "camera_deleted", "camera_id": camera_id})
        return {"message": "Camera deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete camera: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start/Connect to a camera"""
    try:
        success = camera_service.start_camera(camera_id)
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
        camera_service.stop_camera(camera_id)
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
        if camera_id not in camera_service.cameras:
            logger.error(f"Camera not found: {camera_id}")
            raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")
        
        camera = camera_service.cameras[camera_id]
        logger.info(f"Camera status: {camera.status}, name: {camera.name}")
        
        recording_id = camera_service.start_recording(camera_id)
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
        camera_service.stop_recording(camera_id)
        await broadcast_message({"type": "recording_stopped", "camera_id": camera_id})
        return {"message": "Recording stopped"}
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/recordings", response_model=List[Recording])
async def get_recordings(camera_id: Optional[str] = None):
    """Get all recordings, optionally filtered by camera"""
    return camera_service.get_recordings(camera_id)

@app.get("/api/system/info")
async def get_system_info():
    """Get overall system and start_server process metrics for dashboard."""
    try:
        return dashboard_service.get_system_info()
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recordings/storage")
async def get_recording_storage():
    """Get recording storage stats and enforce low-space cleanup policy."""
    try:
        return camera_service.get_recording_storage_info(enforce_policy=True)
    except Exception as e:
        logger.error(f"Failed to get recording storage info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording"""
    try:
        camera_service.delete_recording(recording_id)
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
            camera_service.generate_camera_stream(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Failed to get camera stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/cameras/{camera_id}/processing_stream")
async def get_processing_stream(camera_id: str):
    """Get processed video stream from camera"""
    try:
        return StreamingResponse(
            camera_service.generate_processing_stream(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Failed to get processed camera stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
@app.post("/api/cameras/{camera_id}/stream/close")
async def close_camera_stream(camera_id: str):
    """Close camera stream to free resources"""
    try:
        camera_service.close_camera_stream(camera_id)
        return {"message": "Camera stream closed successfully"}
    except Exception as e:
        logger.error(f"Failed to close camera stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras/{camera_id}/stream/blank")
async def get_blank_stream(camera_id: str):
    """Stream a blank video"""
    try:
        return Response(
            camera_service.generate_blank_image(str(camera_id)),
            media_type="image/jpeg"
        )
    except Exception as e:
        logger.error(f"Failed to get blank stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
@app.get("/api/recordings/{recording_id}/stream")
async def get_recording_stream(recording_id: str):
    """Stream a recorded video"""
    try:
        return StreamingResponse(
            camera_service.generate_recording_stream(recording_id),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        logger.error(f"Failed to get recording stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/recordings/{recording_id}/play")
async def play_recording(recording_id: str):
    """Serve a recorded video file for browser playback"""
    try:
        file_path = camera_service.get_browser_playable_recording_path(recording_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Recording file not found")
        
        response = FileResponse(file_path, media_type="video/mp4")
        response.headers["Content-Disposition"] = f'inline; filename="recording_{recording_id}.mp4"'
        return response
    except Exception as e:
        logger.error(f"Failed to play recording: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/cameras/{camera_id}/result_stream")
async def get_camera_results(camera_id: str):
    """Get JSON stream of processing results for a camera"""
    try:
        return camera_service.generate_result_json_stream(camera_id)
            
    except Exception as e:
        logger.error(f"Failed to get camera results: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/recordings/{recording_id}/download")
async def download_recording(recording_id: str):
    """Download a recorded video file"""
    try:
        file_path = camera_service.get_recording_path(recording_id)
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

# Serve React app
@app.get("/")
async def serve_react_app():
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {"message": "NVR Server API is running. Frontend not built. Access the API at /docs"}

@app.get("/{path:path}")
async def serve_react_routes(path: str):
    file_path = f"../frontend/build/{path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    elif os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {"message": "NVR Server API is running. Frontend not built. Access the API at /docs"}

if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").lower() in {"1", "true", "yes", "on"}
    uvicorn.run("start_server:app", host="0.0.0.0", port=9001, reload=reload_enabled)