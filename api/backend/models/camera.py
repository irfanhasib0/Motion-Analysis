from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class CameraType(str, Enum):
    RTSP = "rtsp"
    WEBCAM = "webcam"
    IP_CAMERA = "ip_camera"
    RECORDED = "recorded"

class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RECORDING = "recording"
    ERROR = "error"

class CameraBase(BaseModel):
    name: str
    camera_type: CameraType
    source: str  # RTSP URL, device index, or IP camera URL
    resolution: Optional[str] = "1920x1080"
    fps: Optional[int] = 30
    enabled: bool = True
    description: Optional[str] = None
    location: Optional[str] = None

class CameraCreate(CameraBase):
    pass

class CameraUpdate(CameraBase):
    name: Optional[str] = None
    camera_type: Optional[CameraType] = None
    source: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None
    location: Optional[str] = None

class Camera(CameraBase):
    id: str
    status: CameraStatus
    created_at: datetime
    last_seen: Optional[datetime] = None
    recording_id: Optional[str] = None
    processing_active: bool = False
    processing_type: Optional[str] = None
    
    class Config:
        from_attributes = True