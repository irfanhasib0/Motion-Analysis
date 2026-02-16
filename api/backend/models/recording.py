from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RecordingStatus(str, Enum):
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING = "processing"

class RecordingBase(BaseModel):
    camera_id: str
    filename: str
    duration: Optional[float] = None
    file_size: Optional[int] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    format: str = "mp4"

class RecordingCreate(RecordingBase):
    pass

class Recording(RecordingBase):
    id: str
    status: RecordingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    file_path: str
    thumbnail_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True