import cv2
import os
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil
import logging

from models.camera import Camera, CameraCreate, CameraUpdate, CameraType, CameraStatus
from models.recording import Recording, RecordingCreate, RecordingStatus

logger = logging.getLogger(__name__)

class RecordingService:
    def __init__(self, recordings_dir: str = "recordings"):
        self.cameras: Dict[str, Camera] = {}
        self.recordings: Dict[str, Recording] = {}
        self.active_recordings: Dict[str, dict] = {}  # camera_id -> recording info
        self.recordings_dir = recordings_dir
        self.start_time = datetime.now()
        
        # Create recordings directory
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Load existing recordings from disk
        self._load_existing_recordings()

    def _load_existing_recordings(self):
        """Load existing recording files from disk"""
        if not os.path.exists(self.recordings_dir):
            return
            
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(self.recordings_dir, filename)
                file_stat = os.stat(file_path)
                
                # Extract recording info from filename (assuming format: camera_id_timestamp.mp4)
                base_name = filename[:-4]  # Remove .mp4
                parts = base_name.split('_')
                if len(parts) >= 2:
                    camera_id = '_'.join(parts[:-1])
                    timestamp_str = parts[-1]
                    
                    try:
                        created_at = datetime.fromtimestamp(int(timestamp_str))
                        recording_id = str(uuid.uuid4())
                        
                        recording = Recording(
                            id=recording_id,
                            camera_id=camera_id,
                            filename=filename,
                            file_path=file_path,
                            file_size=file_stat.st_size,
                            status=RecordingStatus.COMPLETED,
                            created_at=created_at,
                            started_at=created_at,
                            ended_at=datetime.fromtimestamp(file_stat.st_mtime),
                            format="mp4"
                        )
                        self.recordings[recording_id] = recording
                        logger.info(f"Loaded existing recording: {filename}")
                    except ValueError:
                        logger.warning(f"Could not parse timestamp from filename: {filename}")

    def get_cameras(self) -> List[Camera]:
        """Get all cameras"""
        return list(self.cameras.values())

    def add_camera(self, camera_data: CameraCreate) -> Camera:
        """Add a new camera"""
        camera_id = str(uuid.uuid4())
        
        # Validate camera source
        if not self._validate_camera_source(camera_data.source, camera_data.camera_type):
            raise ValueError(f"Invalid camera source: {camera_data.source}")
        
        camera = Camera(
            id=camera_id,
            name=camera_data.name,
            camera_type=camera_data.camera_type,
            source=camera_data.source,
            resolution=camera_data.resolution,
            fps=camera_data.fps,
            enabled=camera_data.enabled,
            description=camera_data.description,
            location=camera_data.location,
            status=CameraStatus.OFFLINE,
            created_at=datetime.now(),
            processing_active=False
        )
        
        self.cameras[camera_id] = camera
        logger.info(f"Added camera: {camera.name} ({camera_id})")
        
        # Test camera connection
        self._test_camera_connection(camera_id)
        
        return camera

    def update_camera(self, camera_id: str, camera_update: CameraUpdate) -> Camera:
        """Update camera settings"""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera not found: {camera_id}")
        
        camera = self.cameras[camera_id]
        
        # Update fields if provided
        update_data = camera_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(camera, field, value)
        
        # Validate source if updated
        if 'source' in update_data:
            if not self._validate_camera_source(camera.source, camera.camera_type):
                raise ValueError(f"Invalid camera source: {camera.source}")
            # Test new connection
            self._test_camera_connection(camera_id)
        
        logger.info(f"Updated camera: {camera.name} ({camera_id})")
        return camera

    def remove_camera(self, camera_id: str):
        """Remove a camera"""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera not found: {camera_id}")
        
        # Stop recording if active
        if camera_id in self.active_recordings:
            self.stop_recording(camera_id)
        
        camera_name = self.cameras[camera_id].name
        del self.cameras[camera_id]
        logger.info(f"Removed camera: {camera_name} ({camera_id})")

    def _validate_camera_source(self, source: str, camera_type: CameraType) -> bool:
        """Validate camera source based on type"""
        if camera_type == CameraType.RTSP:
            return source.startswith(('rtsp://', 'rtsps://'))
        elif camera_type == CameraType.WEBCAM:
            try:
                int(source)  # Should be a device index
                return True
            except ValueError:
                return False
        elif camera_type == CameraType.IP_CAMERA:
            return source.startswith(('http://', 'https://'))
        return False

    def _test_camera_connection(self, camera_id: str):
        """Test camera connection and update status"""
        camera = self.cameras[camera_id]
        
        def test_connection():
            try:
                if camera.camera_type == CameraType.WEBCAM:
                    cap = cv2.VideoCapture(int(camera.source))
                else:
                    cap = cv2.VideoCapture(camera.source)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        camera.status = CameraStatus.ONLINE
                        camera.last_seen = datetime.now()
                        logger.info(f"Camera {camera.name} is online")
                    else:
                        camera.status = CameraStatus.OFFLINE
                        logger.warning(f"Camera {camera.name} failed to read frame")
                else:
                    camera.status = CameraStatus.OFFLINE
                    logger.warning(f"Camera {camera.name} failed to open")
                
                cap.release()
                
            except Exception as e:
                camera.status = CameraStatus.ERROR
                logger.error(f"Error testing camera {camera.name}: {e}")
        
        # Run test in background thread
        threading.Thread(target=test_connection, daemon=True).start()

    async def start_recording(self, camera_id: str) -> Recording:
        """Start recording from a camera"""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera not found: {camera_id}")
        
        if camera_id in self.active_recordings:
            raise ValueError(f"Camera {camera_id} is already recording")
        
        camera = self.cameras[camera_id]
        
        # Create recording record
        recording_id = str(uuid.uuid4())
        timestamp = int(time.time())
        filename = f"{camera_id}_{timestamp}.mp4"
        file_path = os.path.join(self.recordings_dir, filename)
        
        recording = Recording(
            id=recording_id,
            camera_id=camera_id,
            filename=filename,
            file_path=file_path,
            status=RecordingStatus.RECORDING,
            created_at=datetime.now(),
            started_at=datetime.now(),
            format="mp4",
            resolution=camera.resolution,
            fps=camera.fps
        )
        
        self.recordings[recording_id] = recording
        camera.recording_id = recording_id
        camera.status = CameraStatus.RECORDING
        
        # Start recording in background thread
        recording_thread = threading.Thread(
            target=self._record_camera,
            args=(camera_id, recording_id, file_path),
            daemon=True
        )
        
        self.active_recordings[camera_id] = {
            'recording_id': recording_id,
            'thread': recording_thread,
            'stop_event': threading.Event()
        }
        
        recording_thread.start()
        logger.info(f"Started recording from camera {camera.name} to {filename}")
        
        return recording

    def stop_recording(self, camera_id: str):
        """Stop recording from a camera"""
        if camera_id not in self.active_recordings:
            raise ValueError(f"Camera {camera_id} is not recording")
        
        # Signal to stop recording
        self.active_recordings[camera_id]['stop_event'].set()
        
        # Wait for thread to finish
        self.active_recordings[camera_id]['thread'].join(timeout=5)
        
        recording_id = self.active_recordings[camera_id]['recording_id']
        del self.active_recordings[camera_id]
        
        # Update recording status
        if recording_id in self.recordings:
            recording = self.recordings[recording_id]
            recording.status = RecordingStatus.COMPLETED
            recording.ended_at = datetime.now()
            
            # Update file size
            if os.path.exists(recording.file_path):
                recording.file_size = os.path.getsize(recording.file_path)
                recording.duration = (recording.ended_at - recording.started_at).total_seconds()
        
        # Update camera status
        camera = self.cameras[camera_id]
        camera.recording_id = None
        camera.status = CameraStatus.ONLINE
        
        logger.info(f"Stopped recording from camera {camera.name}")

    def _record_camera(self, camera_id: str, recording_id: str, output_path: str):
        """Record video from camera in background thread"""
        camera = self.cameras[camera_id]
        stop_event = self.active_recordings[camera_id]['stop_event']
        
        try:
            # Open camera
            if camera.camera_type == CameraType.WEBCAM:
                cap = cv2.VideoCapture(int(camera.source))
            else:
                cap = cv2.VideoCapture(camera.source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera.name}")
                self.recordings[recording_id].status = RecordingStatus.FAILED
                return
            
            # Set camera properties
            if camera.resolution:
                width, height = map(int, camera.resolution.split('x'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if camera.fps:
                cap.set(cv2.CAP_PROP_FPS, camera.fps)
            
            # Get actual camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logger.info(f"Recording {camera.name} at {width}x{height}@{fps}fps")
            
            frame_count = 0
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera.name}")
                    time.sleep(0.1)
                    continue
                
                # Write frame
                out.write(frame)
                frame_count += 1
                
                # Update camera last_seen
                camera.last_seen = datetime.now()
                
                # Small delay to prevent overwhelming the system
                time.sleep(1.0 / fps)
            
            cap.release()
            out.release()
            
            logger.info(f"Recorded {frame_count} frames from camera {camera.name}")
            
        except Exception as e:
            logger.error(f"Error recording from camera {camera.name}: {e}")
            if recording_id in self.recordings:
                self.recordings[recording_id].status = RecordingStatus.FAILED

    def get_recordings(self, camera_id: Optional[str] = None) -> List[Recording]:
        """Get all recordings, optionally filtered by camera"""
        recordings = list(self.recordings.values())
        if camera_id:
            recordings = [r for r in recordings if r.camera_id == camera_id]
        
        # Sort by creation time (newest first)
        recordings.sort(key=lambda x: x.created_at, reverse=True)
        return recordings

    def get_recording_path(self, recording_id: str) -> str:
        """Get the file path for a recording"""
        if recording_id not in self.recordings:
            raise ValueError(f"Recording not found: {recording_id}")
        
        return self.recordings[recording_id].file_path

    def delete_recording(self, recording_id: str):
        """Delete a recording"""
        if recording_id not in self.recordings:
            raise ValueError(f"Recording not found: {recording_id}")
        
        recording = self.recordings[recording_id]
        
        # Delete file if it exists
        if os.path.exists(recording.file_path):
            os.remove(recording.file_path)
        
        # Delete thumbnail if it exists
        if recording.thumbnail_path and os.path.exists(recording.thumbnail_path):
            os.remove(recording.thumbnail_path)
        
        del self.recordings[recording_id]
        logger.info(f"Deleted recording: {recording.filename}")

    def get_camera_status(self) -> Dict[str, dict]:
        """Get status of all cameras"""
        return {
            camera_id: {
                'name': camera.name,
                'status': camera.status.value,
                'recording': camera_id in self.active_recordings,
                'processing_active': camera.processing_active,
                'last_seen': camera.last_seen.isoformat() if camera.last_seen else None
            }
            for camera_id, camera in self.cameras.items()
        }

    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage information"""
        try:
            disk_usage = psutil.disk_usage(self.recordings_dir)
            return {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'percent_used': (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {'error': str(e)}

    def get_uptime(self) -> Dict[str, float]:
        """Get system uptime information"""
        uptime = datetime.now() - self.start_time
        return {
            'days': uptime.days,
            'hours': uptime.seconds // 3600,
            'minutes': (uptime.seconds % 3600) // 60,
            'total_seconds': uptime.total_seconds()
        }