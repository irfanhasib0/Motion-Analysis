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
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class RecordingService:
    def __init__(self, recordings_dir: str = "recordings"):
        self.db = DatabaseService()
        self.active_recordings: Dict[str, dict] = {}  # camera_id -> recording info
        self.recordings_dir = recordings_dir
        self.start_time = datetime.now()
        
        # Create recordings directory
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Load existing recordings from disk
        self._load_existing_recordings()

    def _load_existing_recordings(self):
        """Sync existing recording files with database"""
        if not os.path.exists(self.recordings_dir):
            return
            
        # Get existing recordings from database
        db_recordings = {rec['file_path']: rec for rec in self.db.get_all_recordings()}
            
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(self.recordings_dir, filename)
                file_stat = os.stat(file_path)
                
                # Skip if already in database
                if file_path in db_recordings:
                    continue
                
                # Extract recording info from filename (assuming format: camera_id_timestamp.mp4)
                base_name = filename[:-4]  # Remove .mp4
                parts = base_name.split('_')
                if len(parts) >= 2:
                    camera_id = '_'.join(parts[:-1])
                    timestamp_str = parts[-1]
                    
                    try:
                        created_at = datetime.fromtimestamp(int(timestamp_str))
                        recording_id = str(uuid.uuid4())
                        
                        # Add to database
                        recording_data = {
                            'id': recording_id,
                            'camera_id': camera_id,
                            'file_path': file_path,
                            'start_time': created_at.isoformat(),
                            'duration': 0,
                            'file_size': file_stat.st_size,
                            'status': 'completed'
                        }
                        self.db.create_recording(recording_data)
                        logger.info(f"Added existing recording to database: {filename}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse recording filename {filename}: {e}")

    def get_cameras(self) -> List[Camera]:
        """Get all cameras"""
        db_cameras = self.db.get_all_cameras()
        cameras = []
        
        for db_camera in db_cameras:
            camera = Camera(
                id=db_camera['id'],
                name=db_camera['name'],
                source=db_camera['source'],
                camera_type=CameraType(db_camera.get('camera_type', 'webcam')),
                fps=db_camera['fps'],
                resolution=db_camera['resolution'],
                status=CameraStatus(db_camera['status']),
                created_at=datetime.fromisoformat(db_camera['created_at']),
                processing_active=db_camera['processing_active'],
                processing_type=db_camera.get('processing_type')
            )
            cameras.append(camera)
        
        return cameras

    def add_camera(self, camera_data: CameraCreate) -> Camera:
        """Add a new camera"""
        camera_id = str(uuid.uuid4())
        
        # Validate camera source
        if not self._validate_camera_source(camera_data.source, camera_data.camera_type):
            raise ValueError(f"Invalid camera source: {camera_data.source}")
        
        camera_dict = {
            'id': camera_id,
            'name': camera_data.name,
            'source': camera_data.source,
            'camera_type': camera_data.camera_type.value,
            'fps': camera_data.fps,
            'resolution': camera_data.resolution,
            'status': CameraStatus.OFFLINE.value
        }
        
        # Store in database
        db_camera = self.db.create_camera(camera_dict)
        
        # Convert to Camera model
        camera = Camera(
            id=db_camera['id'],
            name=db_camera['name'],
            source=db_camera['source'],
            camera_type=camera_data.camera_type,
            fps=db_camera['fps'],
            resolution=db_camera['resolution'],
            status=CameraStatus(db_camera['status']),
            created_at=datetime.fromisoformat(db_camera['created_at'])
        )
        
        logger.info(f"Added camera: {camera.name} ({camera_id})")
        return camera

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Get a camera by ID"""
        db_camera = self.db.get_camera(camera_id)
        if not db_camera:
            return None
            
        return Camera(
            id=db_camera['id'],
            name=db_camera['name'],
            source=db_camera['source'],
            camera_type=CameraType(db_camera.get('camera_type', 'webcam')),
            fps=db_camera['fps'],
            resolution=db_camera['resolution'],
            status=CameraStatus(db_camera['status']),
            created_at=datetime.fromisoformat(db_camera['created_at']),
            processing_active=db_camera['processing_active'],
            processing_type=db_camera.get('processing_type')
        )

    def update_camera(self, camera_id: str, camera_update: CameraUpdate) -> Camera:
        """Update camera settings"""
        db_camera = self.db.get_camera(camera_id)
        if not db_camera:
            raise ValueError(f"Camera not found: {camera_id}")
        
        # Update fields if provided
        update_data = camera_update.dict(exclude_unset=True)
        update_dict = {}
        
        if 'name' in update_data:
            update_dict['name'] = update_data['name']
        if 'source' in update_data:
            update_dict['source'] = update_data['source']
        if 'fps' in update_data:
            update_dict['fps'] = update_data['fps']
        if 'resolution' in update_data:
            update_dict['resolution'] = update_data['resolution']
        
        # Validate source if updated
        if 'source' in update_dict:
            if not self._validate_camera_source(update_dict['source'], camera_update.camera_type or CameraType.WEBCAM):
                raise ValueError(f"Invalid camera source: {update_dict['source']}")
        
        # Update in database
        if update_dict:
            updated_camera = self.db.update_camera(camera_id, update_dict)
            if updated_camera:
                camera = Camera(
                    id=updated_camera['id'],
                    name=updated_camera['name'],
                    source=updated_camera['source'],
                    fps=updated_camera['fps'],
                    resolution=updated_camera['resolution'],
                    status=CameraStatus(updated_camera['status']),
                    processing_active=updated_camera['processing_active'],
                    processing_type=updated_camera.get('processing_type'),
                    processing_params=updated_camera.get('processing_params', {})
                )
                
                logger.info(f"Updated camera: {camera.name} ({camera_id})")
                return camera
        
        raise ValueError("No valid updates provided")

    def remove_camera(self, camera_id: str):
        """Remove a camera"""
        db_camera = self.db.get_camera(camera_id)
        
        # Stop any active recording
        if camera_id in self.active_recordings:
            self.stop_recording(camera_id)
        
        camera_name = db_camera['name']
        self.db.delete_camera(camera_id)
        logger.info(f"Removed camera: {camera_name} ({camera_id})")

    def _validate_camera_source(self, source: str, camera_type: CameraType) -> bool:
        """Validate camera source based on type"""
        if camera_type == CameraType.WEBCAM:
            try:
                index = int(source)
                return index >= 0
            except ValueError:
                return False
        elif camera_type == CameraType.RTSP:
            return source.startswith(('rtsp://', 'rtmp://'))
        elif camera_type == CameraType.IP_CAMERA:
            return source.startswith(('http://', 'https://'))
        return False

    def _test_camera_connection(self, camera_id: str) -> bool:
        """Test camera connection"""
        db_camera = self.db.get_camera(camera_id)
        if not db_camera:
            return False
            
        try:
            cap = self.video_capture(db_camera['source'])
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
        except Exception as e:
            logger.error(f"Camera test failed for {camera_id}: {e}")
        
        return False
    
    def video_capture(self, source: int|str):
        try:
            source_int = int(source)
            cap = cv2.VideoCapture(source_int)
        except ValueError:
            cap = cv2.VideoCapture(source)
        return cap
    
    def start_camera(self, camera_id: str) -> bool:
        """Start a camera"""
        camera_started = False
        db_camera = self.db.get_camera(camera_id)
        source = db_camera['source']
        cap = self.video_capture(source)
        
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                self.db.update_camera(camera_id, {'status': CameraStatus.ONLINE.value})
                logger.info(f"Started camera: {db_camera['name']} ({camera_id})")
                camera_started = True
        
        if not camera_started:
            logger.warning(f"Camera: {db_camera['name']} Id: {camera_id} failed to open")
            self.db.update_camera(camera_id, {'status': CameraStatus.OFFLINE.value})
        return camera_started
        
    def stop_camera(self, camera_id: str):
        """Stop a camera"""
        db_camera = self.db.get_camera(camera_id)
        
        # Stop any active recording
        if camera_id in self.active_recordings:
            self.stop_recording(camera_id)
        
        # Update status
        self.db.update_camera(camera_id, {'status': CameraStatus.OFFLINE.value})
        logger.info(f"Stopped camera: {db_camera['name']} ({camera_id})")

    def start_recording(self, camera_id: str) -> str:
        """Start recording from a camera"""
        db_camera = self.db.get_camera(camera_id)
        
        if camera_id in self.active_recordings:
            raise ValueError(f"Camera {camera_id} is already recording")
        
        # Generate recording info
        recording_id = str(uuid.uuid4())
        timestamp = int(time.time())
        filename = f"{camera_id}_{timestamp}.mp4"
        file_path = os.path.join(self.recordings_dir, filename)
        
        # Create recording record in database
        recording_data = {
            'id': recording_id,
            'camera_id': camera_id,
            'file_path': file_path,
            'start_time': datetime.now().isoformat(),
            'status': 'recording'
        }
        self.db.create_recording(recording_data)
        
        # Start recording thread
        source = db_camera['source']
        def record_worker(source, file_path, recording_id, camera_id):
            cap = self.video_capture(source)

            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id} for recording")
                return
            
            # Get camera properties
            fps = db_camera['fps']
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            start_time = time.time()
            
            try:
                while camera_id in self.active_recordings:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    time.sleep(1.0 / fps)
            
            except Exception as e:
                logger.error(f"Failed to start recording for camera {camera_id}: {e}")
                # Clean up on failure
                if recording_id:
                    self.db.delete_recording(recording_id)
                raise
                    
            finally:
                cap.release()
                out.release()
                
                # Calculate duration and file size
                duration = int(time.time() - start_time)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                # Update recording in database
                self.db.update_recording(recording_id, {
                    'end_time': datetime.now().isoformat(),
                    'duration': duration,
                    'file_size': file_size,
                    'status': 'completed'
                })
                
                logger.info(f"Recording completed: {filename}")
        
        # Start recording thread
        recording_thread = threading.Thread(target=record_worker, args=(source, file_path, recording_id, camera_id))
        recording_thread.daemon = True
        recording_thread.start()
        
        # Track active recording
        self.active_recordings[camera_id] = {
            'recording_id': recording_id,
            'thread': recording_thread,
            'start_time': datetime.now()
        }
        
        # Update camera status
        self.db.update_camera(camera_id, {'status': CameraStatus.RECORDING.value})
        
        logger.info(f"Started recording: {db_camera['name']} -> {filename}")
        return recording_id    

    def stop_recording(self, camera_id: str):
        """Stop recording from a camera"""
        db_camera = self.db.get_camera(camera_id)
        if not db_camera:
            raise ValueError(f"Camera not found: {camera_id}")
        
        if camera_id not in self.active_recordings:
            raise ValueError(f"Camera {camera_id} is not recording")
        
        # Remove from active recordings (this will stop the recording thread)
        recording_info = self.active_recordings.pop(camera_id)
        
        # Wait for thread to finish
        recording_info['thread'].join(timeout=5)
        
        # Update camera status
        self.db.update_camera(camera_id, {'status': CameraStatus.ONLINE.value})
        
        logger.info(f"Stopped recording: {db_camera['name']}")

    def get_recordings(self, camera_id: Optional[str] = None) -> List[Recording]:
        """Get recordings, optionally filtered by camera"""
        if camera_id:
            db_recordings = self.db.get_recordings_by_camera(camera_id)
        else:
            db_recordings = self.db.get_all_recordings()
        
        recordings = []
        for db_recording in db_recordings:
            recording = Recording(
                id=db_recording['id'],
                camera_id=db_recording['camera_id'],
                file_path=db_recording['file_path'],
                start_time=datetime.fromisoformat(db_recording['start_time']),
                duration=db_recording.get('duration', 0),
                file_size=db_recording.get('file_size', 0),
                status=RecordingStatus(db_recording['status'])
            )
            if db_recording.get('end_time'):
                recording.end_time = datetime.fromisoformat(db_recording['end_time'])
            recordings.append(recording)
        
        return recordings

    def delete_recording(self, recording_id: str):
        """Delete a recording"""
        db_recording = self.db.get_recording(recording_id)
        if not db_recording:
            raise ValueError(f"Recording not found: {recording_id}")
        
        # Delete file if it exists
        file_path = db_recording['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        self.db.delete_recording(recording_id)
        logger.info(f"Deleted recording: {recording_id}")

    def get_camera_status(self) -> Dict[str, dict]:
        """Get status of all cameras"""
        cameras = self.get_cameras()
        status = {}
        
        for camera in cameras:
            recording_info = self.active_recordings.get(camera.id)
            status[camera.id] = {
                'name': camera.name,
                'status': camera.status.value,
                'is_recording': camera.id in self.active_recordings,
                'recording_start': recording_info['start_time'].isoformat() if recording_info else None,
                'processing_active': camera.processing_active,
                'processing_type': camera.processing_type
            }
        
        return status

    def get_system_info(self) -> Dict:
        """Get system information"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(self.recordings_dir)
        
        uptime = datetime.now() - self.start_time
        
        return {
            'uptime': str(uptime).split('.')[0],
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'active_recordings': len(self.active_recordings),
            'total_cameras': len(self.get_cameras()),
            'total_recordings': len(self.get_recordings())
        }

    # Properties to maintain compatibility
    @property
    def cameras(self) -> Dict[str, Camera]:
        """Get cameras as dict for backward compatibility"""
        cameras = self.get_cameras()
        return {camera.id: camera for camera in cameras}
    
    @property
    def recordings(self) -> Dict[str, Recording]:
        """Get recordings as dict for backward compatibility"""
        recordings = self.get_recordings()
        return {recording.id: recording for recording in recordings}