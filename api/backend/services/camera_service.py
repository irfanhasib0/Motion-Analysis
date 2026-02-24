import cv2
import numpy as np
import os
import json
import asyncio
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psutil
import logging

from models.camera import Camera, CameraCreate, CameraUpdate, CameraType, CameraStatus
from models.recording import Recording, RecordingCreate, RecordingStatus
#from services.database_service import DatabaseService
from services.config_manager import ConfigManager
from services.streaming_service import StreamingService

import sys
sys.path.append('../../src')
from improc.optical_flow import OpticalFlowTracker

logger = logging.getLogger(__name__)

class Capture:
    def __init__(self, source:str|int, width:int =640, height:int =480, fps:int =30):
        self.source = source
        self.cam_type = None
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._consecutive_read_failures = 0
        self._max_read_failures_before_reconnect = 3
        self._reconnect_cooldown_sec = 2.0
        self._last_reconnect_at = 0.0
        
        try:
            source = int(source)
        except:
            source = str(source)

        if isinstance(source, str) and source.startswith(('rtsp://', 'rtmp://')):
            self.cam_type = 'rtsp'
            self.open_rtsp()
        elif isinstance(source, str) and source.startswith(('http://', 'https://')):
            self.cam_type = 'http'
            self.open_rtsp()  # For simplicity, treat HTTP sources as RTSP for now
        elif type(source) == int or (isinstance(source, str) and source.split('.')[-1] in ['mp4', 'avi', 'mkv', 'mov']):
            self.cam_type = 'webcam'
            self.open_wcam()
        else:
            raise ValueError(f"Unsupported camera source: {source}")

    def open_wcam(self):
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self
    
    def open_rtsp(self, ):
        if self.cap:
            self.release_rtsp()

        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            #"-rw_timeout", "5000000",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-i", self.source,
            "-an",                    # no audio
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-pix_fmt", "bgr24",      # 8-bit BGR format
            "-f", "rawvideo",
            "pipe:1"
        ]
        try:
            self.cap = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        except Exception as e:
            logger.error(f"Failed to open RTSP stream: {e}")
            self.cap = None
        return self
    
    def is_opened(self):
        if self.cam_type == 'webcam':
            return self.cap and self.cap.isOpened()
        elif self.cam_type in ['rtsp', 'http']:
            return self.cap and self.cap.poll() is None and self.cap.stdout is not None
        return False

    def reconnect_rtsp(self) -> bool:
        now = time.time()
        if now - self._last_reconnect_at < self._reconnect_cooldown_sec:
            return self.is_opened()

        self._last_reconnect_at = now
        logger.warning(f"Reconnecting stream source: {self.source}")
        self.release_rtsp()
        self.open_rtsp()
        self._consecutive_read_failures = 0
        return self.is_opened()
    
    def read_wcam(self):
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None
    
    def read_rtsp(self):
        if not self.cap:
            return False, None

        if not self.is_opened():
            self.reconnect_rtsp()
            if not self.is_opened():
                return False, None

        frame_size = self.width * self.height * 3
        raw = b""
        try:
            raw = self.cap.stdout.read(frame_size)
        except Exception as e:
            logger.warning(f"RTSP read failed for source {self.source}: {e}")

        if len(raw) == frame_size:
            self._consecutive_read_failures = 0
            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
            return True, frame

        self._consecutive_read_failures += 1
        if self._consecutive_read_failures >= self._max_read_failures_before_reconnect:
            logger.warning(
                f"Short/empty RTSP frame read ({len(raw)}/{frame_size}) from {self.source}; attempting reconnect"
            )
            if self.reconnect_rtsp() and self.cap and self.cap.stdout:
                try:
                    raw = self.cap.stdout.read(frame_size)
                except Exception:
                    raw = b""

                if len(raw) == frame_size:
                    self._consecutive_read_failures = 0
                    frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
                    return True, frame

        return False, None

    def release_wcam(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def release_rtsp(self):
        if self.cap:
            try:
                if self.cap.stdout:
                    self.cap.stdout.close()
            except Exception:
                pass

            try:
                self.cap.terminate()
                self.cap.wait(timeout=1.0)
            except Exception:
                try:
                    self.cap.kill()
                except Exception:
                    pass
            self.cap = None
            
    def open(self):
        if self.cam_type in ['rtsp', 'http']:
            return self.open_rtsp()
        elif self.cam_type == 'webcam':
            return self.open_wcam()
    
    def read(self):
        if self.cam_type in ['rtsp', 'http']:
            return self.read_rtsp()
        elif self.cam_type == 'webcam':
            return self.read_wcam()
        else:
            raise ValueError(f"Unsupported camera type: {self.cam_type}")
        
    def release(self):
        if self.cam_type in ['rtsp', 'http']:
            self.release_rtsp()
        elif self.cam_type == 'webcam':
            self.release_wcam()
        else:
            raise ValueError(f"Unsupported camera type: {self.cam_type}")
        

class CameraService(StreamingService):
    def __init__(self):
        super().__init__()

        self.active_recordings: Dict[str, dict] = {}  # camera_id -> recording info
        self.recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "recordings")
        self.start_time = datetime.now()
        
        # Create recordings directory
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Load existing recordings from disk
        self._load_existing_recordings()
        self._camera_streams = {}
        self._camera_trackers = {}

        self.motion_check_interval = 10  # seconds
        self.max_clip_length = 60  # seconds
        self.max_velocity = 0.1#0.4  # velocity threshold for motion detection
        self.max_bg_diff = 50#200  # background difference threshold for motion detection

    def __del__(self):
        # Clean up any active recordings on shutdown
        for camera_id in list(self.active_recordings.keys()):
            self.stop_recording(camera_id)
        for camera_id in list(self._camera_streams.keys()):
            self.stop_camera(camera_id)
        
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
        camera_id = f"{camera_data.name}_{camera_data.camera_type.value}_{int(time.time())}"
        
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
        
        ## Validate source if updated
        #if 'source' in update_dict:
        #    if not self._validate_camera_source(update_dict['source'], camera_update.camera_type or CameraType.WEBCAM):
        #        raise ValueError(f"Invalid camera source: {update_dict['source']}")
        
        # Update in database
        if update_dict:
            updated_camera = self.db.update_camera(camera_id, update_dict)
            if updated_camera:
                camera = Camera(
                    id=updated_camera['id'],
                    name=updated_camera['name'],
                    source=updated_camera['source'],
                    camera_type=CameraType(updated_camera.get('camera_type', 'webcam')),
                    fps=updated_camera['fps'],
                    resolution=updated_camera['resolution'],
                    status=CameraStatus(updated_camera['status']),
                    created_at=datetime.fromisoformat(updated_camera['created_at']),
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
    
    def video_capture(self, camera_id: str):
        db_camera = self.db.get_camera(camera_id)
        source = db_camera['source']
        
        try:
            source = int(source)
        except:
            source = str(source)
        
        resolution = [int(res) for res in db_camera['resolution'].split('x')]
        fps = db_camera['fps']
        
        cap = None
        try:
            cap = Capture(source, width=resolution[0], height=resolution[1], fps=fps)    
        except Exception as e:
            logger.error(f"Failed to open video source: {source} with error: {e}")
            logger.error(f"Check if the source is correct and accessible: {source}")
        return cap
    
    def start_camera(self, camera_id: str) -> bool:
        """Start a camera"""
        camera_started = False
        cap = self.video_capture(camera_id)
        
        if cap is None:
            logger.warning(f"Camera: {self.db.get_camera(camera_id)['name']} Id: {camera_id} failed to open")
            self.db.update_camera(camera_id, {'status': CameraStatus.OFFLINE.value})
            return False
            
        tracker = OpticalFlowTracker()
        
        ret, _ = cap.read()
            
        if ret:
            self.db.update_camera(camera_id, {'status': CameraStatus.ONLINE.value})
            logger.info(f"Started camera: {self.db.get_camera(camera_id)['name']} ({camera_id})")
            camera_started = True
            self._camera_streams[camera_id] = cap
            self._camera_trackers[camera_id] = tracker
        else:
            logger.warning(f"Camera {camera_id} opened but failed to read frames")
            cap.release()
        
        if not camera_started:
            logger.warning(f"Camera: {self.db.get_camera(camera_id)['name']} Id: {camera_id} failed to open")
            self.db.update_camera(camera_id, {'status': CameraStatus.OFFLINE.value})

        return camera_started
        
    def stop_camera(self, camera_id: str):
        """Stop a camera"""
        db_camera = self.db.get_camera(camera_id)
        
        # Stop any active recording
        if camera_id in self.active_recordings:
            self.stop_recording(camera_id)

        if camera_id in self._camera_streams:
            cap = self._camera_streams.pop(camera_id)
            tracker = self._camera_trackers.pop(camera_id)
            cap.release()
            self.db.update_camera(camera_id, {'status': CameraStatus.OFFLINE.value})
            logger.info(f"Stopped camera: {db_camera['name']} ({camera_id})")
            
            del tracker
        else:
            logger.warning(f"No active camera object found for id: {camera_id}")
            return

    def close_camera_stream(self, camera_id: str):
        """Close an active camera stream and release related resources."""
        if camera_id not in self._camera_streams:
            logger.info(f"No active stream to close for camera: {camera_id}")
            return

        self.stop_camera(camera_id)

        self.stream_locks.pop(camera_id, None)
        self._latest_frames.pop(camera_id, None)
        self._latest_viz.pop(camera_id, None)
        self._latest_res.pop(camera_id, None)

        self._fps_stats.pop(f"{camera_id}:primary", None)
        self._fps_stats.pop(f"{camera_id}:processing", None)

        if hasattr(self, "active_streams"):
            self.active_streams.pop(camera_id, None)
    
    def init_recording(self, camera_id: str, db_camera: dict, cap: cv2.VideoCapture) -> Tuple[str, str, cv2.VideoWriter]:
        # Generate recording info
        timestamp = int(time.time())
        filename = f"{camera_id}_{timestamp}.mp4"
        recording_id = f"{camera_id}_{timestamp}"
        file_path = os.path.join(self.recordings_dir, str(camera_id), filename)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Create recording record in database
        recording_data = {
            'id': recording_id,
            'camera_id': camera_id,
            'file_path': file_path,
            'start_time': datetime.now().isoformat(),
            'status': 'recording'
        }
        self.db.create_recording(recording_data)

        # Get camera properties
        fps = db_camera['fps']
        width = cap.width
        height = cap.height
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
        return recording_id, file_path, out
    
    def send_notification_to_app(self):
        # Placeholder for sending notification to frontend app about recording status
        pass

    def process_recorded_clip(
        self,
        recording_id: str,
        file_path: str,
        out: cv2.VideoWriter,
        clip_motion_detected: bool,
        clip_start_time: float,
        curr_time: float,
        vel: float = 0.0,
        bg_diff: int = 0,
    ):
        out.release()
        clip_duration = max(0, int(curr_time - clip_start_time))
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        # Delete file if no motion detected, otherwise keep it
        if not clip_motion_detected:
            if os.path.exists(file_path):
                os.system(f'rm -f {file_path}')
                logger.info(f"Deleted no-motion recording: {file_path}")
            # Remove from database
            self.db.delete_recording(recording_id)
        else:
            # Update recording status to completed
            self.db.update_recording(
                recording_id,
                {
                    'status': 'completed',
                    'end_time': datetime.now().isoformat(),
                    'duration': clip_duration,
                    'file_size': file_size,
                    'metadata': {
                        'motion_detected': True,
                        'vel': float(vel),
                        'diff': int(bg_diff),
                    },
                },
            )
            logger.info(f"Saving recording with motion: {file_path}")
            self.send_notification_to_app()

    def record_worker(self, file_path, recording_id, camera_id, cap, out):
            
            start_time = time.time()
            clip_start_time = start_time
            curr_time = start_time
            clip_motion_detected = False
            clip_vel = 0.0
            clip_bg_diff = 0
            while camera_id in self.active_recordings:
                # Prefer frames already read by the streaming loop
                lock = self.stream_locks.get(camera_id)
                frame = None
                
                with lock:
                    frame = getattr(self, '_latest_frames', {}).get(camera_id)
                    res   = getattr(self, '_latest_res', {}).get(camera_id)

                # Rotate files if duration exceeds threshold (placeholder 60s)
                curr_time = time.time()
                if curr_time - start_time > self.motion_check_interval:
                    recent_motion_detected = False
                    
                    vel = res['vel']
                    bg_diff = int(res['bg_diff'])
                    if vel > self.max_velocity or bg_diff >= self.max_bg_diff:
                        recent_motion_detected = True
                        clip_motion_detected = True
                        clip_vel = max(clip_vel, vel)
                        clip_bg_diff = max(clip_bg_diff, bg_diff)

                    if not recent_motion_detected or (curr_time - clip_start_time) > self.max_clip_length:
                        self.process_recorded_clip(
                            recording_id,
                            file_path,
                            out,
                            clip_motion_detected,
                            clip_start_time,
                            curr_time,
                            vel=clip_vel,
                            bg_diff=clip_bg_diff,
                        )
                        recording_id, file_path, out = self.init_recording(camera_id, self.db.get_camera(camera_id), cap)
                        clip_start_time = curr_time
                        clip_motion_detected = False
                        clip_vel = 0.0
                        clip_bg_diff = 0
                    start_time = curr_time
                
                if frame is None:
                    # Fallback: read directly if no stream consumer is running
                    ret, frame = cap.read()
                    if not ret:
                        break

                out.write(frame)    
            self.process_recorded_clip(
                recording_id,
                file_path,
                out,
                clip_motion_detected,
                clip_start_time,
                curr_time,
                vel=clip_vel,
                bg_diff=clip_bg_diff,
            )

    def start_recording(self, camera_id: str) -> str:
        """Start recording from a camera"""
        db_camera = self.db.get_camera(camera_id)
        if camera_id in self.active_recordings:
            logger.info(f"Camera {camera_id} is already recording")
            return self.active_recordings[camera_id]['recording_id']
        
        # Initialize camera capture if not already done
        if camera_id not in self._camera_streams:
            scc = self.start_camera(camera_id, db_camera)
            if not scc:
                return
        else:
            cap = self._camera_streams.get(camera_id)

        recording_id, file_path, out = self.init_recording(camera_id, db_camera, cap)
        # Start recording thread
        recording_thread = threading.Thread(target=self.record_worker, args=(file_path, recording_id, camera_id, cap, out))
        recording_thread.daemon = True
        
        # Track active recording
        self.active_recordings[camera_id] = {
            'recording_id': recording_id,
            'thread': recording_thread,
            'start_time': datetime.now()
        }

        recording_thread.start()
        
        # Update camera status
        self.db.update_camera(camera_id, {'status': CameraStatus.RECORDING.value})
        
        logger.info(f"Started recording: {db_camera['name']} -> {file_path}")
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
            file_path = db_recording['file_path']
            filename = os.path.basename(file_path)
            created_at_str = db_recording.get('created_at')
            started_at_str = db_recording.get('start_time')
            ended_at_str = db_recording.get('end_time')
            metadata = db_recording.get('metadata')
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata) if metadata else None
                except Exception:
                    metadata = None

            recording = Recording(
                id=db_recording['id'],
                camera_id=db_recording['camera_id'],
                filename=filename,
                duration=db_recording.get('duration'),
                file_size=db_recording.get('file_size'),
                status=RecordingStatus(db_recording['status']),
                created_at=datetime.fromisoformat(created_at_str) if created_at_str else datetime.fromisoformat(started_at_str),
                started_at=datetime.fromisoformat(started_at_str) if started_at_str else None,
                ended_at=datetime.fromisoformat(ended_at_str) if ended_at_str else None,
                file_path=file_path,
                metadata=metadata
            )
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

    def get_disk_usage(self) -> float:
        """Return disk usage percent for recordings directory."""
        return psutil.disk_usage(self.recordings_dir).percent

    def get_uptime(self) -> str:
        """Return human-readable uptime string (HH:MM:SS)."""
        uptime = datetime.now() - self.start_time
        return str(uptime).split('.')[0]

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