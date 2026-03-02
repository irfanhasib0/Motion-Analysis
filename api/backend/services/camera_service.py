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
from typing import Dict, List, Optional, Tuple, Union
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
    def __init__(self, source: Union[str, int], width:int =640, height:int =480, fps:int =30, low_power_mode: bool = False):
        self.source = source
        self.cam_type = None
        self.width = width
        self.height = height
        self.fps = fps
        self.low_power_mode = low_power_mode
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
            "-threads", "1",
            #"-rw_timeout", "5000000",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-avioflags", "direct",
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
            pipe_buffer_size = 10**6 if self.low_power_mode else 10**8
            self.cap = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=pipe_buffer_size)
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
    def __init__(self, configs: Optional[str] = 'default'):
        super().__init__()
        self.low_power_mode = True
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.db = ConfigManager(configs_dir=os.path.join(self.root_dir, 'configs', configs))  # Use same YAML-backed DB as recording service
        self.active_recordings: Dict[str, dict] = {}  # camera_id -> recording info
        self.recordings_dir = os.path.join(self.root_dir, "recordings")
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
        self.motion_result_max_age_sec = 3.0  # ignore stale vel/bg_diff samples
        self.min_free_storage_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
        self.processing_stride = 3 if self.low_power_mode else 1
        self.jpeg_quality = 55 if self.low_power_mode else 70

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
            cap = Capture(source, width=resolution[0], height=resolution[1], fps=fps, low_power_mode=self.low_power_mode)
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
        self._latest_frame_seq.pop(camera_id, None)
        self._latest_viz.pop(camera_id, None)
        self._latest_res.pop(camera_id, None)

        self._fps_stats.pop(f"{camera_id}:primary", None)
        self._fps_stats.pop(f"{camera_id}:processing", None)

        if hasattr(self, "active_streams"):
            self.active_streams.pop(camera_id, None)
    
    def init_recording(self, camera_id: str, db_camera: dict, cap: cv2.VideoCapture) -> Tuple[str, str, cv2.VideoWriter]:
        self._ensure_min_free_storage()

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

    def _parse_recording_time(self, db_recording: dict) -> datetime:
        for key in ('start_time', 'created_at', 'end_time'):
            value = db_recording.get(key)
            if not value:
                continue
            try:
                value = value.replace('Z', '+00:00') if isinstance(value, str) else value
                return datetime.fromisoformat(value)
            except Exception:
                continue
        return datetime.min

    def _get_oldest_deletable_recording_id(self) -> Optional[str]:
        active_recording_ids = {
            info.get('recording_id')
            for info in self.active_recordings.values()
            if info.get('recording_id')
        }

        candidates = []
        for recording in self.db.get_all_recordings():
            recording_id = recording.get('id')
            if not recording_id:
                continue
            if recording.get('status') == 'recording':
                continue
            if recording_id in active_recording_ids:
                continue
            candidates.append(recording)

        if not candidates:
            return None

        oldest = min(candidates, key=self._parse_recording_time)
        return oldest.get('id')

    def _ensure_min_free_storage(self) -> int:
        deleted_count = 0
        while True:
            usage = psutil.disk_usage(self.recordings_dir)
            if usage.free >= self.min_free_storage_bytes:
                break

            oldest_recording_id = self._get_oldest_deletable_recording_id()
            if not oldest_recording_id:
                logger.warning(
                    "Low storage detected but no deletable recording found. "
                    f"Free bytes: {usage.free}"
                )
                break

            try:
                self.delete_recording(oldest_recording_id)
                deleted_count += 1
                logger.warning(
                    f"Deleted oldest recording {oldest_recording_id} due to low storage "
                    f"(free={usage.free} bytes)"
                )
            except Exception as e:
                logger.error(f"Failed to delete oldest recording {oldest_recording_id}: {e}")
                break

        return deleted_count

    def get_recording_storage_info(self, enforce_policy: bool = False) -> Dict:
        deleted_count = 0
        if enforce_policy:
            deleted_count = self._ensure_min_free_storage()

        disk = psutil.disk_usage(self.recordings_dir)
        return {
            'total_bytes': int(disk.total),
            'used_bytes': int(disk.used),
            'free_bytes': int(disk.free),
            'total_gb': round(disk.total / (1024 ** 3), 2),
            'used_gb': round(disk.used / (1024 ** 3), 2),
            'free_gb': round(disk.free / (1024 ** 3), 2),
            'percent_used': round(float(disk.percent), 2),
            'min_free_bytes': int(self.min_free_storage_bytes),
            'deleted_oldest_count': int(deleted_count),
        }
    
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
            threading.Thread(
                target=self._warm_browser_playback_cache,
                args=(recording_id,),
                daemon=True,
            ).start()
            self.send_notification_to_app()

    def _warm_browser_playback_cache(self, recording_id: str):
        """Pre-generate browser-playable video in background to reduce first-play latency."""
        try:
            self.get_browser_playable_recording_path(recording_id)
        except Exception as e:
            logger.warning(f"Failed to warm browser playback cache for {recording_id}: {e}")

    def record_worker(self, file_path, recording_id, camera_id, cap, out):
            
            start_time = time.time()
            clip_start_time = start_time
            curr_time = start_time
            clip_motion_detected = False
            clip_vel = 0.0
            clip_bg_diff = 0
            clip_frame_count = 0
            db_camera = self.db.get_camera(camera_id) or {}
            target_fps = max(1, int(db_camera.get('fps', 10) or 10))
            target_interval = 1.0 / float(target_fps)
            next_write_at = time.time()
            last_frame_seq = -1

            while camera_id in self.active_recordings:
                # Prefer frames already read by the streaming loop
                lock = self.stream_locks.get(camera_id)
                frame = None
                frame_seq = -1
                res = {'vel': 0.0, 'bg_diff': 0}
                
                if lock is not None:
                    with lock:
                        frame = getattr(self, '_latest_frames', {}).get(camera_id)
                        frame_seq = int(getattr(self, '_latest_frame_seq', {}).get(camera_id, -1))
                        latest_res = getattr(self, '_latest_res', {}).get(camera_id)
                        if isinstance(latest_res, dict):
                            res = latest_res
                else:
                    latest_res = getattr(self, '_latest_res', {}).get(camera_id)
                    if isinstance(latest_res, dict):
                        res = latest_res

                if frame is not None and frame_seq == last_frame_seq:
                    frame = None

                # Rotate files if duration exceeds threshold (placeholder 60s)
                curr_time = time.time()
                if curr_time - start_time > self.motion_check_interval:
                    recent_motion_detected = False

                    res_timestamp = float(res.get('ts', 0.0) or 0.0)
                    is_stale_motion_sample = (res_timestamp <= 0.0) or ((curr_time - res_timestamp) > self.motion_result_max_age_sec)

                    if is_stale_motion_sample:
                        vel = 0.0
                        bg_diff = 0
                    else:
                        vel = float(res.get('vel', 0.0))
                        bg_diff = int(res.get('bg_diff', 0))

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
                        clip_frame_count = 0
                    start_time = curr_time
                
                if frame is None:
                    # Fallback: read directly if no stream consumer is running
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                else:
                    last_frame_seq = frame_seq

                out.write(frame)
                clip_frame_count += 1

                # Pace writes to camera FPS to avoid lock contention and CPU spikes
                next_write_at += target_interval
                sleep_for = next_write_at - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_write_at = time.time()

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

    def get_recording_path(self, recording_id: str) -> str:
        """Resolve recording file path from DB for playback/download endpoints."""
        db_recording = self.db.get_recording(recording_id)
        if not db_recording:
            raise ValueError(f"Recording not found: {recording_id}")

        file_path = db_recording.get('file_path')
        if not file_path:
            raise ValueError(f"Recording file path missing: {recording_id}")

        # Primary/expected location
        normalized_path = os.path.abspath(file_path)
        if os.path.exists(normalized_path):
            if db_recording.get('file_path') != normalized_path:
                self.db.update_recording(recording_id, {'file_path': normalized_path})
            return normalized_path

        # Minimal fallback locations under recordings directory
        candidates: List[str] = []
        filename = os.path.basename(file_path)
        camera_id = db_recording.get('camera_id')
        if filename and camera_id:
            candidates.append(os.path.join(self.recordings_dir, str(camera_id), filename))
        if filename:
            candidates.append(os.path.join(self.recordings_dir, filename))

        for candidate in candidates:
            candidate_abs = os.path.abspath(candidate)
            if os.path.exists(candidate_abs):
                # Self-heal DB path when it points to a stale location
                if db_recording.get('file_path') != candidate_abs:
                    self.db.update_recording(recording_id, {'file_path': candidate_abs})
                return candidate_abs

        raise ValueError(f"Recording file not found: {recording_id}")

    def _get_video_codec(self, file_path: str) -> Optional[str]:
        """Return codec name (e.g. h264, mpeg4) for the first video stream."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return None
            codec = (result.stdout or "").strip().lower()
            return codec or None
        except Exception:
            return None

    def get_browser_playable_recording_path(self, recording_id: str) -> str:
        """Return recording path optimized for HTML5 video playback.

        If source codec is not broadly browser-compatible, create and reuse a
        transcoded H.264 version alongside the source file.
        """
        source_path = self.get_recording_path(recording_id)
        codec = self._get_video_codec(source_path)

        # Most reliable baseline for browser playback in MP4 containers
        if codec in {"h264", "avc1"}:
            return source_path

        root, ext = os.path.splitext(source_path)
        playable_path = f"{root}.browser{ext or '.mp4'}"

        source_mtime = os.path.getmtime(source_path)
        if os.path.exists(playable_path):
            playable_mtime = os.path.getmtime(playable_path)
            if playable_mtime >= source_mtime and os.path.getsize(playable_path) > 0:
                return playable_path

        cmd = [
            "ffmpeg",
            "-y",
            "-i", source_path,
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            playable_path,
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(playable_path) or os.path.getsize(playable_path) == 0:
            stderr_tail = (result.stderr or "")[-500:]
            logger.error(f"Failed to transcode recording {recording_id} for browser playback: {stderr_tail}")
            return source_path

        logger.info(f"Transcoded recording for browser playback: {playable_path}")
        return playable_path

    def delete_recording(self, recording_id: str):
        """Delete a recording"""
        db_recording = self.db.get_recording(recording_id)
        if not db_recording:
            raise ValueError(f"Recording not found: {recording_id}")
        
        # Delete file if it exists
        file_path = db_recording['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)

        root, ext = os.path.splitext(file_path)
        playable_path = f"{root}.browser{ext or '.mp4'}"
        if os.path.exists(playable_path):
            os.remove(playable_path)
        
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