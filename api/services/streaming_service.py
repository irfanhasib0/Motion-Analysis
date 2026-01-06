import cv2
import asyncio
import threading
import time
from typing import Generator, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class StreamingService:
    def __init__(self):
        self.camera_streams: Dict[str, cv2.VideoCapture] = {}
        self.stream_locks: Dict[str, threading.Lock] = {}

    def generate_camera_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """Generate live video stream from camera"""
        from main import recording_service  # Import here to avoid circular imports
        
        if camera_id not in recording_service.cameras:
            raise ValueError(f"Camera not found: {camera_id}")
        
        camera = recording_service.cameras[camera_id]
        
        # Initialize camera capture if not already done
        if camera_id not in self.camera_streams:
            self._init_camera_stream(camera_id, camera)
        
        cap = self.camera_streams.get(camera_id)
        if not cap or not cap.isOpened():
            raise ValueError(f"Failed to open camera stream: {camera_id}")
        
        # Get or create lock for this camera
        if camera_id not in self.stream_locks:
            self.stream_locks[camera_id] = threading.Lock()
        
        lock = self.stream_locks[camera_id]
        
        try:
            while True:
                with lock:
                    if not cap.isOpened():
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame from camera {camera_id}")
                        continue
                
                # Resize frame if needed for better streaming performance
                frame = self._resize_frame_for_streaming(frame)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to control frame rate
                time.sleep(1.0 / 30)  # 30 FPS max
                
        except Exception as e:
            logger.error(f"Error in camera stream {camera_id}: {e}")
        finally:
            # Don't release the capture here as it might be used for recording
            pass

    def generate_recording_stream(self, recording_id: str) -> Generator[bytes, None, None]:
        """Generate video stream from recorded file"""
        from main import recording_service  # Import here to avoid circular imports
        
        if recording_id not in recording_service.recordings:
            raise ValueError(f"Recording not found: {recording_id}")
        
        recording = recording_service.recordings[recording_id]
        
        cap = cv2.VideoCapture(recording.file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open recording file: {recording.file_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # End of video, loop back to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize frame if needed
                frame = self._resize_frame_for_streaming(frame)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Control playback speed
                time.sleep(1.0 / 30)  # 30 FPS
                
        except Exception as e:
            logger.error(f"Error streaming recording {recording_id}: {e}")
        finally:
            cap.release()

    def _init_camera_stream(self, camera_id: str, camera):
        """Initialize camera stream capture"""
        try:
            if camera.camera_type.value == "webcam":
                cap = cv2.VideoCapture(int(camera.source))
            else:
                cap = cv2.VideoCapture(camera.source)
            
            if cap.isOpened():
                # Set camera properties for optimal streaming
                if camera.resolution:
                    width, height = map(int, camera.resolution.split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                if camera.fps:
                    cap.set(cv2.CAP_PROP_FPS, camera.fps)
                
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.camera_streams[camera_id] = cap
                logger.info(f"Initialized camera stream for {camera.name}")
            else:
                logger.error(f"Failed to open camera stream for {camera.name}")
                
        except Exception as e:
            logger.error(f"Error initializing camera stream {camera_id}: {e}")

    def _resize_frame_for_streaming(self, frame, max_width: int = 1280):
        """Resize frame for optimal streaming performance"""
        height, width = frame.shape[:2]
        
        if width > max_width:
            # Calculate new height to maintain aspect ratio
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame

    def close_camera_stream(self, camera_id: str):
        """Close camera stream"""
        if camera_id in self.camera_streams:
            cap = self.camera_streams[camera_id]
            if cap.isOpened():
                cap.release()
            del self.camera_streams[camera_id]
            
        if camera_id in self.stream_locks:
            del self.stream_locks[camera_id]
            
        logger.info(f"Closed camera stream for {camera_id}")

    def close_all_streams(self):
        """Close all camera streams"""
        for camera_id in list(self.camera_streams.keys()):
            self.close_camera_stream(camera_id)