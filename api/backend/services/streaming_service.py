import gc
import os
import cv2
import numpy as np
import asyncio
import threading
import time
from typing import Generator, Dict, Optional
import logging
#from services.database_service import DatabaseService
from services.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Create a simple mock stream that generates frames programmatically
class MockCapture:
    def __init__(self, width=640, height=480, camera_name="Mock Camera"):
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.frame_count = 0
        self.start_time = time.time()
        
    def isOpened(self):
        return True
        
    def read(self):
        # Generate a test pattern frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Create a gradient background
        for i in range(self.height):
            color_intensity = int((i / self.height) * 255)
            frame[i, :] = [color_intensity // 3, color_intensity // 2, color_intensity]
        
        # Add moving circle
        current_time = time.time() - self.start_time
        circle_x = int((self.width / 2) + 100 * np.sin(current_time))
        circle_y = int((self.height / 2) + 50 * np.cos(current_time))
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 255), -1)
        
        # Add text
        cv2.putText(frame, f"Mock Camera: {self.camera_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, self.height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.frame_count += 1
        return True, frame
        
    def release(self):
        pass
        
    def set(self, prop, value):
        pass
'''
# Parse resolution for mock camera
if db_camera.get('resolution'):
    width, height = map(int, db_camera['resolution'].split('x'))
else:
    width, height = 640, 480
    
mock_cap = MockCapture(width, height, db_camera['name'])
self._camera_streams[camera_id] = mock_cap
logger.info(f"Created mock stream for {db_camera['name']} at {width}x{height}")
'''

class StreamingService:
    def __init__(self):
        #self.db = DatabaseService()  # Original database service for cameras and recordings
        self.db = ConfigManager()  # Use same YAML-backed DB as recording service
        self.stream_locks: Dict[str, threading.Lock] = {}
        self.active_streams: Dict[str, bool] = {}  # Track active stream generators
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._latest_viz: Dict[str, np.ndarray] = {}
        
    def generate_failure_frame(self, msg: str = "Camera Unavailable"):
        failure_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame for errors
        w, h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_COMPLEX, 0.7, 1)[0]
        x = (failure_frame.shape[1] - w) // 2
        y = (failure_frame.shape[0] + h) // 2
        failure_frame = cv2.putText(failure_frame, msg, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 100, 100), 1)
        failure_frame = self.frame_to_bytes(failure_frame)
        return failure_frame
    
    def generate_blank_image(self, msg:str = ''):
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        w, h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_COMPLEX, 0.7, 1)[0]
        x = (blank_frame.shape[1] - w) // 2
        y = (blank_frame.shape[0] + h) // 2
        blank_frame = cv2.putText(blank_frame, msg, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 100, 100), 1)
        ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buffer.tobytes()
    
    def frame_to_bytes(self, frame) -> bytes:
        """Convert a video frame to bytes for streaming"""
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        buffer = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return buffer
    
    def generate_camera_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """Generate live video stream from camera"""
        # Get camera from database
        db_camera = self.db.get_camera(camera_id)
        
        if not db_camera:
            logger.warning(f"Camera not found: {camera_id}")
            return self.generate_failure_frame(f"Camera {camera_id} Not Found")
        
        logger.info(f"Generating stream for camera {camera_id}, status: {db_camera['status']}")
        
        # Initialize camera capture if not already done
        if camera_id not in self._camera_streams:
            scc = self.start_camera(camera_id)
            if not scc:
                return self.generate_failure_frame("Camera Failed to Start")
        
        cap = self._camera_streams.get(camera_id)
        tracker = self._camera_trackers.get(camera_id)

        # Get or create lock for this camera
        if camera_id not in self.stream_locks:
            self.stream_locks[camera_id] = threading.Lock()
        
        lock = self.stream_locks[camera_id]
        
        while cap.is_opened():
            with lock:
                ret, frame = cap.read()
                
                if not ret:
                    yield self.generate_failure_frame("Failed to Read Frame")
                    continue
                # Store original frame for recording consumers
                self._latest_frames[camera_id] = frame
            
            # Resize frame if needed for better streaming performance
            frame  = self._resize_frame_for_streaming(frame)
            frame, _, viz1, viz2  = tracker.detect(frame)
            self._latest_viz[camera_id] = viz1
            buffer = self.frame_to_bytes(frame)
            
            yield buffer
            
            # Small delay to control frame rate
            #time.sleep(1.0 / 30)  # 30 FPS max
    
    def generate_processing_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """Generate processed video stream from camera"""
        # Get camera from database
        db_camera = self.db.get_camera(camera_id)
        
        lock = self.stream_locks.get(camera_id)

        while camera_id in self._camera_trackers:
            #with lock:
            processed_frame = getattr(self, '_latest_viz', {}).get(camera_id, None)
            if processed_frame is None:
                yield self.generate_failure_frame("No Processed Frame Available")
                time.sleep(1.0 / 30)
                continue
            # Resize frame if needed for better streaming performance
            #processed_frame = self._resize_frame_for_streaming(frame)
            buffer = self.frame_to_bytes(processed_frame)
            
            yield buffer
            
            # Small delay to control frame rate
            time.sleep(1.0 / 30)  # 30 FPS max


    def generate_recording_stream(self, recording_id: str) -> Generator[bytes, None, None]:
        """Generate video stream from recorded file"""
        # Get recording from database
        db_recording = self.db.get_recording(recording_id)
        if not db_recording:
            raise ValueError(f"Recording not found: {recording_id}")
        
        # Resolve to absolute path to avoid CWD issues
        file_path = db_recording['file_path']
        abs_path = os.path.abspath(file_path)
        
        # Retry opening in case the writer is finalizing the file
        cap = None
        for _ in range(5):
            cap = cv2.VideoCapture(abs_path)
            if cap.isOpened():
                break
            time.sleep(0.2)
        
        if not cap or not cap.isOpened():
            raise ValueError(f"Failed to open recording file: {abs_path}")
        
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

    def _resize_frame_for_streaming(self, frame, max_width: int = 640):
        """Resize frame for optimal streaming performance"""
        height, width = frame.shape[:2]
        
        if width > max_width:
            # Calculate new height to maintain aspect ratio
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame