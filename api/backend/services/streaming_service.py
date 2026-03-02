import gc
import os
import cv2
import json
import numpy as np
import asyncio
import threading
import time
from typing import Generator, Dict, Optional, Any, Union
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
        self.stream_locks: Dict[str, threading.Lock] = {}
        self.active_streams: Dict[str, str] = {}  # camera_id -> active stream token
        self.active_processing_streams: Dict[str, str] = {}  # camera_id -> active processing stream token
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._latest_frame_seq: Dict[str, int] = {}
        self._stream_frame_index: Dict[str, int] = {}
        self._latest_viz: Dict[str, np.ndarray] = {}
        self._latest_res: Dict[str, Dict[str, Any]] = {}
        self._fps_stats: Dict[str, Dict[str, float]] = {}

    def _update_loop_fps(self, stream_key: str) -> float:
        now = time.time()
        stats = self._fps_stats.get(stream_key)
        if stats is None:
            self._fps_stats[stream_key] = {
                "window_start": now,
                "count": 1.0,
                "fps": 0.0,
            }
            return 0.0

        stats["count"] += 1.0
        elapsed = now - stats["window_start"]
        if elapsed >= 1.0:
            stats["fps"] = stats["count"] / elapsed
            stats["count"] = 0.0
            stats["window_start"] = now
        return stats["fps"]

    def _draw_fps_overlay(self, frame: np.ndarray, fps_value: float, res: dict = {}) -> np.ndarray:
        if frame is None:
            return frame

        if not frame.flags.writeable or not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame).copy()

        texts = [f"FPS: {fps_value:.1f}"]
        if len(res):
            texts += [f" | Vel: {res['vel']}" \
                      f" | Diff: {res['bg_diff']}"]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        margin = 10
        y_ofs  = margin
        x = frame.shape[1] - 300
        y = margin
        for text in texts:
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            frame = cv2.putText(frame, text, (x, y + y_ofs), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            y_ofs += text_h + margin
            
        return frame
        
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
        quality = int(getattr(self, 'jpeg_quality', 70) or 70)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
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

        # Create stream token; if an old stream exists for this camera, this takes over.
        # Old generator loop will exit when it sees token mismatch.
        stream_token = f"{time.time_ns()}:{threading.get_ident()}"
        previous_token = self.active_streams.get(camera_id)
        if previous_token:
            logger.info(f"Taking over existing stream for camera {camera_id}")
        self.active_streams[camera_id] = stream_token
        
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

        while cap.is_opened() and self.active_streams.get(camera_id) == stream_token:
            with lock:
                ret, frame = cap.read()

                if not ret:
                    yield self.generate_failure_frame("Failed to Read Frame")
                    continue
                # Store original frame for recording consumers
                self._latest_frames[camera_id] = frame
                self._latest_frame_seq[camera_id] = self._latest_frame_seq.get(camera_id, 0) + 1
                self._stream_frame_index[camera_id] = self._stream_frame_index.get(camera_id, 0) + 1
                frame_index = self._stream_frame_index[camera_id]

            # Resize frame if needed for better streaming performance
            frame  = self._resize_frame_for_streaming(frame)
            processing_stride = max(1, int(getattr(self, 'processing_stride', 1) or 1))
            should_run_tracker = (processing_stride == 1) or (frame_index % processing_stride == 0)

            if should_run_tracker:
                frame, viz1, _res = tracker.detect(frame)
                res = {'vel': 0, 'bg_diff': 0, 'ts': time.time()}
                for key in _res.keys():
                    res['vel'] = max(_res[key]['vel'], res['vel'])
                    res['bg_diff'] = max(_res[key]['bg_diff'], res['bg_diff'])
                with lock:
                    self._latest_viz[camera_id] = viz1
                    self._latest_res[camera_id] = res
            else:
                with lock:
                    viz1 = self._latest_viz.get(camera_id)
                    res = self._latest_res.get(camera_id, {'vel': 0, 'bg_diff': 0, 'ts': time.time()})
                    if viz1 is None:
                        viz1 = frame
                        self._latest_viz[camera_id] = viz1

            frame_fps = self._update_loop_fps(f"{camera_id}:primary")
            frame = self._draw_fps_overlay(frame, frame_fps)
            buffer = self.frame_to_bytes(frame)

            yield buffer

        # Only clear if this stream is still the active owner
        if self.active_streams.get(camera_id) == stream_token:
            self.active_streams.pop(camera_id, None)
    
    def generate_processing_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """Generate processed video stream from camera"""
        # Get camera from database
        db_camera = self.db.get_camera(camera_id)

        if not db_camera:
            logger.warning(f"Camera not found: {camera_id}")
            return self.generate_failure_frame(f"Camera {camera_id} Not Found")

        stream_token = f"{time.time_ns()}:{threading.get_ident()}"
        previous_token = self.active_processing_streams.get(camera_id)
        if previous_token:
            logger.info(f"Taking over existing processing stream for camera {camera_id}")
        self.active_processing_streams[camera_id] = stream_token
        
        lock = self.stream_locks.get(camera_id)

        while camera_id in self._camera_trackers and self.active_processing_streams.get(camera_id) == stream_token:
            if lock is not None:
                with lock:
                    processed_frame = getattr(self, '_latest_viz', {}).get(camera_id, None)
            else:
                processed_frame = getattr(self, '_latest_viz', {}).get(camera_id, None)
            if processed_frame is None:
                yield self.generate_failure_frame("No Processed Frame Available")
                time.sleep(1.0 / 30)
                continue

            output_frame = processed_frame.copy()
            processing_fps = self._update_loop_fps(f"{camera_id}:processing")
            output_frame = self._draw_fps_overlay(output_frame, processing_fps, res=getattr(self, '_latest_res', {}).get(camera_id, {'vel': 0, 'bg_diff': 0}))

            # Resize frame if needed for better streaming performance
            #processed_frame = self._resize_frame_for_streaming(frame)
            buffer = self.frame_to_bytes(output_frame)
            
            yield buffer
            
            # Small delay to control frame rate
            time.sleep(1.0 / 30)  # 30 FPS max

        if self.active_processing_streams.get(camera_id) == stream_token:
                self.active_processing_streams.pop(camera_id, None)
            
            
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
    
    def generate_result_json_stream(self, camera_id: str) -> Generator[Dict[str, Union[int, float]], None, None]:
        """Generate JSON stream of processing results for a camera"""
        while camera_id in self._camera_trackers:
            res = getattr(self, '_latest_res', {}).get(camera_id, None)
            if res is not None:
                yield json.dumps(res)
            time.sleep(1.0)  # Update every second