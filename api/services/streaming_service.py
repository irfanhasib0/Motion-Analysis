import cv2
import asyncio
import threading
import time
from typing import Generator, Dict, Optional
import logging
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

class StreamingService:
    def __init__(self):
        self.db = DatabaseService()  # Use same database as recording service
        self.camera_streams: Dict[str, cv2.VideoCapture] = {}
        self.stream_locks: Dict[str, threading.Lock] = {}
        self.active_streams: Dict[str, bool] = {}  # Track active stream generators

    def generate_camera_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """Generate live video stream from camera"""
        # Get camera from database
        db_camera = self.db.get_camera(camera_id)
        logger.info(f"Generating stream for camera {camera_id}, status: {db_camera['status']}")
        
        # If there's already an active stream, close it first
        if camera_id in self.active_streams and self.active_streams[camera_id]:
            logger.info(f"Closing existing stream for camera {camera_id}")
            self.close_camera_stream(camera_id)
        
        # Mark stream as active
        self.active_streams[camera_id] = True
        
        # Initialize camera capture if not already done
        if camera_id not in self.camera_streams:
            self._init_camera_stream(camera_id, db_camera)
        
        cap = self.camera_streams.get(camera_id)
        if not cap or not cap.isOpened():
            # Try to start the camera first by updating status
            logger.warning(f"Camera {camera_id} not opened, attempting to start")
            self._start_camera(camera_id, db_camera)
            # Retry initialization
            self._init_camera_stream(camera_id, db_camera)
            cap = self.camera_streams.get(camera_id)
            if not cap or not cap.isOpened():
                self.active_streams[camera_id] = False
                raise ValueError(f"Failed to open camera stream: {camera_id}")
        
        # Get or create lock for this camera
        if camera_id not in self.stream_locks:
            self.stream_locks[camera_id] = threading.Lock()
        
        lock = self.stream_locks[camera_id]
        
        try:
            while self.active_streams.get(camera_id, False):
                with lock:
                    if not cap.isOpened() or not self.active_streams.get(camera_id, False):
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
            # Mark stream as inactive and clean up
            self.active_streams[camera_id] = False
            logger.info(f"Stream generator finished for camera {camera_id}")
            # Don't release the capture here as it might be used for recording or other streams

    def generate_recording_stream(self, recording_id: str) -> Generator[bytes, None, None]:
        """Generate video stream from recorded file"""
        # Get recording from database
        db_recording = self.db.get_recording(recording_id)
        if not db_recording:
            raise ValueError(f"Recording not found: {recording_id}")
        
        cap = cv2.VideoCapture(db_recording['file_path'])
        if not cap.isOpened():
            raise ValueError(f"Failed to open recording file: {db_recording['file_path']}")
        
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

    def _init_camera_stream(self, camera_id: str, db_camera: dict):
        """Initialize camera stream capture"""
        try:
            camera_type = db_camera.get('camera_type', 'webcam')
            logger.info(f"Initializing camera stream for {db_camera['name']} (ID: {camera_id}, Type: {camera_type}, Source: {db_camera['source']})")
            
            # Handle different camera types
            if db_camera['source'] in ['mock', 'test']:
                # For mock cameras, create a placeholder stream
                logger.info(f"Creating mock stream for {db_camera['name']}")
                self._create_mock_stream(camera_id, db_camera)
                return
            elif camera_type == "webcam":
                logger.info(f"Opening webcam with index: {db_camera['source']}")
                try:
                    webcam_index = int(db_camera['source'])
                    cap = cv2.VideoCapture(webcam_index)
                    # Test if webcam is actually working
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if not ret:
                            logger.warning(f"Webcam {webcam_index} opened but cannot read frames")
                            cap.release()
                            cap = None
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid webcam source '{db_camera['source']}': {e}")
                    cap = None
            elif camera_type == "recorded":
                # For recorded cameras, the source should be a file path
                logger.info(f"Opening recorded video file: {db_camera['source']}")
                cap = cv2.VideoCapture(db_camera['source'])
            else:
                # RTSP, IP camera, etc.
                logger.info(f"Opening camera stream: {db_camera['source']}")
                cap = cv2.VideoCapture(db_camera['source'])
            
            if cap and cap.isOpened():
                # Set camera properties for optimal streaming
                if db_camera.get('resolution'):
                    width, height = map(int, db_camera['resolution'].split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                if db_camera.get('fps'):
                    cap.set(cv2.CAP_PROP_FPS, db_camera['fps'])
                
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.camera_streams[camera_id] = cap
                logger.info(f"Successfully initialized camera stream for {db_camera['name']}")
            else:
                error_msg = f"Failed to open camera stream for {db_camera['name']} - VideoCapture could not be opened"
                logger.error(error_msg)
                # Create a mock stream as fallback for debugging
                logger.info(f"Creating mock stream as fallback for {db_camera['name']}")
                self._create_mock_stream(camera_id, db_camera)
                
        except Exception as e:
            logger.error(f"Error initializing camera stream {camera_id} ({db_camera['name']}): {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Create a mock stream as fallback
            logger.info(f"Creating mock stream as fallback due to error for {db_camera['name']}")
            self._create_mock_stream(camera_id, db_camera)

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

    def _create_mock_stream(self, camera_id: str, db_camera: dict):
        """Create a mock video stream for testing"""
        import numpy as np
        
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
        
        # Parse resolution for mock camera
        if db_camera.get('resolution'):
            width, height = map(int, db_camera['resolution'].split('x'))
        else:
            width, height = 640, 480
            
        mock_cap = MockCapture(width, height, db_camera['name'])
        self.camera_streams[camera_id] = mock_cap
        logger.info(f"Created mock stream for {db_camera['name']} at {width}x{height}")

    def _start_camera(self, camera_id: str, db_camera: dict):
        """Start a camera by updating its status"""
        try:
            # Test camera connection first
            source = db_camera['source']
            
            # Handle webcam source conversion
            try:
                source_int = int(source)
                cap = cv2.VideoCapture(source_int)
            except ValueError:
                cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                # Try alternative devices for webcams
                if source.isdigit():
                    for alt_device in [0, 2, 4]:
                        if alt_device != int(source):
                            cap = cv2.VideoCapture(alt_device)
                            if cap.isOpened():
                                logger.info(f"Using alternative device {alt_device} for camera {camera_id}")
                                # Update source in database
                                self.db.update_camera(camera_id, {'source': str(alt_device)})
                                break
                    
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                
                if ret:
                    # Update camera status
                    self.db.update_camera(camera_id, {'status': 'online'})
                    logger.info(f"Started camera: {db_camera['name']} ({camera_id})")
                    return True
            
            logger.warning(f"Camera {db_camera['name']} failed to open")
            self.db.update_camera(camera_id, {'status': 'offline'})
            return False
            
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            self.db.update_camera(camera_id, {'status': 'error'})
            return False

    def close_camera_stream(self, camera_id: str):
        """Close camera stream"""
        # First, stop any active stream generators
        if camera_id in self.active_streams:
            self.active_streams[camera_id] = False
            logger.info(f"Stopped active stream generator for camera {camera_id}")
        
        # Wait a moment for generators to finish
        import time
        time.sleep(0.2)  # Increased wait time
        
        # Force cleanup of camera capture
        if camera_id in self.camera_streams:
            cap = self.camera_streams[camera_id]
            try:
                if cap and hasattr(cap, 'isOpened') and cap.isOpened():
                    cap.release()
                    logger.info(f"Released camera capture for {camera_id}")
                # Force delete the capture object
                del cap
            except Exception as e:
                logger.warning(f"Error releasing camera {camera_id}: {e}")
            finally:
                del self.camera_streams[camera_id]
            
        if camera_id in self.stream_locks:
            del self.stream_locks[camera_id]
            
        # Force garbage collection to ensure resources are freed
        import gc
        gc.collect()
            
        logger.info(f"Closed camera stream for {camera_id}")

    def close_all_streams(self):
        """Close all camera streams"""
        for camera_id in list(self.camera_streams.keys()):
            self.close_camera_stream(camera_id)