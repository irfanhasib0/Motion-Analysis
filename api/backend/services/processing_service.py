import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class VideoProcessor(ABC):
    """Base class for video processors"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_active = False
        self.stop_event = threading.Event()
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Process a single frame"""
        pass
    
    def start(self):
        """Start the processor"""
        self.is_active = True
        self.stop_event.clear()
    
    def stop(self):
        """Stop the processor"""
        self.is_active = False
        self.stop_event.set()

class MotionDetectionProcessor(VideoProcessor):
    """Motion detection processor"""
    
    def __init__(self):
        super().__init__("motion_detection", "Detect motion in video stream")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.min_contour_area = 500
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Process frame for motion detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw motion areas
        processed_frame = frame.copy()
        motion_detected = False
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add status text
        status_text = "MOTION DETECTED" if motion_detected else "NO MOTION"
        color = (0, 255, 0) if motion_detected else (0, 0, 255)
        cv2.putText(processed_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return processed_frame

class EdgeDetectionProcessor(VideoProcessor):
    """Edge detection processor"""
    
    def __init__(self):
        super().__init__("edge_detection", "Apply Canny edge detection")
        self.low_threshold = 50
        self.high_threshold = 150
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Process frame for edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        # Convert back to BGR for consistency
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edges_bgr

class FaceDetectionProcessor(VideoProcessor):
    """Face detection processor"""
    
    def __init__(self):
        super().__init__("face_detection", "Detect faces in video stream")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scale_factor = 1.1
        self.min_neighbors = 5
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Process frame for face detection"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        processed_frame = frame.copy()
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(processed_frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Add face count
        cv2.putText(processed_frame, f'Faces: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed_frame

class ColorFilterProcessor(VideoProcessor):
    """Color filter processor"""
    
    def __init__(self):
        super().__init__("color_filter", "Apply color filters to video stream")
        self.filter_type = "none"  # none, sepia, grayscale, blue, green, red
    
    def process_frame(self, frame: np.ndarray, filter_type: str = "none", **kwargs) -> np.ndarray:
        """Process frame with color filter"""
        if filter_type != "none":
            self.filter_type = filter_type
        
        if self.filter_type == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif self.filter_type == "sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
            sepia_frame = cv2.transform(frame, sepia_filter)
            return np.clip(sepia_frame, 0, 255).astype(np.uint8)
        
        elif self.filter_type == "blue":
            blue_frame = frame.copy()
            blue_frame[:, :, 0] = np.minimum(blue_frame[:, :, 0] * 1.5, 255)  # Enhance blue channel
            return blue_frame
        
        elif self.filter_type == "green":
            green_frame = frame.copy()
            green_frame[:, :, 1] = np.minimum(green_frame[:, :, 1] * 1.5, 255)  # Enhance green channel
            return green_frame
        
        elif self.filter_type == "red":
            red_frame = frame.copy()
            red_frame[:, :, 2] = np.minimum(red_frame[:, :, 2] * 1.5, 255)  # Enhance red channel
            return red_frame
        
        return frame

class ProcessingService:
    def __init__(self):
        self.processors: Dict[str, VideoProcessor] = {
            "motion_detection": MotionDetectionProcessor(),
            "edge_detection": EdgeDetectionProcessor(),
            "face_detection": FaceDetectionProcessor(),
            "color_filter": ColorFilterProcessor()
        }
        self.active_processing: Dict[str, Dict] = {}  # camera_id -> processing info
        
    def get_available_processors(self) -> List[Dict[str, str]]:
        """Get list of available processors"""
        return [
            {
                "name": processor.name,
                "description": processor.description,
                "active": processor.is_active
            }
            for processor in self.processors.values()
        ]
    
    async def start_processing(self, camera_id: str, processor_type: str, params: Optional[Dict] = None):
        """Start processing on a camera stream"""
        if processor_type not in self.processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        if camera_id in self.active_processing:
            await self.stop_processing(camera_id)
        
        from main import recording_service  # Import here to avoid circular imports
        
        if camera_id not in recording_service.cameras:
            raise ValueError(f"Camera not found: {camera_id}")
        
        camera = recording_service.cameras[camera_id]
        processor = self.processors[processor_type]
        
        # Start processor
        processor.start()
        
        # Create processing thread
        processing_thread = threading.Thread(
            target=self._process_camera_stream,
            args=(camera_id, processor, params or {}),
            daemon=True
        )
        
        self.active_processing[camera_id] = {
            'processor_type': processor_type,
            'processor': processor,
            'thread': processing_thread,
            'params': params or {}
        }
        
        processing_thread.start()
        
        # Update camera status
        camera.processing_active = True
        camera.processing_type = processor_type
        
        logger.info(f"Started {processor_type} processing on camera {camera.name}")
    
    async def stop_processing(self, camera_id: str):
        """Stop processing on a camera stream"""
        if camera_id not in self.active_processing:
            return
        
        # Stop processor
        processor = self.active_processing[camera_id]['processor']
        processor.stop()
        
        # Wait for thread to finish
        thread = self.active_processing[camera_id]['thread']
        thread.join(timeout=5)
        
        del self.active_processing[camera_id]
        
        # Update camera status
        from main import recording_service
        if camera_id in recording_service.cameras:
            camera = recording_service.cameras[camera_id]
            camera.processing_active = False
            camera.processing_type = None
        
        logger.info(f"Stopped processing on camera {camera_id}")
    
    def _process_camera_stream(self, camera_id: str, processor: VideoProcessor, params: Dict):
        """Process camera stream in background thread"""
        from main import recording_service
        
        camera = recording_service.cameras[camera_id]
        
        try:
            # Open camera for processing
            if camera.camera_type.value == "webcam":
                cap = cv2.VideoCapture(int(camera.source))
            else:
                cap = cv2.VideoCapture(camera.source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera for processing: {camera.name}")
                return
            
            logger.info(f"Processing camera stream: {camera.name} with {processor.name}")
            
            while processor.is_active and not processor.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera.name}")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                try:
                    processed_frame = processor.process_frame(frame, **params)
                    # Here you could save processed frames, send to WebSocket, etc.
                    # For now, just processing without output
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    continue
                
                # Control processing rate
                time.sleep(1.0 / 15)  # 15 FPS processing
            
            cap.release()
            logger.info(f"Finished processing camera stream: {camera.name}")
            
        except Exception as e:
            logger.error(f"Error in processing thread for camera {camera.name}: {e}")
    
    def get_active_processors(self) -> Dict[str, str]:
        """Get currently active processors by camera"""
        return {
            camera_id: info['processor_type']
            for camera_id, info in self.active_processing.items()
        }
    
    def stop_all_processing(self):
        """Stop all active processing"""
        for camera_id in list(self.active_processing.keys()):
            asyncio.create_task(self.stop_processing(camera_id))