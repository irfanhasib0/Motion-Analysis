import cv2
import subprocess
import numpy as np
from typing import Union

source = 'rtsp://admin:L2CD8412@192.168.0.103:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'

class Capture:
    def __init__(self, source: Union[str, int], width:int =640, height:int =480, fps:int =30):
        self.source = source
        self.cam_type = None
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
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
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
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
            print(f"Failed to open RTSP stream: {e}")
        return self
    
    def is_opened(self):
        if self.cam_type == 'webcam':
            return self.cap and self.cap.isOpened()
        elif self.cam_type in ['rtsp', 'http']:
            return self.cap and self.cap.poll() is None
        return False
    
    def read_wcam(self):
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None
    
    def read_rtsp(self):
        if self.cap:
            frame_size = self.width * self.height * 3  # Assuming default resolution for now
            raw = self.cap.stdout.read(frame_size)
            if len(raw) == frame_size:
                frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
                return True, frame
        return False, None
    
    def release_wcam(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def release_rtsp(self):
        if self.cap:
            self.cap.terminate()
            self.cap = None
            
    def open(self):
        if self.cam_type == 'rtsp':
            return self.open_rtsp()
        elif self.cam_type == 'webcam':
            return self.open_wcam()
    
    def read(self):
        if self.cam_type == 'rtsp':
            return self.read_rtsp()
        elif self.cam_type == 'webcam':
            return self.read_wcam()
        else:
            raise ValueError(f"Unsupported camera type: {self.cam_type}")
        
    def release(self):
        if self.cam_type == 'rtsp':
            self.release_rtsp()
        elif self.cam_type == 'webcam':
            self.release_wcam()
        else:
            raise ValueError(f"Unsupported camera type: {self.cam_type}")



cap = Capture(source, width=640, height=480, fps=30)

while cap.is_opened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame")
        break
    
    # Process the frame (for demonstration, we'll just display it)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break