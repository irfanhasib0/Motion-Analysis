import cv2
import subprocess
import sys
sys.path.append('src')
from improc import optical_flow

W, H, FPS = 640, 480, 30
RTSP_URL = "rtsp://0.0.0.0:8554/camera1"

cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)

encoder = "libx264"

tracker = optical_flow.OpticalFlowTracker()

cmd = [
    "ffmpeg",
    "-hide_banner", "-nostdin", "-loglevel", "verbose", "-stats",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{W}x{H}",
    "-r", str(FPS),
    "-i", "-",  # stdin
    "-an",
    "-vf", "format=yuv420p",
    "-pix_fmt", "yuv420p",
    "-c:v", encoder,
    "-preset", "ultrafast", "-tune", "zerolatency",
    "-b:v", "1500k",
    "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
    "-f", "rtsp",
    "-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp",
    "-muxdelay", "0", "-muxpreload", "0",
    RTSP_URL,
]

p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame, _, res1, res2 = tracker.detect(frame)
    cv2.putText(frame, "processed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    p.stdin.write(frame.tobytes())

cap.release()
p.stdin.close()
p.wait()
