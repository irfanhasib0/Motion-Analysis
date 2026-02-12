import cv2
import numpy as np

W, H, FPS = 640, 480, 30
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)

gst = (
    "appsrc ! videoconvert ! "
    "v4l2h264enc extra-controls=\"controls,video_bitrate=1000000;\" ! "
    "h264parse config-interval=1 ! "
    "rtspclientsink location=rtsp://127.0.0.1:8554/camera1"
)

out = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, FPS, (W, H), True)
if not out.isOpened():
    raise RuntimeError("Failed to open GStreamer VideoWriter. Check GStreamer + encoder availability.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ---- processing example: draw timestamp ----
    cv2.putText(frame, "processed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()
