import subprocess as sp
import numpy as np
import cv2

source = 'rtsp://admin:L2CD8412@192.168.1.30:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'
# First, probe your stream size once (or set known width/height)
W, H = 1920, 1080

cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "1000000",
    "-probesize", "1000000",
    "-i", source,
    "-an",                    # no audio
    "-vf", "fps=15",           # optional: reduce CPU / bandwidth
    "-pix_fmt", "bgr24",
    "-f", "rawvideo",
    "pipe:1"
]

p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**8)

frame_size = W * H * 3

while True:
    raw = p.stdout.read(frame_size)
    if len(raw) != frame_size:
        break
    frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
    cv2.imshow("rtsp", frame)
    if cv2.waitKey(1) == 27:
        break

p.terminate()
