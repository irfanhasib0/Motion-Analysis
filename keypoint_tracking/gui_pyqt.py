import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QRadioButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QSlider, QInputDialog, QButtonGroup, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
#pip install PyQt5
# sudo apt-get install libxcb-xinerama0 libxcb-xinerama0-dev libxcb1 libx11-xcb1 libxrender1 libxi6 libsm6 libxext6
DISPLAY_W, DISPLAY_H = 640, 480

class VideoFlowPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video + Optical Flow Viewer (PyQt5)")
        self.cap = None
        self.total_frames = 0
        self.fps = 25.0
        self.frame_idx = 0
        self.playing = False
        self.speed = 1.0
        self.prev_gray = None
        self.flow_mode = "dense"

        # Layouts
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        display_layout = QHBoxLayout()

        # Controls
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_file)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn = QPushButton("<<")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn = QPushButton(">>")
        self.next_btn.clicked.connect(self.next_frame)
        self.faster_btn = QPushButton("Faster")
        self.faster_btn.clicked.connect(self.faster)
        self.slower_btn = QPushButton("Slower")
        self.slower_btn.clicked.connect(self.slower)
        self.jump_btn = QPushButton("Jump To")
        self.jump_btn.clicked.connect(self.jump_to_dialog)

        for btn in [self.open_btn, self.play_btn, self.prev_btn, self.next_btn, self.faster_btn, self.slower_btn, self.jump_btn]:
            control_layout.addWidget(btn)

        # Flow mode radio buttons
        mode_group = QButtonGroup(self)
        dense_radio = QRadioButton("Dense")
        dense_radio.setChecked(True)
        sparse_radio = QRadioButton("Sparse")
        mode_group.addButton(dense_radio)
        mode_group.addButton(sparse_radio)
        dense_radio.toggled.connect(lambda: self.set_flow_mode("dense"))
        sparse_radio.toggled.connect(lambda: self.set_flow_mode("sparse"))
        mode_layout.addWidget(QLabel("Flow:"))
        mode_layout.addWidget(dense_radio)
        mode_layout.addWidget(sparse_radio)

        # Video display
        self.left_label = QLabel()
        self.left_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        self.right_label = QLabel()
        self.right_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        display_layout.addWidget(self.left_label)
        display_layout.addWidget(self.right_label)

        # Status
        self.status = QLabel("No video loaded")

        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)

        # Assemble layout
        main_layout.addLayout(control_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(display_layout)
        main_layout.addWidget(self.status)
        self.setLayout(main_layout)

    def set_flow_mode(self, mode):
        self.flow_mode = mode

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if not path:
            return
        self.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status.setText("Failed to open: " + path)
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_idx = 0
        self.prev_gray = None
        self.status.setText(f"Loaded {path} | {self.total_frames} frames @ {self.fps:.2f} fps")
        self.show_frame(self.frame_idx)

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.schedule_next()
        else:
            self.timer.stop()

    def schedule_next(self):
        delay = max(1, int(1000 / (self.fps * max(0.25, self.speed))))
        self.timer.start(delay)

    def update(self):
        if not self.cap:
            return
        step = int(self.speed) if self.speed >= 1.0 else 1
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + step)
        self.show_frame(self.frame_idx)
        if self.playing and self.frame_idx < self.total_frames - 1:
            self.schedule_next()
        else:
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()

    def show_frame(self, idx):
        if not self.cap:
            return
        idx = int(max(0, min(self.total_frames - 1, idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            self.status.setText("Failed reading frame")
            return
        self.frame_idx = idx
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            flow_vis = np.zeros_like(frame_rgb)
        else:
            if self.flow_mode == "dense":
                flow_vis = self.compute_dense_flow(self.prev_gray, gray)
            else:
                flow_vis = self.compute_sparse_flow(self.prev_gray, gray, frame_bgr)
        self.prev_gray = gray
        frame_rgb_resized = cv2.resize(frame_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        flow_vis_resized = cv2.resize(flow_vis, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        self.left_label.setPixmap(self.cv2qt(frame_rgb_resized))
        self.right_label.setPixmap(self.cv2qt(flow_vis_resized))
        self.status.setText(f"Frame {self.frame_idx+1}/{self.total_frames} | Mode: {self.flow_mode} | Speed: {self.speed}x")

    def compute_dense_flow(self, prev_gray, gray):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[...,0] = (ang / 2).astype(np.uint8)
        hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsv[...,2] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def compute_sparse_flow(self, prev_gray, gray, frame_bgr):
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        vis_bgr = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        if p0 is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(15,15), maxLevel=2,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if p1 is None or st is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]
        mask = np.zeros_like(frame_bgr)
        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
            vis_bgr = cv2.circle(vis_bgr, (int(a), int(b)), 3, (0, 0, 255), -1)
        out = cv2.add(vis_bgr, mask)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb

    def next_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.setText("Play")
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
        self.show_frame(self.frame_idx)

    def prev_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.setText("Play")
        self.frame_idx = max(0, self.frame_idx - 1)
        self.show_frame(self.frame_idx)

    def faster(self):
        self.speed = min(8.0, self.speed * 2)
        self.status.setText(f"Speed: {self.speed}x")
        if self.playing:
            self.schedule_next()

    def slower(self):
        self.speed = max(0.25, self.speed / 2)
        self.status.setText(f"Speed: {self.speed}x")
        if self.playing:
            self.schedule_next()

    def jump_to_dialog(self):
        if not self.cap:
            return
        ans, ok = QInputDialog.getInt(self, "Jump To", f"Enter frame number (1-{self.total_frames}):", min=1, max=self.total_frames)
        if ok:
            self.playing = False
            self.play_btn.setText("Play")
            self.show_frame(ans - 1)

    def release(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def closeEvent(self, event):
        self.release()
        event.accept()

    def cv2qt(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoFlowPlayer()
    player.show()
    sys.exit(app.exec_())
