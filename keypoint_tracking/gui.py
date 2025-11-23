# Simple Tkinter GUI video player with dense/sparse optical flow displays.
# Buttons: Open, Play/Pause, <<, >>, Faster, Slower, Jump To
# Flow modes: Dense (Farneback) and Sparse (Shi-Tomasi + LK)
import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# Pillow 10 removed Image.ANTIALIAS; select a compatible resampling filter
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

DISPLAY_W, DISPLAY_H = 640, 480

class VideoFlowPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Video + Optical Flow Viewer")
        # Video state
        self.cap = None
        self.total_frames = 0
        self.fps = 25.0
        self.frame_idx = 0
        self.playing = False
        self.speed = 1.0  # playback speed factor: 0.25,0.5,1,2,4...
        self.prev_gray = None
        self.flow_mode = tk.StringVar(value="dense")  # "dense" or "sparse"

        # UI layout using place()
        control_frame = tk.Frame(master)
        control_frame.place(relx=0, rely=0, relwidth=1, height=40)

        tk.Button(control_frame, text="Open", command=self.open_file).pack(side=tk.LEFT)
        self.play_btn = tk.Button(control_frame, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT)
        tk.Button(control_frame, text="<<", command=self.prev_frame).pack(side=tk.LEFT)
        tk.Button(control_frame, text=">>", command=self.next_frame).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Faster", command=self.faster).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Slower", command=self.slower).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Jump To", command=self.jump_to_dialog).pack(side=tk.LEFT)

        mode_frame = tk.Frame(master)
        mode_frame.place(relx=0, rely=0.05, relwidth=1, height=30)
        tk.Label(mode_frame, text="Flow:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Dense", variable=self.flow_mode, value="dense").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Sparse", variable=self.flow_mode, value="sparse").pack(side=tk.LEFT)

        display_frame = tk.Frame(master)
        display_frame.place(relx=0, rely=0.12, relwidth=1, relheight=0.78)
        # Left: original video, Right: processed output
        self.left_label = tk.Label(display_frame)
        # place at fixed positions inside display_frame
        self.left_label.place(x=10, y=10, width=DISPLAY_W, height=DISPLAY_H)
        self.right_label = tk.Label(display_frame)
        self.right_label.place(x=10 + DISPLAY_W + 10, y=10, width=DISPLAY_W, height=DISPLAY_H)

        self.status = tk.Label(master, text="No video loaded", anchor="w")
        self.status.place(relx=0, rely=0.92, relwidth=1, height=24)

        self._after_id = None
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not path:
            return
        self.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status.config(text="Failed to open: " + path)
            self.cap = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_idx = 0
        self.prev_gray = None
        self.status.config(text=f"Loaded {path} | {self.total_frames} frames @ {self.fps:.2f} fps")
        self.show_frame(self.frame_idx)

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.schedule_next()

    def schedule_next(self):
        if self._after_id:
            self.master.after_cancel(self._after_id)
        delay = max(1, int(1000 / (self.fps * max(0.25, self.speed))))  # ms
        self._after_id = self.master.after(delay, self.update)

    def update(self):
        if not self.cap:
            return
        # For speed > 1, skip frames
        step = int(self.speed) if self.speed >= 1.0 else 1
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + step)
        self.show_frame(self.frame_idx)
        if self.playing and self.frame_idx < self.total_frames - 1:
            self.schedule_next()
        else:
            self.playing = False
            self.play_btn.config(text="Play")

    def show_frame(self, idx):
        if not self.cap:
            return
        idx = int(max(0, min(self.total_frames - 1, idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            self.status.config(text="Failed reading frame")
            return
        self.frame_idx = idx
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Prepare processed output
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            flow_vis = np.zeros_like(frame_rgb)
        else:
            if self.flow_mode.get() == "dense":
                flow_vis = self.compute_dense_flow(self.prev_gray, gray)
            else:
                flow_vis = self.compute_sparse_flow(self.prev_gray, gray, frame_bgr)
        self.prev_gray = gray
        # Resize frames for display
        frame_rgb_resized = cv2.resize(frame_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        flow_vis_resized = cv2.resize(flow_vis, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

        left_img = ImageTk.PhotoImage(Image.fromarray(frame_rgb_resized).resize((DISPLAY_W, DISPLAY_H), RESAMPLE))
        right_img = ImageTk.PhotoImage(Image.fromarray(flow_vis_resized).resize((DISPLAY_W, DISPLAY_H), RESAMPLE))
        self.left_label.img = left_img
        self.left_label.config(image=left_img)
        self.right_label.img = right_img
        self.right_label.config(image=right_img)
        self.status.config(text=f"Frame {self.frame_idx+1}/{self.total_frames} | Mode: {self.flow_mode.get()} | Speed: {self.speed}x")

    def compute_dense_flow(self, prev_gray, gray):
        # Farneback dense optical flow, visualize with HSV mapping
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[...,0] = (ang / 2).astype(np.uint8)             # OpenCV HSV hue [0,179]
        hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsv[...,2] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def compute_sparse_flow(self, prev_gray, gray, frame_bgr):
        # Shi-Tomasi + LK sparse optical flow; draw tracks on a BGR image then convert to RGB for display
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        # use a BGR image so drawing colors match the frame_bgr mask
        vis_bgr = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        if p0 is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(15,15), maxLevel=2,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if p1 is None or st is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]
        # Draw tracks on BGR canvas
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
        self.play_btn.config(text="Play")
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
        self.show_frame(self.frame_idx)

    def prev_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="Play")
        self.frame_idx = max(0, self.frame_idx - 1)
        self.show_frame(self.frame_idx)

    def faster(self):
        # multiply speed by 2 up to 8
        self.speed = min(8.0, self.speed * 2)
        self.status.config(text=f"Speed: {self.speed}x")
        if self.playing:
            self.schedule_next()

    def slower(self):
        # divide speed by 2 down to 0.25
        self.speed = max(0.25, self.speed / 2)
        self.status.config(text=f"Speed: {self.speed}x")
        if self.playing:
            self.schedule_next()

    def jump_to_dialog(self):
        if not self.cap:
            return
        ans = simpledialog.askinteger("Jump To", f"Enter frame number (1-{self.total_frames}):", minvalue=1, maxvalue=self.total_frames)
        if ans is not None:
            self.playing = False
            self.play_btn.config(text="Play")
            self.show_frame(ans - 1)

    def release(self):
        if self._after_id:
            try:
                self.master.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def on_close(self):
        self.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFlowPlayer(root)
    root.mainloop()