"""
Video Flow Player using the flexible GUI Framework
Demonstrates backend switching capabilities between Tkinter and PyQt5.

Usage:
    python video_player_framework.py tkinter
    python video_player_framework.py pyqt5
"""

import sys
import cv2
import numpy as np
from gui_framework import GUIFramework

DISPLAY_W, DISPLAY_H = 640, 480

class VideoFlowPlayer:
    def __init__(self, backend='tkinter'):
        self.gui = GUIFramework(backend=backend)
        self.backend_name = backend
        
        # Video state
        self.cap = None
        self.total_frames = 0
        self.fps = 25.0
        self.frame_idx = 0
        self.playing = False
        self.speed = 1.0
        self.prev_gray = None
        self.flow_mode = "dense"
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.window = self.gui.create_window(
            f"Video + Optical Flow Viewer ({self.backend_name.upper()})", 
            1350, 600
        )
        
        # Control frame
        control_frame = self.gui.create_frame(self.window, "control_frame", 0, 0, 1350, 50)
        
        # Control buttons
        self.gui.create_button(control_frame, "open_btn", "Open", self.open_file, 10, 10)
        self.gui.create_button(control_frame, "play_btn", "Play", self.toggle_play, 80, 10)
        self.gui.create_button(control_frame, "prev_btn", "<<", self.prev_frame, 150, 10)
        self.gui.create_button(control_frame, "next_btn", ">>", self.next_frame, 190, 10)
        self.gui.create_button(control_frame, "faster_btn", "Faster", self.faster, 230, 10)
        self.gui.create_button(control_frame, "slower_btn", "Slower", self.slower, 290, 10)
        self.gui.create_button(control_frame, "jump_btn", "Jump To", self.jump_to_dialog, 350, 10)
        
        # Flow mode selection
        mode_frame = self.gui.create_frame(self.window, "mode_frame", 0, 60, 1350, 40)
        self.gui.create_label(mode_frame, "mode_label", "Flow Mode:", 10, 10)
        self.gui.create_radiobutton(mode_frame, "dense_radio", "Dense", "flow_mode", "dense", self.set_flow_mode)
        self.gui.create_radiobutton(mode_frame, "sparse_radio", "Sparse", "flow_mode", "sparse", self.set_flow_mode)
        
        # Video display area
        display_frame = self.gui.create_frame(self.window, "display_frame", 0, 110, 1350, 480)
        
        # Video labels
        self.gui.create_label(display_frame, "left_label", "", 10, 10, DISPLAY_W, DISPLAY_H)
        self.gui.create_label(display_frame, "right_label", "", 20 + DISPLAY_W, 10, DISPLAY_W, DISPLAY_H)
        
        # Status bar
        self.gui.create_label(self.window, "status_label", "No video loaded", 10, 570)
        
        # Set initial radio button state
        if self.backend_name == 'tkinter':
            # For tkinter, we need to access the radio variable directly
            dense_radio = self.gui.get_component("dense_radio")
            dense_radio.select()
    
    def set_flow_mode(self, mode):
        """Set the optical flow mode"""
        self.flow_mode = mode
        print(f"Flow mode set to: {mode}")
    
    def open_file(self):
        """Open video file dialog"""
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        path = self.gui.show_file_dialog(filetypes)
        
        if not path:
            return
            
        self.release()
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            self.gui.update_text("status_label", f"Failed to open: {path}")
            self.cap = None
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_idx = 0
        self.prev_gray = None
        
        status_text = f"Loaded {path.split('/')[-1]} | {self.total_frames} frames @ {self.fps:.2f} fps"
        self.gui.update_text("status_label", status_text)
        self.show_frame(self.frame_idx)
    
    def toggle_play(self):
        """Toggle play/pause"""
        if not self.cap:
            return
            
        self.playing = not self.playing
        play_btn = self.gui.get_component("play_btn")
        
        if self.backend_name == 'tkinter':
            play_btn.config(text="Pause" if self.playing else "Play")
        else:  # PyQt5
            play_btn.setText("Pause" if self.playing else "Play")
            
        if self.playing:
            self.schedule_next()
        else:
            self.gui.stop_timer()
    
    def schedule_next(self):
        """Schedule next frame update"""
        delay = max(1, int(1000 / (self.fps * max(0.25, self.speed))))
        self.gui.start_timer(self.update, delay)
    
    def update(self):
        """Update to next frame during playback"""
        if not self.cap:
            return
            
        step = int(self.speed) if self.speed >= 1.0 else 1
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + step)
        self.show_frame(self.frame_idx)
        
        if self.playing and self.frame_idx < self.total_frames - 1:
            self.schedule_next()
        else:
            self.playing = False
            play_btn = self.gui.get_component("play_btn")
            if self.backend_name == 'tkinter':
                play_btn.config(text="Play")
            else:
                play_btn.setText("Play")
            self.gui.stop_timer()
    
    def show_frame(self, idx):
        """Display frame and optical flow visualization"""
        if not self.cap:
            return
            
        idx = int(max(0, min(self.total_frames - 1, idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if not ret:
            self.gui.update_text("status_label", "Failed reading frame")
            return
            
        self.frame_idx = idx
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Compute optical flow
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            flow_vis = np.zeros_like(frame_rgb)
        else:
            if self.flow_mode == "dense":
                flow_vis = self.compute_dense_flow(self.prev_gray, gray)
            else:
                flow_vis = self.compute_sparse_flow(self.prev_gray, gray, frame_bgr)
        
        self.prev_gray = gray
        
        # Resize for display
        frame_rgb_resized = self.gui.resize_image(frame_rgb, DISPLAY_W, DISPLAY_H)
        flow_vis_resized = self.gui.resize_image(flow_vis, DISPLAY_W, DISPLAY_H)
        
        # Update display
        self.gui.update_image("left_label", frame_rgb_resized)
        self.gui.update_image("right_label", flow_vis_resized)
        
        status_text = f"Frame {self.frame_idx+1}/{self.total_frames} | Mode: {self.flow_mode} | Speed: {self.speed}x"
        self.gui.update_text("status_label", status_text)
    
    def compute_dense_flow(self, prev_gray, gray):
        """Compute dense optical flow visualization"""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (ang / 2).astype(np.uint8)
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsv[..., 2] = 255
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    
    def compute_sparse_flow(self, prev_gray, gray, frame_bgr):
        """Compute sparse optical flow visualization"""
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, mask=None, maxCorners=200, qualityLevel=0.01,
            minDistance=7, blockSize=7
        )
        
        vis_bgr = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        if p0 is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None, winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
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
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    
    def next_frame(self):
        """Go to next frame"""
        if not self.cap:
            return
        self.stop_playback()
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
        self.show_frame(self.frame_idx)
    
    def prev_frame(self):
        """Go to previous frame"""
        if not self.cap:
            return
        self.stop_playback()
        self.frame_idx = max(0, self.frame_idx - 1)
        self.show_frame(self.frame_idx)
    
    def faster(self):
        """Increase playback speed"""
        self.speed = min(8.0, self.speed * 2)
        self.update_speed_status()
        if self.playing:
            self.schedule_next()
    
    def slower(self):
        """Decrease playback speed"""
        self.speed = max(0.25, self.speed / 2)
        self.update_speed_status()
        if self.playing:
            self.schedule_next()
    
    def update_speed_status(self):
        """Update speed in status"""
        self.gui.update_text("status_label", f"Speed: {self.speed}x")
    
    def jump_to_dialog(self):
        """Show dialog to jump to specific frame"""
        if not self.cap:
            return
            
        frame_num = self.gui.show_input_dialog(
            "Jump To", f"Enter frame number (1-{self.total_frames}):",
            min_val=1, max_val=self.total_frames
        )
        
        if frame_num is not None:
            self.stop_playback()
            self.show_frame(frame_num - 1)
    
    def stop_playback(self):
        """Stop playback and update UI"""
        self.playing = False
        play_btn = self.gui.get_component("play_btn")
        if self.backend_name == 'tkinter':
            play_btn.config(text="Play")
        else:
            play_btn.setText("Play")
        self.gui.stop_timer()
    
    def release(self):
        """Release video capture"""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
    
    def run(self):
        """Start the application"""
        self.gui.run()

def main():
    """Main function with backend selection"""
    backend = 'tkinter'  # Default backend
    
    if len(sys.argv) > 1:
        backend = sys.argv[1].lower()
        if backend not in ['tkinter', 'pyqt5']:
            print("Usage: python video_player_framework.py [tkinter|pyqt5]")
            print("Invalid backend specified. Using tkinter as default.")
            backend = 'tkinter'
    
    print(f"Starting Video Flow Player with {backend.upper()} backend...")
    
    try:
        player = VideoFlowPlayer(backend=backend)
        player.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure you have the required dependencies installed:")
        if backend == 'tkinter':
            print("  - tkinter (usually comes with Python)")
            print("  - Pillow (PIL)")
        else:
            print("  - PyQt5")
        print("  - OpenCV (cv2)")
        print("  - NumPy")

if __name__ == "__main__":
    main()