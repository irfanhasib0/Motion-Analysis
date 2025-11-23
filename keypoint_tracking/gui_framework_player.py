#!/usr/bin/env python3
"""
Video + Optical Flow Viewer using the flexible GUI Framework
Equivalent to gui.py and gui_pyqt.py but using the backend-agnostic framework.

Usage:
    python gui_framework_player.py            # Default to tkinter
    python gui_framework_player.py tkinter    # Use Tkinter backend
    python gui_framework_player.py pyqt5      # Use PyQt5 backend

Features:
- Video playback with play/pause/step controls
- Dense and sparse optical flow visualization  
- Speed control (0.25x to 8x)
- Frame jumping
- Backend switching without code changes
"""

import sys
import cv2
import time
import threading
import numpy as np
from gui_framework import GUIFramework
from gui_utils import GUIComponents, ImageProcessor

DISPLAY_W, DISPLAY_H = 640, 480

class VideoFlowPlayer:
    def __init__(self, backend='tkinter'):
        self.gui = GUIFramework(backend=backend)
        self.backend_name = backend
        
        # Video state
        self.cap = None
        self.total_frames = 0
        self.fps = 60.0#25.0
        self.frame_idx = 0
        self.playing = False
        self.speed = 1.0
        self.prev_gray = None
        self.flow_mode = "dense"
        
        # UI state
        self.setup_ui()
        self.frame_thread = threading.Thread(target=self._update_frame, daemon=True)
        # Performance tracking
        self.perf_info = {
            'update_frame': {
            'fps': 0.0,
            'ms': 0.0,
            'itr': 0.0,
            'time': time.time()
            },
            'gui_update': {
            'fps': 0.0,
            'itr': 0.0,
            'ms': 0.0,
            'time': time.time()
            }
        }
        
        # Performance display labels (will be created in setup_ui)
        self.perf_frame = None
        self.open_file(path='../../tokyo.mov')
    
    def calculate_performance(self,  key):
        """Calculate and update performance metrics using incremental frame counting"""
        current_time = time.time()
        
        if key not in self.perf_info:
            self.perf_info[key] = {
                'fps': 0.0,
                'ms': 0.0,
                'itr': 0.0,
                'time': current_time,
                'fps_time': current_time
            }
        
        self.perf_info[key]['ms']   = (current_time - self.perf_info[key]['time']) * 1000.0  # ms
        self.perf_info[key]['time'] = current_time
        
        self.perf_info[key]['itr'] += 1
        time_elapsed = current_time - self.perf_info[key]['fps_time']
        if time_elapsed >= 1.0:
            self.perf_info[key]['fps']  = self.perf_info[key]['itr'] / time_elapsed    
            self.perf_info[key]['itr'] = 0
            self.perf_info[key]['fps_time'] = current_time
        
        # Update performance labels
        self.gui.update_text(
            key, 
            f"Frame Update: {self.perf_info[key]['ms']:.1f}ms ({self.perf_info[key]['fps']:.1f} FPS)"
        )

    def _create_performance_display(self, parent):
        """Create performance monitoring labels"""        
        # Performance labels
        self.gui.create_label(
            parent, "perf_title", "Performance Monitoring:", 
            10, 5
        )
        
        self.gui.create_label(
            parent, "frame", 
            "Frame Update: 0.0ms (0.0 FPS)", 
            200, 5
        )
        
        self.gui.create_label(
            parent, "gui", 
            "GUI Update: 0.0ms (0.0 FPS)", 
            400, 5
        )

    def setup_ui(self):
        """Setup the user interface"""
        window_width = 1350
        window_height = 650
        
        self.window = self.gui.create_window(
            f"Video + Optical Flow Viewer ({self.backend_name.upper()})", 
            window_width, window_height
        )
        
        # Control buttons frame
        control_frame = self.gui.create_frame(
            self.window, "control_frame", 0, 0, window_width, 50
        )
        
        # Create control buttons
        self._create_control_buttons(control_frame)
        
        # Flow mode selection frame
        mode_frame = self.gui.create_frame(
            self.window, "mode_frame", 0, 60, window_width, 40
        )

        info_frame = self.gui.create_frame(
            self.window, "info_frame", 0, 110, window_width, 40
        )
        
        # Create flow mode controls
        self._create_flow_mode_controls(mode_frame)
        
        # Video display frame
        display_frame = self.gui.create_frame(
            self.window, "display_frame", 0, 150, window_width, DISPLAY_H + 20
        )
        
        # Create video displays
        self._create_video_displays(display_frame)
        
        # Status bar
        self.gui.create_label(
            self.window, "status_label", 
            "No video loaded", 10, window_height - 40
        )
        self._create_performance_display(info_frame)
        # Set initial radio button state
        self._set_initial_flow_mode()
    
    def _create_control_buttons(self, parent):
        """Create control buttons"""
        button_configs = [
            ("open_btn", "Open", self.open_file),
            ("play_btn", "Play", self.toggle_play),
            ("prev_btn", "<<", self.prev_frame),
            ("next_btn", ">>", self.next_frame),
            ("faster_btn", "Faster", self.faster),
            ("slower_btn", "Slower", self.slower),
            ("jump_btn", "Jump To", self.jump_to_dialog),
        ]
        
        # Use the improved control panel creation
        buttons = [(text, callback) for _, text, callback in button_configs]
        GUIComponents.create_control_panel(
            self.gui, parent, "control_buttons", 10, 5, 600, 40, buttons
        )
    
    def _create_flow_mode_controls(self, parent):
        """Create flow mode radio buttons"""
        self.gui.create_label(parent, "mode_label", "Flow Mode:", 10, 10)
        
        # Create radio button group for flow modes
        flow_options = [("Dense", "dense"), ("Sparse", "sparse")]
        GUIComponents.create_radio_group(
            self.gui, parent, "flow_mode", 100, 5, 
            flow_options, "dense", self.set_flow_mode
        )
    
    def _create_video_displays(self, parent):
        """Create video display labels"""
        # Left display - original video
        self.gui.create_label(
            parent, "left_title", "Original Video", 
            10, 5
        )
        self.gui.create_label(
            parent, "left_label", "", 
            10, 25, DISPLAY_W, DISPLAY_H
        )
        
        # Right display - optical flow
        self.gui.create_label(
            parent, "right_title", "Optical Flow", 
            30 + DISPLAY_W, 5
        )
        self.gui.create_label(
            parent, "right_label", "", 
            30 + DISPLAY_W, 25, DISPLAY_W, DISPLAY_H
        )
    
    def _set_initial_flow_mode(self):
        """Set initial flow mode selection"""
        # Get the first radio button (Dense) and select it
        dense_radio = self.gui.get_component("flow_mode_radio_0")
        if dense_radio and self.backend_name == 'tkinter':
            dense_radio.select()
        elif dense_radio and self.backend_name == 'pyqt5':
            dense_radio.setChecked(True)
    
    def set_flow_mode(self, mode):
        """Set the optical flow computation mode"""
        self.flow_mode = mode
        print(f"Flow mode changed to: {mode}")
        
        # Update the right display title
        title_text = f"Optical Flow - {mode.capitalize()}"
        self.gui.update_text("right_title", title_text)
        
        # Recompute current frame if video is loaded
        if self.cap and self.prev_gray is not None:
            self.show_frame(self.frame_idx)
    
    def open_file(self, path=None):
        if path is None:
            """Open video file dialog"""
            filetypes = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv"), 
                ("All files", "*.*")
            ]
            path = self.gui.show_file_dialog(filetypes)
        
        if path is None:
            return
        
        self.release_video()
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            error_msg = f"Failed to open: {path.split('/')[-1]}"
            self.gui.update_text("status_label", error_msg)
            self.cap = None
            return
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_idx = 0
        self.prev_gray = None
        
        # Update status
        filename = path.split('/')[-1]
        status_text = f"Loaded {filename} | {self.total_frames} frames @ {self.fps:.2f} fps"
        self.gui.update_text("status_label", status_text)
        
        # Show first frame
        self.show_frame(self.frame_idx)
        
        print(f"Video loaded: {filename}")
    
    def toggle_play(self):
        """Toggle play/pause"""
        if not self.cap:
            return
        
        self.playing = not self.playing
        if self.playing and not self.frame_thread.is_alive():
                self.frame_thread.start()
        
        # Update button text - find the play button
        play_btn = self.gui.get_component("control_buttons_btn_1")  # Play is the second button (index 1)
        if self.backend_name == 'tkinter':
            play_btn.config(text="Pause" if self.playing else "Play")
        else:  # PyQt5
            play_btn.setText("Pause" if self.playing else "Play")
        
        print(f"Playback {'started' if self.playing else 'paused'}")
    
    def _update_frame(self):
        while True:
            """Update to next frame during playback"""
            self.calculate_performance('frame')
            # Calculate frame step based on speed
            step = int(self.speed) if self.speed >= 1.0 else 1
            self.frame_idx = min(self.total_frames - 1, self.frame_idx + step)
            
            # Continue playback or stop at end
            if self.playing and self.frame_idx < self.total_frames - 1:
                self.show_frame(self.frame_idx)
            else:
                time.sleep(0.01)
    
    def _stop_playback(self):
        """Stop playback and update UI"""
        self.playing = False
        
        play_btn = self.gui.get_component("control_buttons_btn_1")  # Play is the second button (index 1)
        if self.backend_name == 'tkinter':
            play_btn.config(text="Play")
        else:
            play_btn.setText("Play")
    
    def show_frame(self, idx):
        """Display frame and compute optical flow"""
        if not self.cap:
            return
        
        # Clamp frame index
        idx = int(max(0, min(self.total_frames - 1, idx)))
        
        # Set video position and read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if not ret:
            self.gui.update_text("status_label", "Failed to read frame")
            return
        
        self.frame_idx = idx
        
        # Process frame
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        if self.prev_gray is None:
            flow_vis = np.zeros_like(frame_rgb)
        else:
            if self.flow_mode == "dense":
                flow_vis = self._compute_dense_flow(self.prev_gray, gray)
            else:
                flow_vis = self._compute_sparse_flow(self.prev_gray, gray, frame_bgr)
        
        self.prev_gray = gray
        
        # Resize images for display
        frame_resized = ImageProcessor.preprocess_for_display(
            frame_rgb, DISPLAY_W, DISPLAY_H, maintain_aspect=False
        )
        flow_resized = ImageProcessor.preprocess_for_display(
            flow_vis, DISPLAY_W, DISPLAY_H, maintain_aspect=False
        )
        
        # Update displays
        self.gui.update_image("left_label", frame_resized)
        self.gui.update_image("right_label", flow_resized)
        
        # Update status
        status_text = (f"Frame {self.frame_idx+1}/{self.total_frames} | "
                      f"Mode: {self.flow_mode} | Speed: {self.speed}x")
        self.gui.update_text("status_label", status_text)
    
    def _compute_dense_flow(self, prev_gray, gray):
        """Compute dense optical flow using Farneback method"""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Convert flow to HSV visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (ang / 2).astype(np.uint8)  # Hue from angle
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Saturation from magnitude
        hsv[..., 2] = 255  # Full value
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    
    def _compute_sparse_flow(self, prev_gray, gray, frame_bgr):
        """Compute sparse optical flow using Lucas-Kanade method"""
        # Detect corners to track
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, mask=None, maxCorners=200, qualityLevel=0.01,
            minDistance=7, blockSize=7
        )
        
        # Start with grayscale background
        vis_bgr = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        
        if p0 is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        
        # Calculate optical flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if p1 is None or st is None:
            return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        
        # Select good points
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]
        
        # Draw tracks
        mask = np.zeros_like(frame_bgr)
        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            # Draw line for trajectory
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1)
            # Draw point
            vis_bgr = cv2.circle(vis_bgr, (a, b), 3, (0, 0, 255), -1)
        
        # Combine visualization
        out = cv2.add(vis_bgr, mask)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    
    def next_frame(self):
        """Advance to next frame"""
        if not self.cap:
            return
        
        self._stop_playback()
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
        self.show_frame(self.frame_idx)
        print(f"Next frame: {self.frame_idx + 1}")
    
    def prev_frame(self):
        """Go back to previous frame"""
        if not self.cap:
            return
        
        self._stop_playback()
        self.frame_idx = max(0, self.frame_idx - 1)
        self.show_frame(self.frame_idx)
        print(f"Previous frame: {self.frame_idx + 1}")
    
    def faster(self):
        """Increase playback speed"""
        old_speed = self.speed
        self.speed = min(8.0, self.speed * 2)
        print(f"Speed changed: {old_speed}x -> {self.speed}x")
        
        # Update status immediately
        self._update_speed_status()
        
        # Restart timer if playing
        if self.playing:
            self.gui.stop_timer()
            self._schedule_next_frame()
    
    def slower(self):
        """Decrease playback speed"""
        old_speed = self.speed
        self.speed = max(0.25, self.speed / 2)
        print(f"Speed changed: {old_speed}x -> {self.speed}x")
        
        # Update status immediately
        self._update_speed_status()
        
        # Restart timer if playing
        if self.playing:
            self.gui.stop_timer()
            self._schedule_next_frame()
    
    def _update_speed_status(self):
        """Update speed in status bar"""
        status_text = f"Speed: {self.speed}x"
        if self.cap:
            status_text = (f"Frame {self.frame_idx+1}/{self.total_frames} | "
                          f"Mode: {self.flow_mode} | Speed: {self.speed}x")
        self.gui.update_text("status_label", status_text)
    
    def jump_to_dialog(self):
        """Show dialog to jump to specific frame"""
        if not self.cap:
            return
        
        frame_num = self.gui.show_input_dialog(
            "Jump To Frame", 
            f"Enter frame number (1-{self.total_frames}):",
            min_val=1, max_val=self.total_frames
        )
        
        if frame_num is not None:
            self._stop_playback()
            target_frame = frame_num - 1  # Convert to 0-based index
            self.show_frame(target_frame)
            print(f"Jumped to frame: {frame_num}")
    
    def release_video(self):
        """Release video capture resources"""
        self._stop_playback()
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Error releasing video: {e}")
            self.cap = None
        self.prev_gray = None
    
    def run(self):
        """Start the application"""
        print(f"Starting Video Flow Player with {self.backend_name.upper()} backend")
        print("Controls:")
        print("  Open - Load video file")
        print("  Play/Pause - Toggle playback")
        print("  <</>>, - Step frame by frame")  
        print("  Faster/Slower - Adjust playback speed")
        print("  Jump To - Jump to specific frame")
        print("  Dense/Sparse - Switch optical flow mode")
        print()
        
        try:
            self.gui.run()
        finally:
            self.release_video()

def main():
    """Main function with backend selection and error handling"""
    # Default backend
    backend = 'tkinter'
    
    # Parse command line argument
    if len(sys.argv) > 1:
        backend = sys.argv[1].lower()
        if backend not in ['tkinter', 'pyqt5']:
            print(f"Error: Invalid backend '{backend}'")
            print("Usage: python gui_framework_player.py [tkinter|pyqt5]")
            print("Supported backends: tkinter, pyqt5")
            sys.exit(1)
    
    print("=" * 60)
    print(f"Video + Optical Flow Viewer")
    print(f"Backend: {backend.upper()}")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python (cv2)")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if backend == 'tkinter':
        try:
            import tkinter
            from PIL import Image, ImageTk
        except ImportError as e:
            if "tkinter" in str(e):
                missing_deps.append("tkinter (usually included with Python)")
            else:
                missing_deps.append("Pillow (PIL)")
    
    elif backend == 'pyqt5':
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            missing_deps.append("PyQt5")
    
    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall missing packages:")
        for dep in missing_deps:
            if "opencv" in dep:
                print("  pip install opencv-python")
            elif "numpy" in dep:
                print("  pip install numpy")
            elif "Pillow" in dep:
                print("  pip install Pillow")
            elif "PyQt5" in dep:
                print("  pip install PyQt5")
        sys.exit(1)
    
    # Create and run application
    try:
        player = VideoFlowPlayer(backend=backend)
        player.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application closed")

if __name__ == "__main__":
    main()