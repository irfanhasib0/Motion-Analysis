#!/usr/bin/env python3
"""
Video + Optical Flow Viewer using the flexible GUI Framework
Equivalent to gui.py and gui_pyqt.py but using the backend-agnostic framework.

Usage:
    python gui_framework_player.py            # Default to pyqt5
    python gui_framework_player.py tkinter    # Use Tkinter backend
    python gui_framework_player.py pyqt5      # Use PyQt5 backend
    python gui_framework_player.py wxpython   # Use wxPython backend

Features:
- Video playback with play/pause/step controls
- Dense and sparse optical flow visualization  
- Speed control (0.25x to 8x)
- Frame jumping
- Backend switching without code changes
"""

import io
import os
# Fix Qt plugin path conflict with OpenCV
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import sys
import cv2
import time
import threading
import queue
import glob
import numpy as np
from pathlib import Path
from gui.gui_framework import GUIFramework
from gui.gui_comp import GUIComponents, ImageProcessor
#from trackers.pose_tracker import RTMPoseTracker
from improc.optical_flow import OpticalFlowTracker
import cProfile, pstats, io

DISPLAY_W, DISPLAY_H = 320, 240
CTRL_H = 320

sys.path.append('../cpp/build')
sys.path.append('../cpp/gui/build')
USE_CPP_GUI = False
#try:
#import gui_framework_cpp as gfc
# C++ equivalents

USE_CPP_GUI = False
if USE_CPP_GUI:
    print("Using C++ GUI Framework")
    from gui_framework_cpp import GUIFrameworkCpp as GUIFrameworkCpp
    from gui_framework_cpp import GUIComponents as GUIComponents
    from gui_framework_cpp import ImageProcessor as ImageProcessor
    print("Imported C++ GUI Framework modules")

#ipcam = "rtsp://admin:L2D841A1@192.168.0.102:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif

import motionflow_cpp

class GUI:
    def __init__(self, backend='tk'):
        if backend == 'qt' and USE_CPP_GUI:
            # Use C++ Qt GUI framework for Qt backend
            print("Initializing C++ Qt GUI Framework")
            self.gui = GUIFrameworkCpp()
            print("Using C++ Qt GUI Framework")
        else:
            self.gui = GUIFramework(backend=backend)
        self.perf_keys = ['frame', 'det', 'gui']
        self.info_keys = ['frame_idx']

        self.window_width  = 4* DISPLAY_W + 100
        self.window_height = CTRL_H + DISPLAY_H + 80

    def setup_ui(self):
        """Setup the user interface"""
        print("Setting up UI...")
        
        
        self.window = self.gui.create_window(
            f"Video + Optical Flow Viewer ({self.backend_name.upper()})", 
            self.window_width, self.window_height
        )
        
        # Control buttons frame
        control_frame = self.gui.create_frame(
            self.window, "control_frame", 0, 0, self.window_width, 50
        )
        
        # Flow mode selection frame
        mode_frame = self.gui.create_frame(
            self.window, "mode_frame", 0, 40, self.window_width, 40
        )

        #  Create file selector frame
        file_selector_frame = self.gui.create_frame(
            self.window, "file_selector_frame", -80, 80, self.window_width, 40
        )

        info_frame = self.gui.create_frame(
            self.window, "info_frame", 0, 120, self.window_width//2, 200
        )

        perf_frame = self.gui.create_frame(
            self.window, "perf_frame", self.window_width//2, 110, self.window_width//2, 200
        )

        # Video display frame
        display_frame = self.gui.create_frame(
            self.window, "display_frame", 0, 320, self.window_width, 320+DISPLAY_H + 20
        )
        
        # Create control buttons
        self._create_control_buttons(control_frame)
        
        self._create_file_selector(file_selector_frame)
        
        # Create flow mode controls
        self._create_flow_mode_controls(mode_frame)
        
        # Create video displays
        self._create_video_displays(display_frame)
        
        # Status bar
        self.gui.create_label(
            self.window, "status_label", 
            "No video loaded", 10, self.window_height - 40
        )
        self._create_info_display(info_frame)
        self._create_performance_display(perf_frame)
        # Set initial radio button state
        self._set_initial_flow_mode()
    
    def _create_performance_display(self, parent):
        """Create performance monitoring labels"""        
        # Performance labels
        self.gui.create_label(
            parent, "perf_title", "Performance", 
            10, 5, 250, 20
        )
        _h = 25
        for key in self.perf_keys:
            self.gui.create_label(
                parent, key, 
                f"{key.capitalize()} Update: 0.0ms (0.0 FPS)", 
                10, _h, 250, 20
            )
            _h += 20
    
    def _create_info_display(self, parent):
        """Create information display labels"""        
        # Information labels
        self.gui.create_label(
            parent, "info_title", "Info", 
            10, 5, 200, 20
        )
        
        _h = 25
        for key in self.info_keys:
            self.gui.create_entry(
                parent, key, 
                f"0".zfill(4), 
                20, _h, 50, 20
            )
            _h += 20
            
    def _create_control_buttons(self, parent):
        """Create control buttons"""
        button_configs = [
            ("open_btn", "Open", self.open_file),
            ("play_btn", "Play", self.toggle_play),
            ("prev_btn", "<<", self.prev_frame),
            ("next_btn", ">>", self.next_frame),
            ("faster_btn", "Faster", self.faster),
            ("slower_btn", "Slower", self.slower),
            ("clear_log_btn", "Clear Log", self.clear_log)
        ]
        
        # Use the improved control panel creation
        buttons = [(text, callback) for _, text, callback in button_configs]
        GUIComponents.create_control_panel(
            self.gui, parent, "control_buttons", 10, 5, 600, 40, buttons
        )

    def _create_file_selector(self, parent):
        # Get list of video files from current directory or a default path
        files = sorted(os.listdir(str(self.curr_source_dir)))
        GUIComponents.create_file_selector(
            self.gui, parent, files, "file_selector", 100, 10,
            callback_browse=self.browse_data,
            callback_load=self.load_data,
            callback_back=self.prev_dir,
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
    
    def _create_video_displays(self, parent, wpad = 20, hpad = 20):
        """Create video display areas"""
        x , y = wpad, hpad
        for i in range(4):
            self.gui.create_label(
                parent, f"title_frame_{i+1}", 
                f"Disp_{i+1}", 
                x, y - 15
            )
            self.gui.create_label(
                parent, f"frame_{i+1}", "", 
                x, y, 
                DISPLAY_W + x, DISPLAY_H + y
            )
            x += DISPLAY_W + wpad


class VideoFlowPlayer(GUI):
    def __init__(self, backend='tk', tracker=None):
        super().__init__(backend=backend)
        self.backend_name = backend
        
        # Video state
        self.cap = None
        self.total_frames = 0
        self.fps = 60.0#25.0
        self.frame_idx = 0
        self.playing = False
        self.speed = 8.0
        self.cap = None
        self.files = None
        self.frame_width = DISPLAY_W
        self.frame_height = DISPLAY_H
        self.curr_source_dir = Path('/media/irfan/TRANSCEND/action_data/')
        if not self.curr_source_dir.exists():
            self.curr_source_dir = Path('./data/')

        self.empty_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # Performance tracking
        self.perf_info = {}
        self.perf_lock = threading.Lock()
        
        self.tracker = tracker
        
        # Thread-safe queue for frame updates
        self.frame_queue = queue.Queue(maxsize=2)
        self.prof = cProfile.Profile()
        # UI state
        self.setup_ui()
        
        # Use timer for GUI updates instead of direct calls from worker thread
        self.gui.start_timer(self._process_frame_queue, 33)  # ~30 FPS GUI updates

        # Worker thread for inference (to avoid blocking GUI)
        self.frame_thread = threading.Thread(target=self._update_frame, daemon=True)
        self.frame_thread.start()
        #self.open_file(path='../../01_001.avi')
        self.open_file(path=self.curr_source_dir / 'stech/training/videos/01_001.avi')
        #self.open_file(path='/media/irfan/TRANSCEND/action_data/aven/training_videos/01.avi')
        self._show_frame()
        self.prof.enable()
    
    def calculate_performance(self,  key):
        """Calculate performance metrics using incremental frame counting and optionally update GUI."""
        current_time = time.perf_counter()
        
        with self.perf_lock:
            if key not in self.perf_info:
                self.perf_info[key] = {
                    'fps': 0.0,
                    'ms': 0.0,
                    'itr': 0.0,
                    'init_time': current_time,
                    'end_time': current_time,
                    'fps_time': current_time
                }
            
            self.perf_info[key]['ms']   = (self.perf_info[key]['end_time'] - self.perf_info[key]['init_time']) * 1000.0  # ms
            self.perf_info[key]['time'] = current_time
            
            self.perf_info[key]['itr'] += 1
            time_elapsed = current_time - self.perf_info[key]['fps_time']
            if time_elapsed >= 1.0:
                self.perf_info[key]['fps']  = self.perf_info[key]['itr'] / time_elapsed    
                self.perf_info[key]['itr'] = 0
                self.perf_info[key]['fps_time'] = current_time
            
            # Copy values for GUI update
            ms_val = self.perf_info[key]['ms']
            fps_val = self.perf_info[key]['fps']
        
        # Prepare text for GUI update (outside lock)
        text = f"{key.capitalize()} Update:".ljust(20) + f"{ms_val:.1f}ms ({fps_val:.1f} FPS)"
        self.gui.update_text(key, text)
    
    def _set_initial_flow_mode(self):
        """Set initial flow mode selection"""
        # C++ Qt path sets default in GUIComponents; skip manual selection
        if USE_CPP_GUI and self.backend_name == 'qt':
            return
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
    
    def open_file(self, path=None):
        """Open video file or image folder"""
        if path is None or path == False:
            """Open video file dialog"""
            filetypes = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv"), 
                ("All files", "*")
            ]
            
            # Show dialog to choose between file or folder
            choice = self.gui.show_message_box(
                "Select Input Type",
                "Choose input type:",
                buttons=["Video File", "Image Folder", "Cancel"]
            )
            
            if choice == "Video File":
                path = self.gui.show_file_dialog(filetypes=filetypes)
            elif choice == "Image Folder":
                path = self.gui.show_folder_dialog("Select Image Folder", self.curr_source_dir)
                if path and not path.endswith('/'):
                    path += '/'
            else:
                print('Operation cancelled.')
                return
        self.open_folder_data(path)

    def open_folder_data(self, path: Path):
        if str(path).endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.release_video()
            print(f"Opening video file: {path}")
            self.cap = cv2.VideoCapture(str(path))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)#640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)#480)
            print(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(self.cap.get(cv2.CAP_PROP_FPS))

            if not self.cap.isOpened():
                error_msg = f"Failed to open: {path.name}"
                self.gui.update_text("status_label", error_msg)
                self.cap = None
                return
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.files = None
            print(f"Video opened: {path} | {self.total_frames} frames @ {self.fps:.2f} fps")

            self.tracker.restart()

        if path.is_dir():
            for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']:
                self.files = sorted(glob.glob(os.path.join(str(path), f'*{ext}')))
                if len(self.files):
                    break
            if (self.files): 
                self.annot_files  = sorted(glob.glob(os.path.join(str(path), '*.npy')))
                self.total_frames = len(self.files)
                self.fps = 25.0
                self.cap = None
                print(f"Found {self.total_frames} image files.")
            else:
                print(f"Unsupported file type: {path}")
                self.browse_data(path = path)
                return

        self.frame_idx = 0
        # Update status
        filename = '/'.join(str(path).split('/')[-2:])
        status_text = f"Loaded {filename} | {self.total_frames} frames @ {self.fps:.2f} fps"
        self.gui.update_text("status_label", status_text)
    
    def browse_data(self, path = None):
        """Update current source directory and refresh file selector"""
        if path is None or path == False:
           path = self.gui.show_folder_dialog("Select Source Directory", str(self.curr_source_dir))
        if path is None or path == False:
            print("No directory selected.")
            return
        self.curr_source_dir = Path(path)
        files = sorted(os.listdir(self.curr_source_dir))
        self.gui.update_dropdown_options("file_selector", files)
        print(f"Source directory updated to: {self.curr_source_dir}")

    def load_data(self):
        path = self.curr_source_dir / self.gui.get_dropdown_value("file_selector")
        print(f"Loading: {str(path)}")
        self.open_folder_data(path)
    
    def prev_dir(self):
        """Go to parent directory and refresh file selector"""
        parent_dir = self.curr_source_dir.parent
        self.curr_source_dir = parent_dir
        files = sorted(os.listdir(self.curr_source_dir))
        self.gui.update_dropdown_options("file_selector", files)
        print(f"Moved to parent directory: {self.curr_source_dir}")
        
    def read_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        elif self.files is not None:
            if self.frame_idx < len(self.files):
                frame = cv2.imread(self.files[self.frame_idx])
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                if len(self.annot_files):
                    annot = np.load(self.annot_files[self.frame_idx])
                self.frame_idx += 1
                return frame
    
    def _stop_playback(self):
        """Stop playback and update UI"""
        self.playing = False
        # Update button text using backend-agnostic method
        self.gui.update_text("control_buttons_btn_1", "Play")

    def next_frame(self):
        """Advance to next frame"""
        if not self.cap:
            return
        
        self._stop_playback()
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self._show_frame()
        print(f"Next frame: {self.frame_idx + 1}")
    
    def prev_frame(self):
        """Go back to previous frame"""
        if not self.cap:
            return
        
        self._stop_playback()
        self.frame_idx = max(0, self.frame_idx - 2)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self._show_frame()
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
            # Ensure GUI updates continue
            self.gui.start_timer(self._process_frame_queue, 5)
    
    def slower(self):
        """Decrease playback speed"""
        old_speed = self.speed
        self.speed = max(0.1, self.speed / 2)
        print(f"Speed changed: {old_speed}x -> {self.speed}x")
        
        # Update status immediately
        self._update_speed_status()
        
        # Restart timer if playing
        if self.playing:
            self.gui.stop_timer()
            # Ensure GUI updates continue
            self.gui.start_timer(self._process_frame_queue, 5)
            
    
    def _update_speed_status(self):
        """Update speed in status bar"""
        status_text = f"Speed: {self.speed}x"
        if self.cap:
            status_text = (f"Frame {self.frame_idx+1}/{self.total_frames} | "
                          f"Mode: {self.flow_mode} | Speed: {self.speed}x")
        self.gui.update_text("status_label", status_text)
    
    def clear_log(self):
        """Clear the log output"""
        self.tracker.coreset.clear_viz()

    def toggle_play(self):
        """Toggle play/pause"""
        if self.cap is None and self.files is None:
            return

        if not self.playing:
            self.frame_idx = int(self.gui.get_text('frame_idx'))
            print(f"Playback resumed at frame: {self.frame_idx}")
            if self.cap is not None:
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                    if self.frame_idx == 0:
                        self.tracker.restart()
                except Exception as e:
                    print(f"Error setting frame position: {e}")

        self.playing = not self.playing
        if self.playing:
            if not self.frame_thread.is_alive():
                # Recreate worker thread if it has terminated
                self.frame_thread = threading.Thread(target=self._update_frame, daemon=True)
                self.frame_thread.start()
            print("Playback started")
        self.gui.update_text("control_buttons_btn_1", "Pause" if self.playing else "Play")
    
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

    def _update_frame(self):
        """Worker thread: Read and process frames, put results in queue"""
        while True:
            # Do not update GUI from worker thread; only compute metrics
            if not self.playing or (self.cap is None and self.files is None):
                time.sleep(0.01)
                continue
            self.calculate_performance('det')
            self.perf_info['det']['init_time'] = time.perf_counter()
            self._show_frame()
            self.perf_info['det']['end_time'] = time.perf_counter()

    def _show_frame(self):   
        step = 1
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + step)
        
        # Read frame
        if self.frame_idx >= self.total_frames - 1:
            self.toggle_play()
            self.frame_idx = 0
        
        frame_bgr = self.read_frame()
        
        if frame_bgr is None:
            print("End of video or failed to read frame")
            self.toggle_play()
            return
        
        #pose_viz = self.tracker.compute_bbox_pose(frame_bgr.copy())
        pose_viz, pts, mem_viz1, mem_viz2 = self.tracker.detect(frame_bgr.copy())
        cv2.putText(pose_viz, f"Frame: {self.frame_idx+1}/{self.total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_resized = ImageProcessor.preprocess_for_display(
            frame_bgr, DISPLAY_W, DISPLAY_H, maintain_aspect=False
        )
        flow_resized = ImageProcessor.preprocess_for_display(
            pose_viz, DISPLAY_W, DISPLAY_H, maintain_aspect=False
        )
        mem_viz_resized1 = ImageProcessor.preprocess_for_display(
            mem_viz1, DISPLAY_W, DISPLAY_H, maintain_aspect=False
        )
        mem_viz_resized2 = ImageProcessor.preprocess_for_display(
            mem_viz2, DISPLAY_W, DISPLAY_H, maintain_aspect=False
        )

        # Put frame data in queue (non-blocking, skip if full)
        try:
            self.frame_queue.put_nowait({
                'frame_1': frame_resized,
                'frame_2': flow_resized,
                'frame_3': mem_viz_resized1,
                'frame_4': mem_viz_resized2,
                'idx': self.frame_idx,
                'total': self.total_frames,
                'mode': self.flow_mode,
                'speed': self.speed
            })
        except queue.Full:
            print("Frame queue full, skipping frame", self.frame_idx)
            pass  # Skip frame if queue is full
        
        # Control playback speed (no GUI updates from worker thread)
        if self.speed < 8.0:
            time.sleep(max(0.001, 1.0 / (self.fps * self.speed)))
    
    def _process_frame_queue(self):
        """Timer callback on main thread: Update GUI with queued frame data"""
        # Begin frame timing
        self.calculate_performance('frame')
        self.perf_info['frame']['init_time'] = time.perf_counter()
        
        try:
            # Get frame data from queue (non-blocking)
            frame_data = self.frame_queue.get_nowait()
        
            self.gui.update_image("frame_1", frame_data['frame_1'])
            self.gui.update_image("frame_2", frame_data['frame_2'])
            self.gui.update_image("frame_3", frame_data['frame_3'])
            self.gui.update_image("frame_4", frame_data['frame_4'])
            # Update frame index entry on main thread
            self.gui.update_text('frame_idx', str(frame_data['idx']).zfill(5))
        
            status_text = (f"Frame {frame_data['idx']+1}/{frame_data['total']} | "
                          f"Mode: {frame_data['mode']} | Speed: {frame_data['speed']}x")
            self.gui.update_text("status_label", status_text)
        
        except queue.Empty:
            pass
        
        # End frame timing and update both frame and det labels from main thread
        self.perf_info['frame']['end_time'] = time.perf_counter()
        
    
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
        
        self.gui.run()

def main():
    """Main function with backend selection and error handling"""
    # Default backend
    backend = 'qt'
    
    # Parse command line argument
    if len(sys.argv) > 1:
        backend = sys.argv[1].lower()
        if backend not in ['tk', 'qt', 'wx']:
            print(f"Error: Invalid backend '{backend}'")
            print("Usage: python gui_framework_player.py [tk|qt|wx]")
            print("Supported backends: tk, qt, wx")
            sys.exit(1)
    
    print("=" * 60)
    print(f"Video + Optical Flow Viewer")
    print(f"Backend: {backend.upper()}")
    print("=" * 60)
    
    import cv2
    import numpy
    
    if backend == 'tk':
        import tkinter
        from PIL import Image, ImageTk
    
    elif backend == 'qt':
        # Prefer PySide6 (Qt for Python); framework will fall back to PyQt5
        #os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '1'
        #from PyQt5.QtWidgets import QApplication  # noqa: F401
        from PySide6.QtWidgets import QApplication  # noqa: F401
        #sudo apt install -y libxcb-cursor0
    
    elif backend == 'wx':
        import wx
        os.environ['GDK_BACKEND'] = 'x11'
        os.environ['GDK_RENDERING'] = '1'
        os.environ['GDK_SYNCHRONIZE'] = '1'
        os.environ['QT_X11_NO_MITSHM'] = '1'
    
    # Initialize tracker
    #tracker = RTMPoseTracker(device='cuda', conf_threshold=0.2, det_fw='torch', pose_fw='torch')
    #tracker = MovenetPoseTracker()
    #tracker = BlazePoseTracker()
    tracker  = OpticalFlowTracker()
    #tracker  = motionflow_cpp.OpticalFlowTrackerCpp("sparse")
    
    # Create and run application
    player = VideoFlowPlayer(backend=backend, tracker=tracker)
    player.run()
    
if __name__ == "__main__":
    main()
