"""
Common GUI utilities and helper functions for the flexible GUI framework.
Provides pre-built components and common patterns.
"""

import cv2
import numpy as np
from gui_framework import GUIFramework

class GUIComponents:
    """Collection of common GUI component builders"""
    
    @staticmethod
    def create_control_panel(gui, parent, name, x, y, width, height, buttons):
        """
        Create a control panel with multiple buttons
        
        Args:
            gui: GUIFramework instance
            parent: Parent widget
            name: Panel name
            x, y, width, height: Panel dimensions
            buttons: List of (text, callback) tuples
        """
        panel = gui.create_frame(parent, name, x, y, width, height)
        
        # Better spacing calculation for PyQt5
        if gui.backend_type.value == 'pyqt5':
            # Use adaptive spacing based on button text length
            current_x = 5
            for i, (text, callback) in enumerate(buttons):
                btn_name = f"{name}_btn_{i}"
                btn = gui.create_button(panel, btn_name, text, callback, current_x, 5)
                
                # Calculate next position based on button width + spacing
                btn_width = max(60, len(text) * 8 + 20)  # Estimate width from text
                current_x += btn_width + 10  # Add spacing between buttons
        else:
            # Original approach for Tkinter
            button_width = width // len(buttons) if len(buttons) > 0 else width
            for i, (text, callback) in enumerate(buttons):
                btn_x = i * button_width + 5
                btn_name = f"{name}_btn_{i}"
                gui.create_button(panel, btn_name, text, callback, btn_x, 5)
        
        return panel
    
    @staticmethod
    def create_radio_group(gui, parent, name, x, y, options, default=None, callback=None):
        """
        Create a radio button group
        
        Args:
            gui: GUIFramework instance
            parent: Parent widget
            name: Group name
            x, y: Position
            options: List of (text, value) tuples
            default: Default selected value
            callback: Callback function for selection changes
        """
    @staticmethod
    def create_radio_group(gui, parent, name, x, y, options, default=None, callback=None):
        """
        Create a radio button group
        
        Args:
            gui: GUIFramework instance
            parent: Parent widget
            name: Group name
            x, y: Position
            options: List of (text, value) tuples
            default: Default selected value
            callback: Callback function for selection changes
        """
        group_name = f"{name}_group"
        
        # Better spacing for different backends
        if gui.backend_type.value == 'pyqt5':
            current_x = x
            for i, (text, value) in enumerate(options):
                radio_name = f"{name}_radio_{i}"
                radio = gui.create_radiobutton(parent, radio_name, text, group_name, value, callback)
                radio.move(current_x, y)
                
                # Calculate next position with proper spacing
                radio_width = max(80, len(text) * 8 + 40)  # Account for radio button circle
                current_x += radio_width
                
                # Set default selection for first option or specified default
                if (default is None and i == 0) or value == default:
                    radio.setChecked(True)
        else:
            # Original approach for Tkinter
            for i, (text, value) in enumerate(options):
                radio_x = x + (i * 80)  # Space buttons 80px apart
                radio_name = f"{name}_radio_{i}"
                radio = gui.create_radiobutton(parent, radio_name, text, group_name, value, callback)
                
                # Set default selection for first option or specified default
                if (default is None and i == 0) or value == default:
                    radio.select()
        
        return group_name
    
    @staticmethod
    def create_image_viewer(gui, parent, name, x, y, width, height, title=None):
        """
        Create an image viewer with optional title
        
        Args:
            gui: GUIFramework instance
            parent: Parent widget
            name: Viewer name
            x, y, width, height: Viewer dimensions
            title: Optional title above the image
        """
        viewer_dict = {}
        
        if title:
            title_label = gui.create_label(parent, f"{name}_title", title, x, y)
            viewer_dict['title'] = title_label
            image_y = y + 25  # Offset for title
        else:
            image_y = y
        
        image_label = gui.create_label(parent, f"{name}_image", "", x, image_y, width, height)
        viewer_dict['image'] = image_label
        viewer_dict['name'] = name
        
        return viewer_dict

class ImageProcessor:
    """Common image processing utilities for GUI applications"""
    
    @staticmethod
    def preprocess_for_display(image, target_width, target_height, maintain_aspect=True):
        """
        Preprocess image for GUI display
        
        Args:
            image: Input image (BGR or RGB)
            target_width, target_height: Target dimensions
            maintain_aspect: Whether to maintain aspect ratio
        """
        if image is None:
            return None
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR and convert to RGB
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image.copy()
        
        # Resize image
        if maintain_aspect:
            h, w = image.shape[:2]
            aspect = w / h
            
            if aspect > target_width / target_height:
                # Width is limiting factor
                new_width = target_width
                new_height = int(target_width / aspect)
            else:
                # Height is limiting factor
                new_height = target_height
                new_width = int(target_height * aspect)
            
            resized = cv2.resize(display_image, (new_width, new_height))
            
            # Create centered image on black background
            result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return result
        else:
            return cv2.resize(display_image, (target_width, target_height))
    
    @staticmethod
    def create_grid_overlay(image, grid_size=50, color=(255, 255, 255), thickness=1):
        """Add grid overlay to image"""
        result = image.copy()
        h, w = image.shape[:2]
        
        # Vertical lines
        for x in range(0, w, grid_size):
            cv2.line(result, (x, 0), (x, h), color, thickness)
        
        # Horizontal lines  
        for y in range(0, h, grid_size):
            cv2.line(result, (0, y), (w, y), color, thickness)
        
        return result
    
    @staticmethod
    def add_text_overlay(image, text, position=(10, 30), font_scale=0.7, color=(255, 255, 255), thickness=2):
        """Add text overlay to image"""
        result = image.copy()
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return result

class VideoPlayer:
    """Reusable video player component"""
    
    def __init__(self, gui, parent, name, x, y, width, height):
        self.gui = gui
        self.name = name
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 25.0
        self.playing = False
        
        # Create UI components
        self.viewer = GUIComponents.create_image_viewer(gui, parent, name, x, y, width, height)
        
        control_buttons = [
            ("Play", self.toggle_play),
            ("Stop", self.stop),
            ("Prev", self.prev_frame),
            ("Next", self.next_frame)
        ]
        self.controls = GUIComponents.create_control_panel(
            gui, parent, f"{name}_controls", x, y + height + 10, width, 40, control_buttons
        )
    
    def load_video(self, path):
        """Load video file"""
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.current_frame = 0
            self.show_frame(0)
            return True
        return False
    
    def show_frame(self, frame_idx):
        """Display specific frame"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame_idx
            # Process image for display
            processed = ImageProcessor.preprocess_for_display(
                frame, 
                self.viewer['image'].width if hasattr(self.viewer['image'], 'width') else 320,
                self.viewer['image'].height if hasattr(self.viewer['image'], 'height') else 240
            )
            self.gui.update_image(f"{self.name}_image", processed)
    
    def toggle_play(self):
        """Toggle play/pause"""
        self.playing = not self.playing
        if self.playing:
            self.play()
        else:
            self.pause()
    
    def play(self):
        """Start playback"""
        if self.cap and self.playing:
            self.gui.start_timer(self.next_frame, int(1000 / self.fps))
    
    def pause(self):
        """Pause playback"""
        self.gui.stop_timer()
    
    def stop(self):
        """Stop playback and return to beginning"""
        self.playing = False
        self.pause()
        self.show_frame(0)
    
    def next_frame(self):
        """Advance to next frame"""
        if self.current_frame < self.total_frames - 1:
            self.show_frame(self.current_frame + 1)
            if self.playing:
                self.play()  # Continue playback
        else:
            self.playing = False
            self.pause()
    
    def prev_frame(self):
        """Go back to previous frame"""
        if self.current_frame > 0:
            self.show_frame(self.current_frame - 1)

def demo_components():
    """Demonstrate the common GUI components"""
    
    # You can change this to test different backends
    gui = GUIFramework(backend='tkinter')  # or 'pyqt5'
    
    window = gui.create_window("GUI Components Demo", 800, 600)
    
    # Demo control panel
    def button_callback(btn_name):
        print(f"Button {btn_name} clicked!")
    
    buttons = [
        ("Button 1", lambda: button_callback("1")),
        ("Button 2", lambda: button_callback("2")), 
        ("Button 3", lambda: button_callback("3")),
        ("Button 4", lambda: button_callback("4"))
    ]
    
    GUIComponents.create_control_panel(gui, window, "demo_panel", 10, 10, 400, 50, buttons)
    
    # Demo radio group
    def radio_callback(value):
        print(f"Radio selected: {value}")
        gui.update_text("status_label", f"Selected: {value}")
    
    options = [("Option A", "a"), ("Option B", "b"), ("Option C", "c")]
    GUIComponents.create_radio_group(gui, window, "demo_radio", 10, 80, options, "a", radio_callback)
    
    # Demo image viewer
    viewer = GUIComponents.create_image_viewer(gui, window, "demo_viewer", 10, 120, 320, 240, "Demo Image")
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    sample_with_overlay = ImageProcessor.add_text_overlay(sample_image, "Sample Image", (10, 30))
    gui.update_image("demo_viewer_image", sample_with_overlay)
    
    # Status label
    gui.create_label(window, "status_label", "Ready - Click buttons to test", 10, 380)
    
    gui.run()

if __name__ == "__main__":
    demo_components()