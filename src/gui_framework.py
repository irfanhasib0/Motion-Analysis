"""
Flexible GUI Framework - Backend Agnostic Interface
Supports both Tkinter and PyQt5 backends with seamless switching.

Usage:
    from gui_framework import GUIFramework
    
    # Use Tkinter backend
    gui = GUIFramework(backend='tkinter')
    
    # Use PyQt5 backend
    gui = GUIFramework(backend='pyqt5')
"""

import sys
import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

class GUIBackend(Enum):
    TKINTER = "tkinter"
    PYQT5 = "pyqt5"

class AbstractGUIBackend(ABC):
    """Abstract base class for GUI backends"""
    
    @abstractmethod
    def create_window(self, title, width, height):
        pass
    
    @abstractmethod
    def create_button(self, parent, text, callback, x=None, y=None):
        pass
    
    @abstractmethod
    def create_label(self, parent, text="", x=None, y=None, width=None, height=None):
        pass
    
    @abstractmethod
    def create_radiobutton(self, parent, text, group, value, callback=None):
        pass
    
    @abstractmethod
    def create_frame(self, parent, x=None, y=None, width=None, height=None):
        pass
    
    @abstractmethod
    def update_image(self, label, image_array):
        pass
    
    @abstractmethod
    def update_text(self, label, text):
        pass
    
    @abstractmethod
    def start_timer(self, callback, interval):
        pass
    
    @abstractmethod
    def stop_timer(self):
        pass
    
    @abstractmethod
    def show_file_dialog(self, filetypes=None):
        pass
    
    @abstractmethod
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def quit(self):
        pass

class TkinterBackend(AbstractGUIBackend):
    """Tkinter implementation of GUI backend"""
    
    def __init__(self):
        import tkinter as tk
        from tkinter import filedialog, simpledialog
        from PIL import Image, ImageTk
        
        self.tk = tk
        self.filedialog = filedialog
        self.simpledialog = simpledialog
        self.Image = Image
        self.ImageTk = ImageTk
        
        # Handle Pillow version compatibility
        try:
            self.RESAMPLE = Image.Resampling.LANCZOS
        except AttributeError:
            self.RESAMPLE = Image.LANCZOS
            
        self.root = None
        self.timer_id = None
        self.radio_vars = {}
        
    def create_window(self, title, width, height):
        self.root = self.tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        return self.root
    
    def create_button(self, parent, text, callback, x=None, y=None):
        btn = self.tk.Button(parent, text=text, command=callback)
        if x is not None and y is not None:
            btn.place(x=x, y=y)
        else:
            btn.pack(side=self.tk.LEFT)
        return btn
    
    def create_label(self, parent, text="", x=None, y=None, width=None, height=None):
        label = self.tk.Label(parent, text=text)
        if x is not None and y is not None:
            kwargs = {'x': x, 'y': y}
            if width: kwargs['width'] = width
            if height: kwargs['height'] = height
            label.place(**kwargs)
        else:
            label.pack()
        return label
    
    def create_radiobutton(self, parent, text, group, value, callback=None):
        if group not in self.radio_vars:
            self.radio_vars[group] = self.tk.StringVar(value=value)
        
        def radio_callback():
            if callback:
                callback(value)
        
        radio = self.tk.Radiobutton(
            parent, text=text, 
            variable=self.radio_vars[group], 
            value=value,
            command=radio_callback
        )
        radio.pack(side=self.tk.LEFT)
        return radio
    
    def create_frame(self, parent, x=None, y=None, width=None, height=None):
        frame = self.tk.Frame(parent)
        if x is not None and y is not None:
            kwargs = {'x': x, 'y': y}
            if width: kwargs['width'] = width
            if height: kwargs['height'] = height
            frame.place(**kwargs)
        else:
            frame.pack()
        return frame
    
    def update_image(self, label, image_array):
        """Update label with numpy array image (RGB format expected)"""
        h, w = image_array.shape[:2]
        if len(image_array.shape) == 3:
            img = self.Image.fromarray(image_array)
        else:
            img = self.Image.fromarray(image_array, mode='L')
        
        photo = self.ImageTk.PhotoImage(img)
        label.img = photo  # Keep reference
        label.config(image=photo)
    
    def update_text(self, label, text):
        label.config(text=text)
    
    def start_timer(self, callback, interval):
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(interval, callback)
    
    def stop_timer(self):
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
    
    def show_file_dialog(self, filetypes=None):
        if filetypes is None:
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        return self.filedialog.askopenfilename(filetypes=filetypes)
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        kwargs = {}
        if min_val is not None: kwargs['minvalue'] = min_val
        if max_val is not None: kwargs['maxvalue'] = max_val
        return self.simpledialog.askinteger(title, prompt, **kwargs)
    
    def run(self):
        if self.root:
            self.root.mainloop()
    
    def quit(self):
        if self.root:
            self.root.destroy()

class PyQt5Backend(AbstractGUIBackend):
    """PyQt5 implementation of GUI backend"""
    
    def __init__(self):
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QLabel, QPushButton, QRadioButton, 
            QHBoxLayout, QVBoxLayout, QFileDialog, QInputDialog, QButtonGroup
        )
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QImage, QPixmap
        
        self.QApplication = QApplication
        self.QWidget = QWidget
        self.QLabel = QLabel
        self.QPushButton = QPushButton
        self.QRadioButton = QRadioButton
        self.QHBoxLayout = QHBoxLayout
        self.QVBoxLayout = QVBoxLayout
        self.QFileDialog = QFileDialog
        self.QInputDialog = QInputDialog
        self.QButtonGroup = QButtonGroup
        self.Qt = Qt
        self.QTimer = QTimer
        self.QImage = QImage
        self.QPixmap = QPixmap
        
        # Initialize QApplication if not exists
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
            
        self.window = None
        self.timer = None
        self.radio_groups = {}
    
    def create_window(self, title, width, height):
        self.window = self.QWidget()
        self.window.setWindowTitle(title)
        self.window.setFixedSize(width, height)
        return self.window
    
    def create_button(self, parent, text, callback, x=None, y=None):
        btn = self.QPushButton(text, parent)
        btn.clicked.connect(callback)
        
        # Set reasonable minimum size for buttons
        btn.setMinimumSize(60, 30)
        
        if x is not None and y is not None:
            btn.move(x, y)
            # Auto-size based on text content with some padding
            btn.adjustSize()
            # Ensure minimum width to prevent overlapping
            if btn.width() < 60:
                btn.setFixedWidth(60)
        return btn
    
    def create_label(self, parent, text="", x=None, y=None, width=None, height=None):
        label = self.QLabel(text, parent)
        if x is not None and y is not None:
            label.move(x, y)
        if width and height:
            label.setFixedSize(width, height)
        return label
    
    def create_radiobutton(self, parent, text, group, value, callback=None):
        if group not in self.radio_groups:
            self.radio_groups[group] = self.QButtonGroup(parent)
        
        radio = self.QRadioButton(text, parent)
        self.radio_groups[group].addButton(radio)
        
        # Set reasonable size for radio buttons
        radio.adjustSize()
        
        if callback:
            radio.toggled.connect(lambda checked: callback(value) if checked else None)
        
        return radio
    
    def create_frame(self, parent, x=None, y=None, width=None, height=None):
        frame = self.QWidget(parent)
        if x is not None and y is not None:
            frame.move(x, y)
        if width and height:
            frame.setFixedSize(width, height)
        return frame
    
    def update_image(self, label, image_array):
        """Update label with numpy array image (RGB format expected)"""
        h, w = image_array.shape[:2]
        if len(image_array.shape) == 3:
            bytes_per_line = 3 * w
            qimg = self.QImage(image_array.data, w, h, bytes_per_line, self.QImage.Format_RGB888)
        else:
            bytes_per_line = w
            qimg = self.QImage(image_array.data, w, h, bytes_per_line, self.QImage.Format_Grayscale8)
        
        pixmap = self.QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
    
    def update_text(self, label, text):
        label.setText(text)
    
    def start_timer(self, callback, interval):
        if self.timer is None:
            self.timer = self.QTimer()
            self.timer.timeout.connect(callback)
        self.timer.start(interval)
    
    def stop_timer(self):
        if self.timer:
            self.timer.stop()
    
    def show_file_dialog(self, filetypes=None):
        if filetypes is None:
            filter_str = "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        else:
            # Convert tkinter format to PyQt format
            filters = []
            for name, pattern in filetypes:
                filters.append(f"{name} ({pattern})")
            filter_str = ";;".join(filters)
        
        filename, _ = self.QFileDialog.getOpenFileName(
            self.window, "Open File", "", filter_str
        )
        return filename
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        value, ok = self.QInputDialog.getInt(
            self.window, title, prompt,
            min=min_val if min_val is not None else -2147483647,
            max=max_val if max_val is not None else 2147483647
        )
        return value if ok else None
    
    def run(self):
        if self.window:
            self.window.show()
        sys.exit(self.app.exec_())
    
    def quit(self):
        if self.window:
            self.window.close()
        self.app.quit()

class GUIFramework:
    """Main GUI Framework class with backend abstraction"""
    
    def __init__(self, backend='tkinter'):
        """
        Initialize GUI Framework with specified backend
        
        Args:
            backend (str): Either 'tkinter' or 'pyqt5'
        """
        self.backend_type = GUIBackend(backend.lower())
        
        if self.backend_type == GUIBackend.TKINTER:
            self.backend = TkinterBackend()
        elif self.backend_type == GUIBackend.PYQT5:
            self.backend = PyQt5Backend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.window = None
        self.components = {}
    
    def create_window(self, title="GUI Application", width=800, height=600):
        """Create main application window"""
        self.window = self.backend.create_window(title, width, height)
        return self.window
    
    def create_button(self, parent, name, text, callback, x=None, y=None):
        """Create a button widget"""
        button = self.backend.create_button(parent, text, callback, x, y)
        self.components[name] = button
        return button
    
    def create_label(self, parent, name, text="", x=None, y=None, width=None, height=None):
        """Create a label widget"""
        label = self.backend.create_label(parent, text, x, y, width, height)
        self.components[name] = label
        return label
    
    def create_radiobutton(self, parent, name, text, group, value, callback=None):
        """Create a radio button widget"""
        radio = self.backend.create_radiobutton(parent, text, group, value, callback)
        self.components[name] = radio
        return radio
    
    def create_frame(self, parent, name, x=None, y=None, width=None, height=None):
        """Create a frame container"""
        frame = self.backend.create_frame(parent, x, y, width, height)
        self.components[name] = frame
        return frame
    
    def update_image(self, label_name, image_array):
        """Update image in a label widget"""
        if label_name in self.components:
            self.backend.update_image(self.components[label_name], image_array)
    
    def update_text(self, label_name, text):
        """Update text in a label widget"""
        if label_name in self.components:
            self.backend.update_text(self.components[label_name], text)
    
    def start_timer(self, callback, interval):
        """Start a timer with specified interval (ms)"""
        self.backend.start_timer(callback, interval)
    
    def stop_timer(self):
        """Stop the current timer"""
        self.backend.stop_timer()
    
    def show_file_dialog(self, filetypes=None):
        """Show file selection dialog"""
        return self.backend.show_file_dialog(filetypes)
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        """Show input dialog for integer input"""
        return self.backend.show_input_dialog(title, prompt, min_val, max_val)
    
    def get_component(self, name):
        """Get component by name"""
        return self.components.get(name)
    
    def run(self):
        """Start the GUI event loop"""
        self.backend.run()
    
    def quit(self):
        """Quit the application"""
        self.backend.quit()
    
    @staticmethod
    def cv2_to_rgb(cv2_image):
        """Convert OpenCV BGR image to RGB format"""
        if len(cv2_image.shape) == 3:
            return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return cv2_image
    
    @staticmethod
    def resize_image(image, width, height):
        """Resize image to specified dimensions"""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

# Example usage and demonstration
if __name__ == "__main__":
    # You can switch between backends here
    BACKEND = 'pyqt5'  # Change to 'tkinter' to use Tkinter
    
    gui = GUIFramework(backend=BACKEND)
    window = gui.create_window(f"Demo App ({BACKEND.upper()})", 400, 300)
    
    # Create some widgets
    frame = gui.create_frame(window, "main_frame", 10, 10, 380, 280)
    
    def button_clicked():
        print(f"Button clicked! Backend: {BACKEND}")
        gui.update_text("status_label", f"Button clicked using {BACKEND}!")
    
    def radio_changed(value):
        print(f"Radio changed to: {value}")
        gui.update_text("status_label", f"Selected: {value}")
    
    gui.create_button(frame, "test_btn", "Click Me", button_clicked, 10, 10)
    gui.create_label(frame, "title_label", f"GUI Framework Demo - {BACKEND.upper()}", 10, 50)
    gui.create_radiobutton(frame, "radio1", "Option 1", "group1", "opt1", radio_changed)
    gui.create_radiobutton(frame, "radio2", "Option 2", "group1", "opt2", radio_changed)
    gui.create_label(frame, "status_label", "Ready...", 10, 150)
    
    gui.run()