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
    TKINTER = "tk"
    PYQT5 = "qt"  # 'qt' selects a Qt for Python backend (PySide6 preferred)
    WXPYTHON = "wx"

class AbstractGUIBackend(ABC):
    """Abstract base class for GUI backends"""
    
    @abstractmethod
    def create_window(self, title, width, height):
        pass
    
    @abstractmethod
    def create_button(self, parent, text, callback, x=None, y=None, width=None, height=None):
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
    def create_entry(self, parent, text="", x=None, y=None, width=None):
        pass
    
    @abstractmethod
    def create_dropdown(self, parent, options, callback=None, x=None, y=None, width=None, height=None):
        """Create a dropdown (combo box) with given options and selection callback"""
        pass
    
    @abstractmethod
    def update_dropdown_options(self, dropdown, options):
        """Update dropdown options after creation"""
        pass
    
    @abstractmethod
    def get_dropdown_value(self, dropdown):
        """Get currently selected value from dropdown"""
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
    def show_folder_dialog(self, title="Select Folder", initial_dir=None):
        pass
    
    @abstractmethod
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        pass
    
    @abstractmethod
    def show_message_box(self, title, message, buttons=None):
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
        from tkinter import filedialog, simpledialog, ttk
        from PIL import Image, ImageTk
        
        self.tk = tk
        self.filedialog = filedialog
        self.simpledialog = simpledialog
        self.ttk = ttk
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
    
    def create_button(self, parent, text, callback, x=None, y=None, width=None, height=None):
        btn = self.tk.Button(parent, text=text, command=callback)
        if width:
            btn.config(width=width)
        if height:
            btn.config(height=height)
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

    def create_entry(self, parent, text="", x=None, y=None, width=None, height=None):
        entry = self.tk.Entry(parent)
        if width:
            entry.config(width=width)
        if text:
            entry.insert(0, text)
        if x is not None and y is not None:
            entry.place(x=x, y=y)
        else:
            entry.pack()
        return entry
    
    def create_dropdown(self, parent, options, callback=None, x=None, y=None, width=None, height=None):
        """Create a dropdown using ttk.Combobox (readonly)"""
        combo = self.ttk.Combobox(parent, values=list(options) if options is not None else [], state="readonly")
        if width:
            combo.config(width=width)
        if options:
            combo.current(0)
        def on_select(event=None):
            if callback:
                callback(combo.get())
        combo.bind("<<ComboboxSelected>>", on_select)
        if x is not None and y is not None:
            combo.place(x=x, y=y)
        else:
            combo.pack()
        return combo
    
    def update_dropdown_options(self, dropdown, options):
        """Update dropdown options using ttk.Combobox"""
        current_selection = dropdown.get()
        dropdown.config(values=list(options) if options is not None else [])
        # Try to preserve selection if it still exists in new options
        if current_selection in options:
            dropdown.set(current_selection)
        elif options:
            dropdown.current(0)
    
    def get_dropdown_value(self, dropdown):
        """Get currently selected value from ttk.Combobox"""
        return dropdown.get()
    
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
    
    def get_text(self, entry):
        return entry.get()
    
    def start_timer(self, callback, interval):
        # Tkinter's after is one-shot; wrap to make it repeating
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        def _tick():
            try:
                callback()
            finally:
                # Reschedule for continuous updates
                self.timer_id = self.root.after(interval, _tick)
        self.timer_id = self.root.after(interval, _tick)
    
    def stop_timer(self):
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
    
    def show_file_dialog(self, filetypes=None):
        if filetypes is None:
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        return self.filedialog.askopenfilename(filetypes=filetypes)
    
    def show_folder_dialog(self, title="Select Folder", initial_dir=None):
        """Show a folder selection dialog"""
        kwargs = {"title": title}
        if initial_dir is not None:
            kwargs["initialdir"] = initial_dir
        return self.filedialog.askdirectory(**kwargs)
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        kwargs = {}
        if min_val is not None: kwargs['minvalue'] = min_val
        if max_val is not None: kwargs['maxvalue'] = max_val
        return self.simpledialog.askinteger(title, prompt, **kwargs)
    
    def show_message_box(self, title, message, buttons=None):
        """Show a message box with custom buttons"""
        import tkinter.messagebox as msgbox
        
        if buttons is None:
            buttons = ["OK"]
        
        if len(buttons) == 1:
            msgbox.showinfo(title, message)
            return buttons[0]
        elif len(buttons) == 2:
            result = msgbox.askyesno(title, message)
            return buttons[0] if result else buttons[1]
        elif len(buttons) == 3:
            # For 3 buttons, use askyesnocancel
            result = msgbox.askyesnocancel(title, message)
            if result is True:
                return buttons[0]
            elif result is False:
                return buttons[1]
            else:
                return buttons[2]
        else:
            # For more complex cases, create custom dialog
            dialog = self.tk.Toplevel()
            dialog.title(title)
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            result = [None]
            
            # Message label
            label = self.tk.Label(dialog, text=message, wraplength=250)
            label.pack(pady=20)
            
            # Button frame
            btn_frame = self.tk.Frame(dialog)
            btn_frame.pack(pady=10)
            
            def on_button_click(button_text):
                result[0] = button_text
                dialog.destroy()
            
            for button_text in buttons:
                btn = self.tk.Button(btn_frame, text=button_text, 
                                   command=lambda bt=button_text: on_button_click(bt))
                btn.pack(side=self.tk.LEFT, padx=5)
            
            dialog.wait_window()
            return result[0]
    
    def run(self):
        if self.root:
            self.root.mainloop()
    
    def quit(self):
        if self.root:
            self.root.destroy()

class PySide6Backend(AbstractGUIBackend):
    """PySide6 implementation of GUI backend (Qt for Python)"""
    def __init__(self):
        from PySide6.QtWidgets import (
            QApplication, QWidget, QLabel, QPushButton, QRadioButton,
            QHBoxLayout, QVBoxLayout, QFileDialog, QInputDialog, QButtonGroup, QComboBox
        )
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QImage, QPixmap

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
        self.QComboBox = QComboBox
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
        self.timer  = None
        self.radio_groups = {}

    def create_window(self, title, width, height):
        self.window = self.QWidget()
        self.window.setWindowTitle(title)
        self.window.resize(width, height)
        return self.window

    def create_button(self, parent, text, callback, x=None, y=None, width=None, height=None):
        btn = self.QPushButton(text, parent)
        btn.clicked.connect(callback)
        if width and height:
            btn.setFixedSize(width, height)
        elif width:
            btn.setFixedWidth(width)
        elif height:
            btn.setFixedHeight(height)
        if x is not None and y is not None:
            btn.move(x, y)
        return btn

    def create_label(self, parent, text="", x=None, y=None, width=None, height=None):
        label = self.QLabel(text, parent)
        if x is not None and y is not None:
            label.move(x, y)
        if width and height:
            label.setFixedSize(width, height)
        return label

    def create_entry(self, parent, text="", x=None, y=None, width=None, height=None):
        from PySide6.QtWidgets import QLineEdit
        entry = QLineEdit(parent)
        if text:
            entry.setText(text)
        if width and height:
            entry.setFixedSize(width, height)
        entry.setMinimumHeight(24)
        if x is not None and y is not None:
            entry.move(x, y)
        return entry

    def create_dropdown(self, parent, options, callback=None, x=None, y=None, width=None, height=None):
        """Create a dropdown using QComboBox"""
        combo = self.QComboBox(parent)
        if options:
            combo.addItems(list(options))
        if width and height:
            combo.setFixedSize(width, height)
        if x is not None and y is not None:
            combo.move(x, y)
        if callback:
            combo.currentTextChanged.connect(lambda text: callback(text))
        return combo

    def update_dropdown_options(self, dropdown, options):
        """Update dropdown options using QComboBox"""
        current_text = dropdown.currentText()
        dropdown.clear()
        if options:
            dropdown.addItems(list(options))
            # Try to preserve selection if it still exists in new options
            index = dropdown.findText(current_text)
            if index >= 0:
                dropdown.setCurrentIndex(index)
            else:
                dropdown.setCurrentIndex(0)

    def get_dropdown_value(self, dropdown):
        """Get currently selected value from QComboBox"""
        return dropdown.currentText()

    def create_radiobutton(self, parent, text, group, value, callback=None):
        if group not in self.radio_groups:
            self.radio_groups[group] = self.QButtonGroup(parent)
        radio = self.QRadioButton(text, parent)
        self.radio_groups[group].addButton(radio)
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

    def get_text(self, entry):
        return entry.text()

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
            filters = []
            for name, pattern in filetypes:
                filters.append(f"{name} ({pattern})")
            filter_str = ";;".join(filters)
        filename, _ = self.QFileDialog.getOpenFileName(
            self.window, "Open File", "", filter_str
        )
        return filename
    
    def show_folder_dialog(self, title="Select Folder", initial_dir=None):
        """Show a folder selection dialog"""
        start_dir = initial_dir if initial_dir is not None else ""
        folder = self.QFileDialog.getExistingDirectory(None, title, start_dir)
        return folder if folder else None
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        value, ok = self.QInputDialog.getText(
            self.window, title, prompt
        )
        return value if ok else None

    def show_message_box(self, title, message, buttons=None):
        """Show a message box with custom buttons"""
        from PySide6.QtWidgets import QMessageBox
        
        if buttons is None:
            buttons = ["OK"]
        
        msg_box = QMessageBox(self.window)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        
        # Add custom buttons
        button_objects = []
        for button_text in buttons:
            btn = msg_box.addButton(button_text, QMessageBox.AcceptRole)
            button_objects.append((btn, button_text))
        
        msg_box.exec_()
        clicked_button = msg_box.clickedButton()
        
        # Find which button was clicked
        for btn_obj, btn_text in button_objects:
            if btn_obj == clicked_button:
                return btn_text
        
        return buttons[0] if buttons else None

    def run(self):
        if self.window:
            self.window.show()
        sys.exit(self.app.exec())

    def quit(self):
        if self.window:
            self.window.close()
        self.app.quit()

class WxPythonBackend(AbstractGUIBackend):
    """wxPython implementation of GUI backend"""
    
    def __init__(self):
        import wx
        
        self.wx = wx
        
        # Initialize wxApp if not exists
        if not wx.App.Get():
            self.app = wx.App(False)
        else:
            self.app = wx.App.Get()
            
        self.window = None
        self.timer = None
        self.radio_groups = {}
    
    def create_window(self, title, width, height):
        self.window = self.wx.Frame(None, title=title, size=(width, height))
        # Create main panel
        self.main_panel = self.wx.Panel(self.window)
        return self.main_panel
    
    def create_button(self, parent, text, callback, x=None, y=None, width=None, height=None):
        if width and height:
            btn = self.wx.Button(parent, label=text, size=(width, height))
        elif width:
            btn = self.wx.Button(parent, label=text, size=(width, -1))
        elif height:
            btn = self.wx.Button(parent, label=text, size=(-1, height))
        else:
            btn = self.wx.Button(parent, label=text)
        btn.Bind(self.wx.EVT_BUTTON, lambda evt: callback())
        
        if x is not None and y is not None:
            btn.SetPosition((x, y))
        
        return btn
    
    def create_label(self, parent, text="", x=None, y=None, width=None, height=None):
        # Create StaticBitmap for image display (when width and height specified and no text)
        if width and height and not text:
            # Create a blank bitmap for image display
            bitmap = self.wx.Bitmap(width, height)
            label = self.wx.StaticBitmap(parent, bitmap=bitmap, size=(width, height))
        elif width and height:
            label = self.wx.StaticText(parent, label=text, size=(width, height))
        else:
            label = self.wx.StaticText(parent, label=text)
        
        if x is not None and y is not None:
            label.SetPosition((x, y))
        
        return label
    
    def create_entry(self, parent, text="", x=None, y=None, width=None, height=None):
        """Create a text entry widget for user input"""
        if width and height:
            entry = self.wx.TextCtrl(parent, value=text, size=(width, height))
        elif width:
            entry = self.wx.TextCtrl(parent, value=text, size=(width, -1))
        else:
            entry = self.wx.TextCtrl(parent, value=text)
        
        if x is not None and y is not None:
            entry.SetPosition((x, y))
        
        return entry
    
    def create_dropdown(self, parent, options, callback=None, x=None, y=None, width=None, height=None):
        """Create a dropdown using wx.Choice"""
        size = (width, height) if (width and height) else self.wx.DefaultSize
        choice = self.wx.Choice(parent, choices=list(options) if options is not None else [], size=size)
        if x is not None and y is not None:
            choice.SetPosition((x, y))
        if options:
            choice.SetSelection(0)
        if callback:
            def on_select(evt):
                idx = choice.GetSelection()
                if idx != self.wx.NOT_FOUND:
                    callback(choice.GetString(idx))
            choice.Bind(self.wx.EVT_CHOICE, on_select)
        return choice
    
    def update_dropdown_options(self, dropdown, options):
        """Update dropdown options using wx.Choice"""
        current_selection = dropdown.GetSelection()
        current_text = dropdown.GetString(current_selection) if current_selection != self.wx.NOT_FOUND else None
        
        dropdown.Clear()
        if options:
            dropdown.AppendItems(list(options))
            # Try to preserve selection if it still exists in new options
            if current_text and current_text in options:
                new_index = list(options).index(current_text)
                dropdown.SetSelection(new_index)
            else:
                dropdown.SetSelection(0)
    
    def get_dropdown_value(self, dropdown):
        """Get currently selected value from wx.Choice"""
        selection = dropdown.GetSelection()
        if selection != self.wx.NOT_FOUND:
            return dropdown.GetString(selection)
        return None
    
    def create_radiobutton(self, parent, text, group, value, callback=None):
        # Create radio button group if it doesn't exist
        if group not in self.radio_groups:
            self.radio_groups[group] = {
                'buttons': [],
                'group': self.wx.RadioButton(parent, label=text, style=self.wx.RB_GROUP),
                'callback': callback,
                'values': {}
            }
            radio = self.radio_groups[group]['group']
            self.radio_groups[group]['buttons'].append(radio)
            self.radio_groups[group]['values'][radio] = value
        else:
            radio = self.wx.RadioButton(parent, label=text)
            self.radio_groups[group]['buttons'].append(radio)
            self.radio_groups[group]['values'][radio] = value
        
        if callback:
            def on_select(event):
                if radio.GetValue():
                    callback(value)
            radio.Bind(self.wx.EVT_RADIOBUTTON, on_select)
        
        return radio
    
    def create_frame(self, parent, x=None, y=None, width=None, height=None):
        if width and height:
            frame = self.wx.Panel(parent, size=(width, height))
        else:
            frame = self.wx.Panel(parent)
        
        if x is not None and y is not None:
            frame.SetPosition((x, y))
        
        return frame
    
    def update_image(self, label, image_array):
        """Update label with image from numpy array (RGB format expected)"""
        if image_array is None:
            return
        
        # Ensure this runs on the main thread
        if not self.wx.IsMainThread():
            self.wx.CallAfter(self.update_image, label, image_array)
            return
        
        height, width = image_array.shape[:2]
        
        # Ensure RGB format
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Ensure contiguous memory
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        # Convert to wx.Image using buffer API (more reliable across platforms)
        try:
            wx_image = self.wx.ImageFromBuffer(width, height, image_array)
        except Exception:
            # Fallback for older wx versions
            wx_image = self.wx.Image(width, height)
            wx_image.SetData(image_array.tobytes())
        bitmap = self.wx.Bitmap(wx_image)
        
        # Update the label (assuming it's a StaticBitmap)
        if isinstance(label, self.wx.StaticBitmap):
            label.SetBitmap(bitmap)
            # Ensure repaint; some environments need explicit refresh
            label.Refresh(False)
            label.Update()
        elif isinstance(label, self.wx.StaticText):
            # If it's a StaticText, we can't display images directly
            pass
    
    def update_text(self, label, text):
        """Update label text"""
        # Ensure this runs on the main thread
        if not self.wx.IsMainThread():
            self.wx.CallAfter(self.update_text, label, text)
            return
        if isinstance(label, self.wx.StaticText) or isinstance(label, self.wx.Button):
            label.SetLabel(text)
        elif  isinstance(label, self.wx.TextCtrl):
            label.SetValue(text)

    def get_text(self, entry):
        """Get text from entry widget"""
        return entry.GetValue()
    
    def start_timer(self, callback, interval):
        """Start a timer that calls callback every interval milliseconds"""
        if self.timer:
            self.timer.Stop()
        
        # Timer must be owned by the window for proper event handling
        self.timer = self.wx.Timer(self.window)
        self.window.Bind(self.wx.EVT_TIMER, lambda evt: callback(), self.timer)
        self.timer.Start(interval)
    
    def stop_timer(self):
        """Stop the timer"""
        if self.timer:
            self.timer.Stop()
    
    def show_file_dialog(self, filetypes=None):
        """Show file open dialog"""
        if filetypes is None:
            wildcard = "All files (*.*)|*.*"
        else:
            # Convert to wx wildcard format
            wildcards = []
            for name, pattern in filetypes:
                # Convert *.ext to proper wildcard
                wildcards.append(f"{name} ({pattern})|{pattern}")
            wildcard = "|".join(wildcards)
        
        with self.wx.FileDialog(self.window, "Open File", wildcard=wildcard,
                               style=self.wx.FD_OPEN | self.wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == self.wx.ID_OK:
                return dlg.GetPath()
        return None
    
    def show_folder_dialog(self, title="Select Folder", initial_dir=None):
        """Show a folder selection dialog"""
        default_path = initial_dir if initial_dir is not None else ""
        with self.wx.DirDialog(self.window, title, defaultPath=default_path, style=self.wx.DD_DEFAULT_STYLE) as dialog:
            if dialog.ShowModal() == self.wx.ID_OK:
                return dialog.GetPath()
        return None
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        """Show input dialog for integer input"""
        dlg = self.wx.TextEntryDialog(self.window, prompt, title)
        
        if dlg.ShowModal() == self.wx.ID_OK:
            try:
                value = int(dlg.GetValue())
                if min_val is not None and value < min_val:
                    return None
                if max_val is not None and value > max_val:
                    return None
                return value
            except ValueError:
                return None
        
        dlg.Destroy()
        return None
    
    def show_message_box(self, title, message, buttons=None):
        """Show a message box with custom buttons"""
        if buttons is None:
            buttons = ["OK"]
        
        if len(buttons) == 1:
            dlg = self.wx.MessageDialog(self.window, message, title, self.wx.OK)
            dlg.ShowModal()
            dlg.Destroy()
            return buttons[0]
        elif len(buttons) == 2:
            dlg = self.wx.MessageDialog(self.window, message, title, 
                                      self.wx.YES_NO)
            result = dlg.ShowModal()
            dlg.Destroy()
            return buttons[0] if result == self.wx.ID_YES else buttons[1]
        elif len(buttons) == 3:
            dlg = self.wx.MessageDialog(self.window, message, title, 
                                      self.wx.YES_NO | self.wx.CANCEL)
            result = dlg.ShowModal()
            dlg.Destroy()
            if result == self.wx.ID_YES:
                return buttons[0]
            elif result == self.wx.ID_NO:
                return buttons[1]
            else:
                return buttons[2]
        else:
            # For more complex cases, create custom dialog
            dlg = self.wx.Dialog(self.window, title=title, size=(300, 150))
            
            # Create sizer for layout
            sizer = self.wx.BoxSizer(self.wx.VERTICAL)
            
            # Message text
            text = self.wx.StaticText(dlg, label=message)
            sizer.Add(text, 0, self.wx.ALL | self.wx.CENTER, 20)
            
            # Button sizer
            btn_sizer = self.wx.BoxSizer(self.wx.HORIZONTAL)
            
            result = [None]
            
            def on_button(event, button_text):
                result[0] = button_text
                dlg.EndModal(self.wx.ID_OK)
            
            for i, button_text in enumerate(buttons):
                btn = self.wx.Button(dlg, id=self.wx.ID_ANY, label=button_text)
                btn.Bind(self.wx.EVT_BUTTON, lambda evt, bt=button_text: on_button(evt, bt))
                btn_sizer.Add(btn, 0, self.wx.ALL, 5)
            
            sizer.Add(btn_sizer, 0, self.wx.CENTER)
            dlg.SetSizer(sizer)
            
            dlg.ShowModal()
            dlg.Destroy()
            return result[0]
    
    def run(self):
        """Start the GUI event loop"""
        if self.window:
            self.window.Show()
        self.app.MainLoop()
    
    def quit(self):
        """Quit the application"""
        if self.window:
            self.window.Close()
        self.app.ExitMainLoop()

class GUIFramework:
    """Main GUI Framework class with backend abstraction"""
    
    def __init__(self, backend='tk'):
        """
        Initialize GUI Framework with specified backend
        
        Args:
            backend (str): Either 'tk', 'qt', or 'wx' for Tkinter, PyQt5, or WxPython
        """
        self.backend_type = GUIBackend(backend.lower())
        
        if self.backend_type == GUIBackend.TKINTER:
            self.backend = TkinterBackend()
        elif self.backend_type == GUIBackend.PYQT5:
            self.backend = PySide6Backend()
        #elif self.backend_type == GUIBackend.PYQT5:
        #    self.backend = PyQt5Backend()
        elif self.backend_type == GUIBackend.WXPYTHON:
            self.backend = WxPythonBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.window = None
        self.components = {}
    
    def create_window(self, title="GUI Application", width=800, height=600):
        """Create main application window"""
        self.window = self.backend.create_window(title, width, height)
        return self.window
    
    def create_button(self, parent, name, text, callback, x=None, y=None, width=None, height=None):
        """Create a button widget"""
        button = self.backend.create_button(parent, text, callback, x, y, width, height)
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

    def create_entry(self, parent, name, text="", x=None, y=None, width=None, height=None):
        """Create a text entry widget"""
        entry = self.backend.create_entry(parent, text, x, y, width, height)
        self.components[name] = entry
        return entry
    
    def create_dropdown(self, parent, name, options, callback=None, x=None, y=None, width=None, height=None):
        """Create a dropdown (combo box) widget"""
        dropdown = self.backend.create_dropdown(parent, options, callback, x, y, width, height)
        self.components[name] = dropdown
        return dropdown
    
    def update_dropdown_options(self, dropdown_name, options):
        """Update dropdown options by name"""
        if dropdown_name in self.components:
            self.backend.update_dropdown_options(self.components[dropdown_name], options)
    
    def get_dropdown_value(self, dropdown_name):
        """Get currently selected value from dropdown by name"""
        if dropdown_name in self.components:
            return self.backend.get_dropdown_value(self.components[dropdown_name])
        return None
    
    def update_image(self, label_name, image_array):
        """Update image in a label widget"""
        if label_name in self.components:
            self.backend.update_image(self.components[label_name], image_array)
    
    def update_text(self, label_name, text):
        """Update text in a label widget"""
        if label_name in self.components:
            self.backend.update_text(self.components[label_name], text)

    def get_text(self, entry_name):
        """Get text from an entry widget"""
        if entry_name in self.components:
            return self.backend.get_text(self.components[entry_name])
        return None
    
    def start_timer(self, callback, interval):
        """Start a timer with specified interval (ms)"""
        self.backend.start_timer(callback, interval)
    
    def stop_timer(self):
        """Stop the current timer"""
        self.backend.stop_timer()
    
    def show_file_dialog(self, filetypes=None):
        """Show file selection dialog"""
        return self.backend.show_file_dialog(filetypes)
    
    def show_folder_dialog(self, title="Select Folder", initial_dir=None):
        """Show folder selection dialog"""
        return self.backend.show_folder_dialog(title, initial_dir)
    
    def show_input_dialog(self, title, prompt, min_val=None, max_val=None):
        """Show input dialog for integer input"""
        return self.backend.show_input_dialog(title, prompt, min_val, max_val)
    
    def show_message_box(self, title, message, buttons=None):
        """Show message box with custom buttons"""
        return self.backend.show_message_box(title, message, buttons)
    
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
    BACKEND = 'qt'  # Change to 'tkinter' to use Tkinter
    if BACKEND == 'qt':
        import os
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = BACKEND
    
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
    
    gui.create_button(frame, "test_btn", "Click Me", button_clicked, 10, 30)
    gui.create_label(frame, "title_label", f"GUI Framework Demo - {BACKEND.upper()}", 10, 60)
    gui.create_radiobutton(frame, "radio1", "Option 1", "group1", "opt1", radio_changed)
    gui.create_radiobutton(frame, "radio2", "Option 2", "group1", "opt2", radio_changed)
    gui.create_label(frame, "status_label", "Ready...", 10, 150)
    
    gui.run()