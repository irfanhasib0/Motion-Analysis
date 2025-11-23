FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR .
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    python3-pyqt5 \
    python3-tk \
    libgl1-mesa-glx \
    libx11-xcb1 \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxkbcommon-x11-0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Mesa OpenGL implementation — provides GL libs required by GUI rendering (PyQt5 / OpenCV imshow)
# X11/XCB bridge — required for X11 client/server communication used by Qt apps
# Xinerama extension support — enables multi-monitor/monitor geometry features some Qt components expect
# Development headers for Xinerama — only needed if building native XCB/Xinerama extensions; usually not required at runtime
# Keyboard handling for X11 — required for correct key event handling in Qt applications
# Utilities (xclock, xeyes, etc.) — useful to test X11 forwarding/display; optional in production

RUN apt-get update && apt-get install -y --no-install-recommends \
    xfce4 \
    xfce4-goodies \
    tightvncserver \
    dbus-x11 \
    xfonts-base \
    && rm -rf /var/lib/apt/lists/*

# Lightweight desktop environment (Xfce) — provides a full desktop session for VNC or when running multiple GUI apps; optional if you only run single X11 apps.
# Extra Xfce plugins/utilities — improves UX (panel plugins, file manager extras); convenient but not strictly required.
# VNC server — enables remote graphical access into the container (useful when X11 forwarding is not possible). Only needed if you plan to use VNC.
# D-Bus X11 bridge — some desktop components and Qt apps expect a session D-Bus; helps with proper session integration. Often required for full desktop behavior.
# Basic X11 fonts — provides common fonts so GUI apps render text correctly; small and generally necessary for text display.

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Build tools and libraries — essential for compiling native extensions and libraries often required by ML frameworks and computer vision packages.
# Image format libraries — support for JPEG, PNG, TIFF image handling in OpenCV and other libraries.
# Video codec libraries — enable video reading/writing capabilities in OpenCV and multimedia applications.

# Install CUDA/cuDNN and CUPTI required by TensorFlow GPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev libcupti-dev \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH CUDA_HOME=/usr/local/cuda

# Install Python packages
RUN python3 -m pip install --upgrade pip setuptools==59.6.0 wheel==0.37.1
COPY ./util_pkgs.txt ./util_pkgs.txt
COPY ./core_pkgs.txt ./core_pkgs.txt
COPY ./ext_pckgs.txt ./ext_pckgs.txt
RUN pip install -r util_pkgs.txt
RUN pip install -r core_pkgs.txt
RUN pip install -r ext_pckgs.txt
RUN pip install jupyterlab

# Ensure CUDA libraries are on the loader path for the pip-installed wheel
RUN pip install --upgrade pip
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchmetrics==1.6.1
RUN pip install tensorflow==2.20.0

EXPOSE 8010
# Optional: install via pip for latest versions
RUN pip3 install --no-cache-dir PyQt5
# Set up working directory
WORKDIR /app

# Copy your PyQt5 app code
COPY . /app

# Set display variable for X11 forwarding
ENV DISPLAY=:0

# Default command to run your PyQt5 app (change as needed)
CMD ["python3", "keypoint_tracking/gui_pyqt.py"]
