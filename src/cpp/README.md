# Optical Flow Viewer - C++ Qt Application

A minimal C++ Qt application for video playback with optical flow visualization, supporting both dense (Farneback) and sparse (Lucas-Kanade) optical flow computation.

## Features

- **Video Playback**: Load and play video files (MP4, AVI, MOV, MKV)
- **Optical Flow Visualization**: 
  - Dense flow using Farneback method (HSV color-coded)
  - Sparse flow using Lucas-Kanade method (tracked feature points)
- **Playback Controls**:
  - Play/Pause
  - Next/Previous frame
  - Speed control (faster/slower)
  - Jump to specific frame
- **Real-time switching** between dense and sparse flow modes

## Requirements

### System Dependencies

- **Qt5**: Core and Widgets modules
- **OpenCV**: 4.x (with video support)
- **CMake**: 3.16 or higher
- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    qtbase5-dev \
    libopencv-dev \
    cmake \
    build-essential
```

#### Arch Linux
```bash
sudo pacman -S qt5-base opencv cmake gcc
```

#### macOS (Homebrew)
```bash
brew install qt@5 opencv cmake
```

## Build Instructions

1. **Navigate to the source directory**:
   ```bash
   cd /home/irfan/Desktop/Code/Motion-Analysis/src/cpp
   ```

2. **Create build directory**:
   ```bash
   mkdir build && cd build
   ```

3. **Configure with CMake**:
   ```bash
   cmake ..
   ```

4. **Build**:
   ```bash
   cmake --build . -j$(nproc)
   ```

5. **Run**:
   ```bash
   ./OpticalFlowViewer
   ```

## Usage

1. **Open Video**: Click "Open" button and select a video file
2. **Playback Controls**:
   - **Play/Pause**: Toggle video playback
   - **<< / >>**: Step backward/forward one frame
   - **Faster/Slower**: Adjust playback speed (0.25x - 8x)
   - **Jump To**: Enter frame number to jump directly
3. **Flow Mode**: Select "Dense" or "Sparse" radio button to switch optical flow method
4. **Displays**:
   - **Left**: Original video frame
   - **Right**: Optical flow visualization

## Implementation Details

### Optical Flow Modes

#### Dense Flow (Farneback)
- **Algorithm**: Gunnar Farneback's method
- **Visualization**: HSV color coding
  - Hue: Flow direction (0-360°)
  - Saturation: Flow magnitude
  - Value: Maximum brightness
- **Parameters**:
  - Pyramid scale: 0.5
  - Levels: 3
  - Window size: 15
  - Iterations: 3
  - Poly N: 5
  - Poly Sigma: 1.2

#### Sparse Flow (Lucas-Kanade)
- **Algorithm**: Lucas-Kanade pyramidal method
- **Visualization**: Green trajectories + red feature points
- **Parameters**:
  - Max corners: 200
  - Quality level: 0.01
  - Min distance: 7
  - Window size: 15x15
  - Max pyramid level: 2

### Architecture

```
OpticalFlowViewer/
├── main.cpp              # Application entry point
├── mainwindow.h/cpp      # Main GUI window (video player)
├── opticalflow.h/cpp     # Optical flow computation
└── CMakeLists.txt        # Build configuration
```

**Key Classes**:
- `MainWindow`: Qt main window handling UI, video playback, and user interactions
- `OpticalFlow`: Optical flow computation engine (dense/sparse methods)

## Performance Notes

- Dense flow is computationally intensive; expect lower FPS on high-resolution videos
- Sparse flow is faster and suitable for real-time tracking
- Playback speed is adjusted dynamically based on timer intervals

## Troubleshooting

**Qt not found**:
```bash
# Set Qt installation path
export Qt5_DIR=/path/to/qt5/lib/cmake/Qt5
cmake ..
```

**OpenCV not found**:
```bash
# Set OpenCV installation path
export OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake ..
```

**Build errors**:
- Ensure C++17 support: `g++ --version` (GCC 7+)
- Check CMake version: `cmake --version` (3.16+)

## License

This implementation is based on the Python GUI framework player from the Motion-Analysis project.

## References

- [Farneback Optical Flow](https://link.springer.com/chapter/10.1007/3-540-45103-X_50)
- [Lucas-Kanade Method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)
- [Qt Documentation](https://doc.qt.io/)
- [OpenCV Optical Flow Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
