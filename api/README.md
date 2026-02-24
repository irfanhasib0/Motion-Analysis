# NVR Server

A simple Network Video Recorder (NVR) server built with FastAPI and React that supports RTSP streams, system cameras, video recording/playback, and custom video processing.

## Features

### Backend (FastAPI)
- **Camera Management**: Add/edit/delete cameras (RTSP, Webcam, IP Camera)
- **Video Recording**: Record from multiple cameras simultaneously
- **Live Streaming**: Real-time video streaming via HTTP
- **Video Playback**: Stream recorded videos
- **Custom Processing**: Apply video processing (motion detection, face detection, edge detection, color filters)
- **WebSocket Support**: Real-time updates for camera status and recordings
- **RESTful API**: Full REST API for camera and recording management

### Frontend (React)
- **Dashboard**: System overview with statistics and camera status
- **Camera Management**: Add, configure, and manage cameras
- **Live View**: Monitor cameras in real-time with grid and single view modes
- **Recording Management**: Browse, search, play, and download recordings
- **System Settings**: View system information and configuration
- **Responsive Design**: Mobile-friendly interface

### Video Processing
- **Motion Detection**: Detect and highlight motion in video streams
- **Face Detection**: Identify and track faces
- **Edge Detection**: Apply Canny edge detection
- **Color Filters**: Apply various color effects (grayscale, sepia, etc.)

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+ (optional, for frontend)
- OpenCV dependencies

### **Option 1: Complete Setup (Recommended)**
```bash
cd api
./start_server.sh
```

### **Option 2: API Only (Fast Start)**
```bash
cd api
pip install -r requirements.txt
python3 run_api.py
```
Access API documentation at http://localhost:8000/docs

### **Option 3: Manual Setup**
```bash
cd api
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup
```bash
cd api/frontend
npm install
npm start
```

The frontend will be available at `http://localhost:3000`

### Production Build
```bash
cd api/frontend
npm run build
cd ..
python main.py
```

Access the full application at `http://localhost:8000`

## Camera Configuration

### RTSP Cameras
- **Source**: `rtsp://username:password@ip:port/path`
- **Example**: `rtsp://admin:password@192.168.1.100:554/stream1`

### Webcams
- **Source**: Device index (usually `0` for default camera)
- **Example**: `0`

### IP Cameras
- **Source**: HTTP/HTTPS URL to video stream
- **Example**: `http://192.168.1.100:8080/video`

## API Endpoints

### Cameras
- `GET /api/cameras` - List all cameras
- `POST /api/cameras` - Add new camera
- `PUT /api/cameras/{id}` - Update camera
- `DELETE /api/cameras/{id}` - Delete camera

### Recording
- `POST /api/cameras/{id}/start-recording` - Start recording
- `POST /api/cameras/{id}/stop-recording` - Stop recording
- `GET /api/recordings` - List recordings
- `DELETE /api/recordings/{id}` - Delete recording

### Streaming
- `GET /api/cameras/{id}/stream` - Live camera stream
- `GET /api/recordings/{id}/stream` - Recorded video stream
- `GET /api/recordings/{id}/download` - Download recording

### Processing
- `GET /api/processing/types` - Available processors
- `POST /api/cameras/{id}/processing/{type}/start` - Start processing
- `POST /api/cameras/{id}/processing/stop` - Stop processing

### System
- `GET /api/system/info` - System information
- `WebSocket /ws/{client_id}` - Real-time updates

## Configuration

Copy `.env.example` to `.env` and adjust settings:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Recording
RECORDINGS_DIR=./recordings
DEFAULT_RESOLUTION=1920x1080
DEFAULT_FPS=30
```

### Password Login (Backend)

Set in `.env`:

```bash
AUTH_ENABLED=true
API_PASSWORD=change-this-password
AUTH_TOKEN_TTL_SECONDS=86400
```

Login and get bearer token:

```bash
curl -X POST http://localhost:8000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"password":"change-this-password"}'
```

Use token for API calls:

```bash
curl http://localhost:8000/api/cameras \
    -H "Authorization: Bearer <access_token>"
```

Quick alternative (no token exchange):

```bash
curl http://localhost:8000/api/cameras \
    -H "x-api-password: change-this-password"
```

## Directory Structure

```
api/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
├── models/                # Pydantic data models
│   ├── camera.py
│   └── recording.py
├── services/              # Business logic services
│   ├── recording_service.py
│   ├── streaming_service.py
│   └── processing_service.py
├── recordings/            # Recorded video files (created automatically)
└── frontend/              # React application
    ├── package.json
    ├── public/
    └── src/
        ├── components/    # React components
        ├── services/      # API client
        └── App.js         # Main application
```

## Usage Examples

### Adding an RTSP Camera
1. Navigate to "Cameras" page
2. Click "Add Camera"
3. Fill in details:
   - Name: "Front Door Camera"
   - Type: "RTSP Stream"
   - Source: "rtsp://admin:password@192.168.1.100:554/stream1"
   - Resolution: "1920x1080"
4. Click "Add Camera"

### Starting Recording
1. Go to "Cameras" page
2. Find your camera
3. Click "Record" button
4. Recording will start and appear in "Recordings" page

### Viewing Live Stream
1. Navigate to "Live View" page
2. See all online cameras in grid view
3. Click a camera for single view
4. Apply video processing if needed

### Managing Recordings
1. Go to "Recordings" page
2. Search and filter recordings
3. Click "Play" to view
4. Download or delete as needed

## Troubleshooting

### Camera Connection Issues
- Verify RTSP URL is correct
- Check network connectivity
- Ensure camera supports the resolution/FPS settings
- Try different RTSP paths (check camera documentation)

### Recording Problems
- Check disk space in recordings directory
- Verify camera stream is stable
- Check Python and OpenCV installation

### Performance Issues
- Reduce camera resolution/FPS
- Limit number of simultaneous recordings
- Disable video processing if not needed
- Use SSD for recordings directory

## Development

### Backend Development
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd api/frontend
npm install
npm start
```

### Adding Custom Video Processors
1. Create new processor class in `services/processing_service.py`
2. Inherit from `VideoProcessor` base class
3. Implement `process_frame()` method
4. Register in `ProcessingService.__init__()`

## License

This project is open source. See individual component licenses for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

For issues and feature requests, please create a GitHub issue.