import React, { useState, useEffect } from 'react';
import { Monitor, Maximize, Settings, Play, Square } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { api } from '../services/api';

// Processing Modal Component
const ProcessingModal = ({ 
  show, 
  onClose, 
  processingTypes, 
  selectedProcessor, 
  setSelectedProcessor, 
  processingParams, 
  setProcessingParams, 
  onStartProcessing 
}) => {
  if (!show) return null;

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal-header">
          <h3 className="modal-title">Start Video Processing</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-body">
          <div className="form-group">
            <label className="form-label">Processing Type</label>
            <select 
              className="form-control form-select"
              value={selectedProcessor}
              onChange={(e) => setSelectedProcessor(e.target.value)}
            >
              <option value="">Select processor...</option>
              {processingTypes.map(processor => (
                <option key={processor.name} value={processor.name}>
                  {processor.name} - {processor.description}
                </option>
              ))}
            </select>
          </div>
          
          {selectedProcessor === 'color_filter' && (
            <div className="form-group">
              <label className="form-label">Filter Type</label>
              <select 
                className="form-control form-select"
                value={processingParams.filter_type || 'none'}
                onChange={(e) => setProcessingParams({...processingParams, filter_type: e.target.value})}
              >
                <option value="none">None</option>
                <option value="grayscale">Grayscale</option>
                <option value="sepia">Sepia</option>
                <option value="blue">Blue Enhanced</option>
                <option value="green">Green Enhanced</option>
                <option value="red">Red Enhanced</option>
              </select>
            </div>
          )}
        </div>
        
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button 
            className="btn btn-primary" 
            onClick={onStartProcessing}
            disabled={!selectedProcessor}
          >
            Start Processing
          </button>
        </div>
      </div>
    </div>
  );
};

// Grid View Component
const GridView = ({ 
  cameras, 
  streamStatus, 
  streamUrls, 
  startCameraStream, 
  stopCameraStream, 
  handleCameraSelection 
}) => {
  const onlineCameras = cameras.filter(c => c.status === 'online' || c.status === 'recording');

  if (onlineCameras.length === 0) {
    return (
      <div className="empty-state">
        <Monitor size={64} />
        <h3>No Active Cameras</h3>
        <p>Start some cameras from the cameras page to see them here</p>
      </div>
    );
  }

  return (
    <div className="camera-grid">
      {onlineCameras.map(camera => (
        <div key={camera.id} className="camera-card">
          <div className="camera-header">
            <h4 className="camera-name">{camera.name}</h4>
            <span className={`status-badge status-${camera.status}`}>
              {camera.status.toUpperCase()}
            </span>
          </div>
          
          <div className="camera-video" onClick={() => handleCameraSelection(camera)}>
            {streamStatus[camera.id] === 'active' ? (
              <img 
                src={streamUrls[camera.id] || api.getCameraStreamUrl(camera.id)}
                alt={`Camera ${camera.name}`}
                className="camera-stream"
                style={{ cursor: 'pointer' }}
                onLoad={() => {/* Handle load if needed */}}
                onError={() => {/* Handle error if needed */}}
              />
            ) : (
              <div className="camera-placeholder" style={{ cursor: 'pointer' }}>
                <Monitor size={48} />
                <p>Stream Stopped</p>
                <button 
                  className="btn btn-success"
                  onClick={(e) => {
                    e.stopPropagation();
                    startCameraStream(camera.id);
                  }}
                >
                  <Play size={16} />
                  Start Stream
                </button>
              </div>
            )}
          </div>
          
          <div className="camera-controls">
            {streamStatus[camera.id] === 'active' ? (
              <button 
                className="btn btn-danger"
                onClick={() => stopCameraStream(camera.id)}
              >
                <Square size={16} />
                Stop Stream
              </button>
            ) : (
              <button 
                className="btn btn-success"
                onClick={() => startCameraStream(camera.id)}
              >
                <Play size={16} />
                Start Stream
              </button>
            )}
          </div>
          
          {camera.processing_active && (
            <div className="processing-indicator">
              {camera.processing_type}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// Single View Component  
const SingleView = ({ 
  selectedCamera, 
  streamStatus, 
  streamUrls, 
  startCameraStream, 
  stopCameraStream, 
  handleStopProcessing, 
  setShowProcessingModal 
}) => {
  if (!selectedCamera) {
    return (
      <div className="empty-state">
        <Monitor size={64} />
        <h3>No Camera Selected</h3>
        <p>Go to grid view and click on a camera to view it here</p>
      </div>
    );
  }

  return (
    <div className="single-camera-view">
      <div className="camera-header">
        <h3 className="camera-name">{selectedCamera.name}</h3>
        <span className={`status-badge status-${selectedCamera.status}`}>
          {selectedCamera.status.toUpperCase()}
        </span>
      </div>
      
      <div className="camera-video" style={{ height: '70vh' }}>
        {streamStatus[selectedCamera.id] === 'active' ? (
          <img 
            src={streamUrls[selectedCamera.id] || api.getCameraStreamUrl(selectedCamera.id)}
            alt={`Camera ${selectedCamera.name}`}
            className="camera-stream"
            style={{ 
              width: '100%', 
              height: '100%', 
              objectFit: 'contain',
              backgroundColor: '#000'
            }}
            onLoad={() => {/* Handle load */}}
            onError={() => {/* Handle error */}}
          />
        ) : (
          <div className="camera-placeholder" style={{ 
            height: '100%', 
            display: 'flex', 
            flexDirection: 'column', 
            justifyContent: 'center', 
            alignItems: 'center',
            backgroundColor: '#000'
          }}>
            <Monitor size={64} />
            <p style={{ color: 'white', fontSize: '18px' }}>Stream Stopped</p>
            <button 
              className="btn btn-success"
              onClick={() => startCameraStream(selectedCamera.id)}
              style={{ marginTop: '16px' }}
            >
              <Play size={16} />
              Start Stream
            </button>
          </div>
        )}
        
        {selectedCamera.processing_active && (
          <div className="processing-indicator">
            {selectedCamera.processing_type}
          </div>
        )}
      </div>

      <div className="camera-controls">
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="btn btn-secondary" style={{ cursor: 'default' }}>
            Source: {selectedCamera.source}
          </span>
          <span className="btn btn-secondary" style={{ cursor: 'default' }}>
            FPS: {selectedCamera.fps}
          </span>
          
          {streamStatus[selectedCamera.id] === 'active' ? (
            <button 
              className="btn btn-danger"
              onClick={() => stopCameraStream(selectedCamera.id)}
            >
              <Square size={14} />
              Stop Stream
            </button>
          ) : (
            <button 
              className="btn btn-success"
              onClick={() => startCameraStream(selectedCamera.id)}
            >
              <Play size={14} />
              Start Stream
            </button>
          )}
          
          {selectedCamera.processing_active ? (
            <button 
              className="btn btn-warning"
              onClick={() => handleStopProcessing(selectedCamera.id)}
            >
              Stop {selectedCamera.processing_type}
            </button>
          ) : (
            <button 
              className="btn btn-primary"
              onClick={() => setShowProcessingModal(true)}
            >
              <Settings size={14} />
              Start Processing
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// Main LiveView Component
const LiveView = ({ cameras }) => {
  // State management
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [viewMode, setViewMode] = useState('grid');
  const [processingTypes, setProcessingTypes] = useState([]);
  const [showProcessingModal, setShowProcessingModal] = useState(false);
  const [selectedProcessor, setSelectedProcessor] = useState('');
  const [processingParams, setProcessingParams] = useState({});
  const [activeStreams, setActiveStreams] = useState(new Set());
  const [streamStatus, setStreamStatus] = useState({});
  const [streamUrls, setStreamUrls] = useState({});

  // Initialize component
  useEffect(() => {
    loadProcessingTypes();
    
    // Initialize stream status for all cameras
    if (cameras.length > 0) {
      const initialStatus = {};
      cameras.forEach(camera => {
        initialStatus[camera.id] = 'stopped';
      });
      setStreamStatus(prev => ({ ...prev, ...initialStatus }));
    }
    
    // Cleanup function to close all active streams when component unmounts
    return () => {
      activeStreams.forEach(async (cameraId) => {
        try {
          await api.closeCameraStream(cameraId);
          console.log(`Cleanup: Closed stream for camera: ${cameraId}`);
        } catch (error) {
          console.error(`Cleanup: Failed to close stream for camera ${cameraId}:`, error);
        }
      });
    };
  }, [cameras]);

  // Load processing types
  const loadProcessingTypes = async () => {
    try {
      const response = await api.getProcessingTypes();
      setProcessingTypes(response.data);
    } catch (error) {
      console.error('Failed to load processing types:', error);
    }
  };

  // Stream management functions
  const startCameraStream = async (cameraId) => {
    console.log('Starting camera stream for:', cameraId);
    try {
      // Actually call the backend to start the camera
      await api.startCamera(cameraId);
      
      setActiveStreams(prev => new Set([...prev, cameraId]));
      setStreamStatus(prev => ({ ...prev, [cameraId]: 'starting' }));
      
      // Generate cache-busted URL to force refresh
      const timestamp = Date.now();
      const streamUrl = `${api.getCameraStreamUrl(cameraId)}?t=${timestamp}`;
      setStreamUrls(prev => ({ ...prev, [cameraId]: streamUrl }));
      
      // Set status to active after a short delay
      setTimeout(() => {
        setStreamStatus(prev => ({ ...prev, [cameraId]: 'active' }));
      }, 1000); // Give camera more time to start
      
      toast.success('Camera stream started');
    } catch (error) {
      console.error(`Failed to start stream for camera ${cameraId}:`, error);
      setStreamStatus(prev => ({ ...prev, [cameraId]: 'error' }));
      toast.error('Failed to start camera stream: ' + (error.response?.data?.detail || error.message));
    }
  };

  const stopCameraStream = async (cameraId) => {
    console.log('Stopping camera stream for:', cameraId);
    try {
      await api.stopCamera(cameraId);
      await api.closeCameraStream(cameraId);
      
      setActiveStreams(prev => {
        const newSet = new Set(prev);
        newSet.delete(cameraId);
        return newSet;
      });
      setStreamStatus(prev => ({ ...prev, [cameraId]: 'stopped' }));
      setStreamUrls(prev => ({ ...prev, [cameraId]: null }));
      console.log(`Stopped stream for camera: ${cameraId}`);
      toast.success('Camera stream stopped');
    } catch (error) {
      console.error(`Failed to stop stream for camera ${cameraId}:`, error);
      toast.error('Failed to stop camera stream: ' + (error.response?.data?.detail || error.message));
    }
  };

  // View management
  const handleViewModeChange = (newMode) => {
    setViewMode(newMode);
  };

  const handleCameraSelection = (camera) => {
    setSelectedCamera(camera);
    setViewMode('single');
  };

  // Processing functions
  const handleStartProcessing = async () => {
    if (!selectedCamera || !selectedProcessor) return;
    
    try {
      await api.startProcessing(selectedCamera.id, selectedProcessor, processingParams);
      setShowProcessingModal(false);
      setSelectedProcessor('');
      setProcessingParams({});
      toast.success('Processing started');
    } catch (error) {
      toast.error('Failed to start processing: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleStopProcessing = async (cameraId) => {
    try {
      await api.stopProcessing(cameraId);
      toast.success('Processing stopped');
    } catch (error) {
      toast.error('Failed to stop processing: ' + (error.response?.data?.detail || error.message));
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Live View</h1>
        <p className="page-subtitle">Monitor your cameras in real-time</p>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">
            {viewMode === 'grid' ? 'All Cameras' : `Camera: ${selectedCamera?.name || 'None'}`}
          </h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              className={`btn ${viewMode === 'grid' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => handleViewModeChange('grid')}
            >
              <Monitor size={16} />
              Grid View
            </button>
            <button 
              className={`btn ${viewMode === 'single' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => handleViewModeChange('single')}
              disabled={!selectedCamera}
            >
              <Maximize size={16} />
              Single View
            </button>
          </div>
        </div>

        {viewMode === 'grid' ? (
          <GridView 
            cameras={cameras}
            streamStatus={streamStatus}
            streamUrls={streamUrls}
            startCameraStream={startCameraStream}
            stopCameraStream={stopCameraStream}
            handleCameraSelection={handleCameraSelection}
          />
        ) : (
          <SingleView 
            selectedCamera={selectedCamera}
            streamStatus={streamStatus}
            streamUrls={streamUrls}
            startCameraStream={startCameraStream}
            stopCameraStream={stopCameraStream}
            handleStopProcessing={handleStopProcessing}
            setShowProcessingModal={setShowProcessingModal}
          />
        )}

        {showProcessingModal && (
          <ProcessingModal
            show={showProcessingModal}
            onClose={() => setShowProcessingModal(false)}
            processingTypes={processingTypes}
            selectedProcessor={selectedProcessor}
            setSelectedProcessor={setSelectedProcessor}
            processingParams={processingParams}
            setProcessingParams={setProcessingParams}
            onStartProcessing={handleStartProcessing}
          />
        )}
      </div>
    </div>
  );
};

export default LiveView;