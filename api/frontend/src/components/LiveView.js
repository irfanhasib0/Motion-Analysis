import React, { useState, useEffect } from 'react';
import { Monitor, Maximize, Settings } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { api } from '../services/api';

const LiveView = ({ cameras }) => {
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'single'
  const [processingTypes, setProcessingTypes] = useState([]);
  const [showProcessingModal, setShowProcessingModal] = useState(false);
  const [selectedProcessor, setSelectedProcessor] = useState('');
  const [processingParams, setProcessingParams] = useState({});

  useEffect(() => {
    loadProcessingTypes();
  }, []);

  const loadProcessingTypes = async () => {
    try {
      const response = await api.getProcessingTypes();
      setProcessingTypes(response.data);
    } catch (error) {
      console.error('Failed to load processing types:', error);
    }
  };

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

  const onlineCameras = cameras.filter(c => c.status === 'online' || c.status === 'recording');

  const ProcessingModal = () => (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && setShowProcessingModal(false)}>
      <div className="modal">
        <div className="modal-header">
          <h3 className="modal-title">Start Video Processing</h3>
          <button className="modal-close" onClick={() => setShowProcessingModal(false)}>×</button>
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
          <button className="btn btn-secondary" onClick={() => setShowProcessingModal(false)}>
            Cancel
          </button>
          <button 
            className="btn btn-primary" 
            onClick={handleStartProcessing}
            disabled={!selectedProcessor}
          >
            Start Processing
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Live View</h1>
        <p className="page-subtitle">Monitor your cameras in real-time</p>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">
            {viewMode === 'grid' ? 'All Cameras' : `Camera: ${selectedCamera?.name}`}
          </h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              className={`btn ${viewMode === 'grid' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('grid')}
            >
              <Monitor size={16} />
              Grid View
            </button>
            <button 
              className={`btn ${viewMode === 'single' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('single')}
              disabled={!selectedCamera}
            >
              <Maximize size={16} />
              Single View
            </button>
          </div>
        </div>

        {viewMode === 'grid' ? (
          <div className="camera-grid">
            {onlineCameras.map(camera => (
              <div key={camera.id} className="camera-card">
                <div className="camera-header">
                  <div className="camera-title">{camera.name}</div>
                  <div className="camera-info">
                    <span className={`status status-${camera.status}`}>
                      {camera.status}
                    </span>
                    <span>{camera.resolution}</span>
                  </div>
                </div>
                
                <div className="camera-video" onClick={() => setSelectedCamera(camera)}>
                  <img 
                    src={api.getCameraStreamUrl(camera.id)}
                    alt={`Camera ${camera.name}`}
                    className="camera-stream"
                    style={{ cursor: 'pointer' }}
                  />
                  
                  {camera.processing_active && (
                    <div className="processing-indicator">
                      {camera.processing_type}
                    </div>
                  )}
                </div>

                <div className="camera-controls">
                  <button 
                    className="btn btn-primary"
                    onClick={() => {
                      setSelectedCamera(camera);
                      setViewMode('single');
                    }}
                  >
                    <Maximize size={14} />
                    Full View
                  </button>
                  
                  {camera.processing_active ? (
                    <button 
                      className="btn btn-warning"
                      onClick={() => handleStopProcessing(camera.id)}
                    >
                      Stop Processing
                    </button>
                  ) : (
                    <button 
                      className="btn btn-secondary"
                      onClick={() => {
                        setSelectedCamera(camera);
                        setShowProcessingModal(true);
                      }}
                    >
                      <Settings size={14} />
                      Process
                    </button>
                  )}
                </div>
              </div>
            ))}
            
            {onlineCameras.length === 0 && (
              <div className="camera-card">
                <div className="camera-video">
                  <div className="camera-placeholder">
                    <Monitor size={48} />
                    <p>No online cameras</p>
                    <small>Check your camera connections</small>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : selectedCamera && (
          <div style={{ maxWidth: '100%', margin: '0 auto' }}>
            <div className="camera-card">
              <div className="camera-header">
                <div>
                  <div className="camera-title">{selectedCamera.name}</div>
                  <div className="camera-info">
                    <span className={`status status-${selectedCamera.status}`}>
                      {selectedCamera.status}
                    </span>
                    <span>{selectedCamera.resolution}</span>
                    <span>{selectedCamera.location}</span>
                  </div>
                </div>
                <button 
                  className="btn btn-secondary"
                  onClick={() => setViewMode('grid')}
                >
                  Back to Grid
                </button>
              </div>
              
              <div className="camera-video" style={{ height: '70vh' }}>
                <img 
                  src={api.getCameraStreamUrl(selectedCamera.id)}
                  alt={`Camera ${selectedCamera.name}`}
                  className="camera-stream"
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'contain',
                    backgroundColor: '#000'
                  }}
                />
                
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
          </div>
        )}
      </div>

      {showProcessingModal && <ProcessingModal />}
    </div>
  );
};

export default LiveView;