import React, { useState } from 'react';
import { Plus, Edit, Trash2, Play, Square, Settings, Power, PowerOff } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { api } from '../api';
import './CameraList.css';

const CameraList = ({ cameras, setCameras }) => {
  console.log('CameraList component rendered with cameras:', cameras);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [newCamera, setNewCamera] = useState({
    name: '',
    camera_type: 'rtsp',
    source: '',
    resolution: '1920x1080',
    fps: 30,
    enabled: true,
    description: '',
    location: ''
  });

  // Per-camera UI settings for supporting screen
  const [cameraSettings, setCameraSettings] = useState({}); // { [id]: { sensitivity: 'low'|'medium'|'high', supportEnabled: boolean } }

  // Derive aspect ratio from resolution (e.g., "1920x1080" -> "1920 / 1080"). Fallback to 16/9.
  const parseAspect = (resolution) => {
    const m = /^([0-9]+)x([0-9]+)$/i.exec(resolution || '');
    if (m) {
      const w = parseInt(m[1], 10) || 16;
      const h = parseInt(m[2], 10) || 9;
      return `${w} / ${h}`;
    }
    return '16 / 9';
  };

  const handleSensitivityChange = (cameraId, level) => {
    setCameraSettings(prev => ({
      ...prev,
      [cameraId]: {
        ...(prev[cameraId] || { supportEnabled: false, sensitivity: 'medium' }),
        sensitivity: level
      }
    }));
  };

  const handleSupportToggle = (cameraId) => {
    setCameraSettings(prev => {
      const cur = prev[cameraId] || { sensitivity: 'medium', supportEnabled: false };
      return {
        ...prev,
        [cameraId]: { ...cur, supportEnabled: !cur.supportEnabled }
      };
    });
  };

  const handleAddCamera = async (formData) => {
    try {
      const response = await api.createCamera(formData);
      setCameras(prev => [...prev, response.data]);
      setShowAddModal(false);
      setNewCamera({
        name: '',
        camera_type: 'rtsp',
        source: '',
        resolution: '1920x1080',
        fps: 30,
        enabled: true,
        description: '',
        location: ''
      });
      toast.success('Camera added successfully');
    } catch (error) {
      toast.error('Failed to add camera: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleUpdateCamera = async (formData) => {
    try {
      const response = await api.updateCamera(formData.id || editingCamera.id, formData);
      setCameras(prev => prev.map(c => c.id === editingCamera.id ? response.data : c));
      setEditingCamera(null);
      toast.success('Camera updated successfully');
    } catch (error) {
      toast.error('Failed to update camera: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleDeleteCamera = async (cameraId) => {
    if (!window.confirm('Are you sure you want to delete this camera?')) {
      return;
    }
    
    try {
      await api.deleteCamera(cameraId);
      setCameras(prev => prev.filter(c => c.id !== cameraId));
      toast.success('Camera deleted successfully');
    } catch (error) {
      toast.error('Failed to delete camera: ' + (error.response?.data?.detail || error.message));
    }
  };
  
  const handleStartCamera = async (cameraId) => {
    console.log('handleStartCamera called for:', cameraId);
    try {
      await api.startCamera(cameraId);
      // Update camera status in local state
      setCameras(prev => prev.map(c => 
        c.id === cameraId 
          ? { ...c, status: 'online', last_seen: new Date().toISOString() }
          : c
      ));
      toast.success('Camera started');
    } catch (error) {
      console.error('Start camera error:', error);
      toast.error('Failed to start camera: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleStopCamera = async (cameraId) => {
    console.log('handleStopCamera called for:', cameraId);
    try {
      await api.stopCamera(cameraId);
      // Also close any active stream generators on the server
      try { await api.closeCameraStream(cameraId); } catch {}
      // Update camera status to offline
      setCameras(prev => prev.map(c => 
        c.id === cameraId 
          ? { ...c, status: 'offline' }
          : c
      ));
      toast.success('Camera stopped');
    } catch (error) {
      console.error('Stop camera error:', error);
      toast.error('Failed to stop camera: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleStartRecording = async (cameraId) => {
    console.log('handleStartRecording called for:', cameraId);
    try {
      console.log('Calling api.startRecording...');
      const response = await api.startRecording(cameraId);
      console.log('Recording API response:', response);
      // Update camera status to recording
      setCameras(prev => prev.map(c => 
        c.id === cameraId 
          ? { ...c, status: 'recording' }
          : c
      ));
      toast.success('Recording started');
    } catch (error) {
      console.error('Start recording error:', error);
      toast.error('Failed to start recording: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleStopRecording = async (cameraId) => {
    console.log('handleStopRecording called for:', cameraId);
    try {
      await api.stopRecording(cameraId);
      // Update camera status back to online
      setCameras(prev => prev.map(c => 
        c.id === cameraId 
          ? { ...c, status: 'online' }
          : c
      ));
      toast.success('Recording stopped');
    } catch (error) {
      console.error('Stop recording error:', error);
      toast.error('Failed to stop recording: ' + (error.response?.data?.detail || error.message));
    }
  };

  const CameraForm = ({ initialCamera, onSubmit, title, submitText }) => {
    const [form, setForm] = useState(initialCamera);
    
    const handleSubmit = async (e) => {
      e.preventDefault();
      await onSubmit(form);
    };

    return (
      <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && (setShowAddModal(false) || setEditingCamera(null))}>
        <div className="modal">
          <div className="modal-header">
            <h3 className="modal-title">{title}</h3>
            <button 
              className="modal-close" 
              onClick={() => {
                setShowAddModal(false);
                setEditingCamera(null);
              }}
            >
              ×
            </button>
          </div>
          
          <form onSubmit={handleSubmit}>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Name</label>
                <input 
                  type="text"
                  className="form-control"
                  value={form.name}
                  onChange={(e) => setForm({...form, name: e.target.value})}
                  required
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Camera Type</label>
                {/* DEBUG: Updated with recorded option - version 1.0 */}
                <select 
                  className="form-control form-select"
                  value={form.camera_type}
                  onChange={(e) => {
                    console.log('Camera type changed to:', e.target.value);
                    setForm({...form, camera_type: e.target.value});
                  }}
                >
                  <option value="recorded">Recorded Data</option>
                  <option value="rtsp">RTSP Stream</option>
                  <option value="webcam">Webcam</option>
                  <option value="ip_camera">IP Camera</option>
                </select>
              </div>
              
              <div className="form-group">
                <label className="form-label">Source</label>
                <input 
                  type="text"
                  className="form-control"
                  value={form.source}
                  onChange={(e) => setForm({...form, source: e.target.value})}
                  placeholder={
                    form.camera_type === 'recorded' ? '/path/to/video/file.mp4 or recording_id' :
                    form.camera_type === 'rtsp' ? 'rtsp://username:password@ip:port/path' :
                    form.camera_type === 'webcam' ? '0' :
                    'http://ip:port/video'
                  }
                  required
                />
              </div>
              
              <div className="grid grid-2">
                <div className="form-group">
                  <label className="form-label">Resolution</label>
                  <select 
                    className="form-control form-select"
                    value={form.resolution}
                    onChange={(e) => setForm({...form, resolution: e.target.value})}
                  >
                    <option value="320x240">320x240</option>
                    <option value="480x360">480x360</option>
                    <option value="640x480">640x480</option>
                    <option value="1280x720">1280x720</option>
                    <option value="1920x1080">1920x1080</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label className="form-label">FPS</label>
                  <input 
                    type="number"
                    className="form-control"
                    value={form.fps}
                    onChange={(e) => setForm({...form, fps: parseInt(e.target.value)})}
                    min="1"
                    max="60"
                  />
                </div>
              </div>
              
              <div className="form-group">
                <label className="form-label">Location</label>
                <input 
                  type="text"
                  className="form-control"
                  value={form.location || ''}
                  onChange={(e) => setForm({...form, location: e.target.value})}
                  placeholder="e.g., Front Door, Parking Lot"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">Description</label>
                <textarea 
                  className="form-control"
                  value={form.description || ''}
                  onChange={(e) => setForm({...form, description: e.target.value})}
                  rows="3"
                  placeholder="Optional description"
                />
              </div>
              
              <div className="form-group">
                <label className="form-label">
                  <input 
                    type="checkbox"
                    checked={form.enabled}
                    onChange={(e) => setForm({...form, enabled: e.target.checked})}
                    style={{ marginRight: '8px' }}
                  />
                  Enabled
                </label>
              </div>
            </div>
            
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={() => {
                setShowAddModal(false);
                setEditingCamera(null);
              }}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary">
                {submitText}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Cameras</h1>
        <p className="page-subtitle">Manage your cameras and recording settings</p>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">Camera List ({cameras.length})</h2>
          <button 
            className="btn btn-primary"
            onClick={() => setShowAddModal(true)}
          >
            <Plus size={16} />
            Add Camera
          </button>
        </div>
        
        <div className="camera-grid">
          {cameras.map(camera => {
            console.log('Rendering camera:', camera.name, 'status:', camera.status, 'id:', camera.id);
            return (
            <div key={camera.id} className="camera-card">
              <div className="camera-header">
                <div className="camera-title">{camera.name}</div>
                <div className="camera-info">
                  <span className={`status status-${camera.status}`}>
                    {camera.status}
                  </span>
                  <span>{camera.resolution}</span>
                  <span>{camera.camera_type}</span>
                </div>
              </div>

              <div className="camera-video" style={{ display: 'flex', gap: '16px' }}>
                {/* LEFT COLUMN: Primary camera with operation controls */}
                <div className="camera-column">
                  <div className="camera-frame">
                    {api.get_}
                    {camera.status === 'online' || camera.status === 'recording' ? (
                        <img
                          key={`${camera.id}:primary`}
                          src={api.getCameraStreamUrl(camera.id)}
                          alt={`Camera ${camera.name}`}
                          className="camera-stream"
                          style={{ width: '100%', height: 'auto', aspectRatio: parseAspect(camera.resolution), objectFit: 'contain' }}
                        />
                    ) : (
                      <img
                          src={`${api.getBlankStreamUrl(camera.id)}`}
                          alt={`Camera ${camera.name}`}
                          className="camera-stream"
                          style={{ width: '100%', height: 'auto', aspectRatio: parseAspect(camera.resolution), objectFit: 'contain' }}
                        />
                    )}
                  </div>

                  {/* Camera operation controls below left camera */}
                  <div className="camera-controls-card">
                    <button 
                      className="btn btn-warning"
                      onClick={() => {
                        console.log('Test button clicked for camera:', camera.id, camera.name);
                        alert(`Camera: ${camera.name}, Status: ${camera.status}, ID: ${camera.id}`);
                      }}
                    >
                      Test Click
                    </button>
                    
                    {camera.status === 'offline' || camera.status === 'error' ? (
                      <button 
                        className="btn btn-success"
                        onClick={() => handleStartCamera(camera.id)}
                      >
                        <Power size={14} />
                        Start
                      </button>
                    ) : (
                      <button 
                        className="btn btn-secondary"
                        onClick={() => handleStopCamera(camera.id)}
                        disabled={camera.status === 'recording'}
                      >
                        <PowerOff size={14} />
                        Stop
                      </button>
                    )}
                    
                    {camera.status === 'recording' ? (
                      <button 
                        className="btn btn-danger"
                        onClick={() => handleStopRecording(camera.id)}
                      >
                        <Square size={14} />
                        Stop Rec
                      </button>
                    ) : (
                      <button 
                        className="btn btn-success"
                        onClick={() => {
                          console.log('Record button clicked! Camera status:', camera.status, 'ID:', camera.id);
                          handleStartRecording(camera.id);
                        }}
                        disabled={camera.status !== 'online'}
                        title={camera.status !== 'online' ? `Camera must be online to record (current: ${camera.status})` : 'Start recording'}
                      >
                        <Play size={14} />
                        Record {camera.status !== 'online' ? '(Disabled)' : ''}
                      </button>
                    )}
                    
                    <button 
                      className="btn btn-primary"
                      onClick={() => setEditingCamera(camera)}
                    >
                      <Edit size={14} />
                      Edit
                    </button>
                    
                    <button 
                      className="btn btn-danger"
                      onClick={() => handleDeleteCamera(camera.id)}
                    >
                      <Trash2 size={14} />
                      Delete
                    </button>
                  </div>
                </div>

                {/* RIGHT COLUMN: Support view with view controls */}
                <div className="camera-column">
                  <div className="camera-frame">
                    {(camera.status === 'online' || camera.status === 'recording') ? (
                        <img
                          key={`${camera.id}:support`}
                          src={api.appendQueryParams(api.getProcessingStreamUrl(camera.id), {
                            view: 'support',
                          })}
                          alt={`Support view for ${camera.name}`}
                          className="camera-stream"
                          style={{ width: '100%', height: 'auto', aspectRatio: parseAspect(camera.resolution), objectFit: 'contain' }}
                        />):( 
                        <img
                          src={`${api.getBlankStreamUrl(camera.id)}`}
                          alt={`Camera ${camera.name}`}
                          className="camera-stream"
                          style={{ width: '100%', height: 'auto', aspectRatio: parseAspect(camera.resolution), objectFit: 'contain' }}
                        />)}
                  </div>

                  {/* View controls below right camera */}
                  <div className="view-controls-card">
                    <div className="sensitivity-group">
                      <span className="sensitivity-label">Sensitivity:</span>
                      {['low','medium','high'].map(level => (
                        <label 
                          key={level} 
                          className={`sensitivity-option ${(cameraSettings[camera.id]?.sensitivity || 'medium') === level ? 'active' : ''}`}
                        >
                          <input
                            type="radio"
                            name={`sens-${camera.id}`}
                            value={level}
                            checked={(cameraSettings[camera.id]?.sensitivity || 'medium') === level}
                            onChange={() => handleSensitivityChange(camera.id, level)}
                          />
                          <span className="sensitivity-option-text">{level}</span>
                        </label>
                      ))}
                    </div>
                    <button
                      className="btn btn-outline support-toggle-btn"
                      onClick={() => handleSupportToggle(camera.id)}
                    >
                      {(cameraSettings[camera.id]?.supportEnabled ?? true) ? 'Disable' : 'Enable'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
          })}

          {cameras.length === 0 && (
            <div className="camera-card">
              <div className="camera-video">
                <div className="camera-placeholder">
                  <p>No cameras configured</p>
                  <button 
                    className="btn btn-primary"
                    onClick={() => setShowAddModal(true)}
                  >
                    <Plus size={16} />
                    Add Your First Camera
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {showAddModal && (
        <CameraForm 
          initialCamera={newCamera}
          onSubmit={handleAddCamera}
          title="Add New Camera ..."
          submitText="Add Camera"
        />
      )}

      {editingCamera && (
        <CameraForm 
          initialCamera={editingCamera}
          onSubmit={handleUpdateCamera}
          title="Edit Camera"
          submitText="Update Camera"
        />
      )}
    </div>
  );
};

export default CameraList;