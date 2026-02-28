import React, { useEffect, useState } from 'react';
import { Play, Download, Trash2, Clock, HardDrive, Camera, Search } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { api } from '../api';

const RecordingList = ({ recordings, setRecordings, cameras }) => {
  const [selectedRecording, setSelectedRecording] = useState(null);
  const [filterCamera, setFilterCamera] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [searchTerm, setSearchTerm] = useState('');
  const [storageInfo, setStorageInfo] = useState(null);
  const [playbackMode, setPlaybackMode] = useState(api.getRecordingPlaybackMode());

  const loadStorageInfo = async () => {
    try {
      const response = await api.getRecordingStorageInfo();
      setStorageInfo(response.data);
    } catch (error) {
      console.error('Failed to load recording storage info:', error);
    }
  };

  useEffect(() => {
    loadStorageInfo();
  }, [recordings.length]);

  const handleDeleteRecording = async (recordingId) => {
    if (!window.confirm('Are you sure you want to delete this recording?')) {
      return;
    }
    
    try {
      await api.deleteRecording(recordingId);
      setRecordings(prev => prev.filter(r => r.id !== recordingId));
      toast.success('Recording deleted successfully');
    } catch (error) {
      toast.error('Failed to delete recording: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleDownloadRecording = (recordingId) => {
    const url = api.downloadRecording(recordingId);
    const link = document.createElement('a');
    link.href = url;
    link.download = '';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '0s';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  // Filter and sort recordings
  const filteredRecordings = recordings
    .filter(recording => {
      const camera = cameras.find(c => c.id === recording.camera_id);
      const cameraName = camera?.name || 'Unknown Camera';
      
      const matchesCamera = !filterCamera || recording.camera_id === filterCamera;
      const matchesStatus = !filterStatus || recording.status === filterStatus;
      const matchesSearch = !searchTerm || 
        cameraName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        recording.filename.toLowerCase().includes(searchTerm.toLowerCase());
      
      return matchesCamera && matchesStatus && matchesSearch;
    })
    .sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'camera':
          const aCameraName = cameras.find(c => c.id === a.camera_id)?.name || 'Unknown';
          const bCameraName = cameras.find(c => c.id === b.camera_id)?.name || 'Unknown';
          aValue = aCameraName;
          bValue = bCameraName;
          break;
        case 'duration':
          aValue = a.duration || 0;
          bValue = b.duration || 0;
          break;
        case 'file_size':
          aValue = a.file_size || 0;
          bValue = b.file_size || 0;
          break;
        default:
          aValue = new Date(a[sortBy]);
          bValue = new Date(b[sortBy]);
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

  const VideoPlayerModal = ({ recording, onClose }) => {
    const camera = cameras.find(c => c.id === recording.camera_id);
    
    return (
      <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
        <div className="modal" style={{ maxWidth: '90vw', maxHeight: '90vh' }}>
          <div className="modal-header">
            <h3 className="modal-title">
              {camera?.name || 'Unknown Camera'} - {recording.filename}
            </h3>
            <button className="modal-close" onClick={onClose}>×</button>
          </div>
          
          <div className="modal-body" style={{ padding: 0 }}>
            <div className="video-container" style={{ height: '60vh' }}>
              {playbackMode === 'stream' ? (
                <img
                  src={api.appendQueryParams(api.getRecordingStreamUrl(recording.id, 'stream'), {
                    ts: Date.now(),
                  })}
                  alt={`Recording ${recording.filename}`}
                  className="video-stream"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    backgroundColor: '#000'
                  }}
                />
              ) : (
                <video
                  src={api.getRecordingStreamUrl(recording.id, 'play')}
                  className="video-stream"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    backgroundColor: '#000'
                  }}
                  controls
                  autoPlay
                  playsInline
                />
              )}
            </div>
          </div>
          
          <div className="modal-footer">
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flex: 1 }}>
              <span className="btn btn-secondary" style={{ cursor: 'default' }}>
                <Clock size={14} /> {formatDuration(recording.duration)}
              </span>
              <span className="btn btn-secondary" style={{ cursor: 'default' }}>
                <HardDrive size={14} /> {formatBytes(recording.file_size)}
              </span>
            </div>
            <button 
              className="btn btn-primary"
              onClick={() => handleDownloadRecording(recording.id)}
            >
              <Download size={14} />
              Download
            </button>
            <button className="btn btn-secondary" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Recordings</h1>
        <p className="page-subtitle">Browse and manage your recorded videos</p>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">Video Recordings ({filteredRecordings.length})</h2>
          <div className="section-subtitle" style={{ marginTop: '8px' }}>
            Available Storage: {storageInfo ? formatBytes(storageInfo.free_bytes) : 'Loading...'}
          </div>
          <div style={{ marginTop: '12px' }}>
            <label className="form-label" style={{ marginRight: '8px' }}>Playback Mode</label>
            <select
              className="form-control form-select"
              value={playbackMode}
              onChange={(e) => {
                const mode = e.target.value === 'stream' ? 'stream' : 'play';
                setPlaybackMode(mode);
                api.setRecordingPlaybackMode(mode);
              }}
              style={{ width: '180px', display: 'inline-block' }}
            >
              <option value="play">File Playback</option>
              <option value="stream">Legacy Stream</option>
            </select>
          </div>
        </div>
        
        {/* Filters */}
        <div className="card" style={{ marginBottom: '24px' }}>
          <div className="grid grid-4">
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Search</label>
              <div style={{ position: 'relative' }}>
                <input 
                  type="text"
                  className="form-control"
                  placeholder="Search recordings..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{ paddingLeft: '36px' }}
                />
                <Search 
                  size={16} 
                  style={{ 
                    position: 'absolute', 
                    left: '12px', 
                    top: '50%', 
                    transform: 'translateY(-50%)',
                    color: '#b0b0b0'
                  }}
                />
              </div>
            </div>
            
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Camera</label>
              <select 
                className="form-control form-select"
                value={filterCamera}
                onChange={(e) => setFilterCamera(e.target.value)}
              >
                <option value="">All Cameras</option>
                {cameras.map(camera => (
                  <option key={camera.id} value={camera.id}>
                    {camera.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Status</label>
              <select 
                className="form-control form-select"
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
              >
                <option value="">All Status</option>
                <option value="recording">Recording</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
            </div>
            
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Sort By</label>
              <div style={{ display: 'flex', gap: '4px' }}>
                <select 
                  className="form-control form-select"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <option value="created_at">Date Created</option>
                  <option value="camera">Camera</option>
                  <option value="duration">Duration</option>
                  <option value="file_size">File Size</option>
                </select>
                <button 
                  className="btn btn-secondary"
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  style={{ minWidth: '60px' }}
                >
                  {sortOrder === 'asc' ? '↑' : '↓'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Recording List */}
        <div className="recording-list">
          {filteredRecordings.length === 0 ? (
            <div className="recording-item">
              <div className="recording-info">
                <div className="recording-name">
                  {recordings.length === 0 ? 'No recordings found' : 'No recordings match your filters'}
                </div>
                <div className="recording-details">
                  <span>
                    {recordings.length === 0 
                      ? 'Start recording from cameras to see them here'
                      : 'Try adjusting your search filters'
                    }
                  </span>
                </div>
              </div>
            </div>
          ) : (
            filteredRecordings.map(recording => {
              const camera = cameras.find(c => c.id === recording.camera_id);
              return (
                <div key={recording.id} className="recording-item">
                  <div className="recording-info">
                    <div className="recording-name">
                      <Camera size={16} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                      {camera?.name || 'Unknown Camera'} - {recording.filename}
                    </div>
                    <div className="recording-details">
                      <span><Clock size={12} /> {formatDate(recording.created_at)}</span>
                      <span><Clock size={12} /> {formatDuration(recording.duration)}</span>
                      <span><HardDrive size={12} /> {formatBytes(recording.file_size)}</span>
                      <span>{recording.resolution || 'Unknown'}</span>
                      <span className={`status status-${recording.status}`}>
                        {recording.status}
                      </span>
                    </div>
                  </div>
                  
                  <div className="recording-actions">
                    {recording.status === 'completed' && (
                      <>
                        <button 
                          className="btn btn-primary"
                          onClick={() => setSelectedRecording(recording)}
                        >
                          <Play size={14} />
                          Play
                        </button>
                        <button 
                          className="btn btn-secondary"
                          onClick={() => handleDownloadRecording(recording.id)}
                        >
                          <Download size={14} />
                          Download
                        </button>
                      </>
                    )}
                    
                    <button 
                      className="btn btn-danger"
                      onClick={() => handleDeleteRecording(recording.id)}
                    >
                      <Trash2 size={14} />
                      Delete
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {selectedRecording && (
        <VideoPlayerModal 
          recording={selectedRecording}
          onClose={() => setSelectedRecording(null)}
        />
      )}
    </div>
  );
};

export default RecordingList;