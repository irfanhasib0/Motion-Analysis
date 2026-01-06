import React from 'react';
import { Camera, Video, HardDrive, Clock } from 'lucide-react';

const Dashboard = ({ cameras, recordings, systemInfo }) => {
  const onlineCameras = cameras.filter(c => c.status === 'online').length;
  const recordingCameras = cameras.filter(c => c.status === 'recording').length;
  const totalRecordings = recordings.length;
  const recentRecordings = recordings.slice(0, 5);

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
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">System overview and statistics</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{cameras.length}</div>
          <div className="stat-label">Total Cameras</div>
          <div className="stat-change positive">
            {onlineCameras} online • {recordingCameras} recording
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-value">{totalRecordings}</div>
          <div className="stat-label">Total Recordings</div>
          <div className="stat-change">
            {recordings.filter(r => r.status === 'recording').length} active
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-value">
            {systemInfo.disk_usage ? 
              Math.round(systemInfo.disk_usage.percent_used) + '%' : 
              'N/A'
            }
          </div>
          <div className="stat-label">Disk Usage</div>
          <div className="stat-change">
            {systemInfo.disk_usage ? 
              `${formatBytes(systemInfo.disk_usage.used_gb * 1024**3)} used` :
              'Loading...'
            }
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-value">
            {systemInfo.uptime ? 
              `${systemInfo.uptime.days}d ${systemInfo.uptime.hours}h` : 
              'N/A'
            }
          </div>
          <div className="stat-label">System Uptime</div>
          <div className="stat-change">
            {systemInfo.processing_active ? 
              `${Object.keys(systemInfo.processing_active).length} processing` :
              'No active processing'
            }
          </div>
        </div>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">Camera Status</h2>
        </div>
        
        <div className="camera-grid">
          {cameras.map(camera => (
            <div key={camera.id} className="camera-card">
              <div className="camera-header">
                <div className="camera-title">{camera.name}</div>
                <div className="camera-info">
                  <span className={`status status-${camera.status}`}>
                    {camera.status}
                  </span>
                  <span>{camera.resolution || 'Unknown'}</span>
                  <span>{camera.camera_type}</span>
                </div>
              </div>
              
              <div className="camera-video">
                {camera.status === 'online' || camera.status === 'recording' ? (
                  <img 
                    src={`/api/cameras/${camera.id}/stream`}
                    alt={`Camera ${camera.name}`}
                    className="camera-stream"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'flex';
                    }}
                  />
                ) : (
                  <div className="camera-placeholder">
                    <Camera size={48} />
                    <p>{camera.status === 'offline' ? 'Camera Offline' : 'Camera Error'}</p>
                  </div>
                )}
                {camera.status === 'offline' && (
                  <div className="camera-placeholder">
                    <Camera size={48} />
                    <p>Camera Offline</p>
                  </div>
                )}
              </div>

              <div className="camera-controls">
                <span className="btn btn-secondary" style={{cursor: 'default'}}>
                  {camera.location || 'No location'}
                </span>
                {camera.processing_active && (
                  <span className="btn btn-warning" style={{cursor: 'default'}}>
                    Processing: {camera.processing_type}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">Recent Recordings</h2>
        </div>
        
        <div className="recording-list">
          {recentRecordings.length === 0 ? (
            <div className="recording-item">
              <div className="recording-info">
                <div className="recording-name">No recordings found</div>
                <div className="recording-details">
                  <span>Start recording from cameras to see them here</span>
                </div>
              </div>
            </div>
          ) : (
            recentRecordings.map(recording => {
              const camera = cameras.find(c => c.id === recording.camera_id);
              return (
                <div key={recording.id} className="recording-item">
                  <div className="recording-info">
                    <div className="recording-name">
                      {camera?.name || 'Unknown Camera'} - {recording.filename}
                    </div>
                    <div className="recording-details">
                      <span><Clock size={12} /> {formatDate(recording.created_at)}</span>
                      <span><Video size={12} /> {formatDuration(recording.duration)}</span>
                      <span><HardDrive size={12} /> {formatBytes(recording.file_size)}</span>
                      <span className={`status status-${recording.status}`}>
                        {recording.status}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;