import React from 'react';
import { Server, HardDrive, Clock, Cpu, Activity } from 'lucide-react';

const SystemSettings = ({ systemInfo }) => {
  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatUptime = (uptime) => {
    if (!uptime) return 'N/A';
    const { days, hours, minutes } = uptime;
    return `${days}d ${hours}h ${minutes}m`;
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">System Settings</h1>
        <p className="page-subtitle">System information and configuration</p>
      </div>

      <div className="system-info-grid">
        {/* System Status */}
        <div className="info-card">
          <h3 className="info-title">
            <Server size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            System Status
          </h3>
          <div className="info-item">
            <span className="info-label">Server Status</span>
            <span className="info-value status status-online">Online</span>
          </div>
          <div className="info-item">
            <span className="info-label">Uptime</span>
            <span className="info-value">{formatUptime(systemInfo.uptime)}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Total Cameras</span>
            <span className="info-value">{systemInfo.cameras ? Object.keys(systemInfo.cameras).length : 0}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Active Recordings</span>
            <span className="info-value">
              {systemInfo.cameras 
                ? Object.values(systemInfo.cameras).filter(c => c.recording).length 
                : 0
              }
            </span>
          </div>
        </div>

        {/* Storage Information */}
        <div className="info-card">
          <h3 className="info-title">
            <HardDrive size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            Storage
          </h3>
          {systemInfo.disk_usage ? (
            <>
              <div className="info-item">
                <span className="info-label">Total Space</span>
                <span className="info-value">{formatBytes(systemInfo.disk_usage.total_gb * 1024**3)}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Used Space</span>
                <span className="info-value">{formatBytes(systemInfo.disk_usage.used_gb * 1024**3)}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Free Space</span>
                <span className="info-value">{formatBytes(systemInfo.disk_usage.free_gb * 1024**3)}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Usage</span>
                <span className="info-value">
                  {Math.round(systemInfo.disk_usage.percent_used)}%
                </span>
              </div>
              <div style={{ marginTop: '12px' }}>
                <div style={{ 
                  width: '100%', 
                  height: '8px', 
                  backgroundColor: '#4a4a4a', 
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: `${systemInfo.disk_usage.percent_used}%`, 
                    height: '100%', 
                    backgroundColor: systemInfo.disk_usage.percent_used > 90 ? '#dc3545' : 
                                   systemInfo.disk_usage.percent_used > 70 ? '#ffc107' : '#28a745',
                    borderRadius: '4px'
                  }} />
                </div>
              </div>
            </>
          ) : (
            <div className="info-item">
              <span className="info-label">Status</span>
              <span className="info-value">Loading...</span>
            </div>
          )}
        </div>

        {/* Camera Status */}
        <div className="info-card">
          <h3 className="info-title">
            <Activity size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            Camera Status
          </h3>
          {systemInfo.cameras ? (
            Object.entries(systemInfo.cameras).map(([cameraId, camera]) => (
              <div key={cameraId} className="info-item">
                <span className="info-label">{camera.name}</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className={`status status-${camera.status}`}>
                    {camera.status}
                  </span>
                  {camera.recording && (
                    <span className="status status-recording">Recording</span>
                  )}
                  {camera.processing_active && (
                    <span className="status status-warning">Processing</span>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="info-item">
              <span className="info-label">Status</span>
              <span className="info-value">No cameras configured</span>
            </div>
          )}
        </div>

        {/* Processing Status */}
        <div className="info-card">
          <h3 className="info-title">
            <Cpu size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            Video Processing
          </h3>
          {systemInfo.processing_active && Object.keys(systemInfo.processing_active).length > 0 ? (
            Object.entries(systemInfo.processing_active).map(([cameraId, processorType]) => {
              const camera = systemInfo.cameras?.[cameraId];
              return (
                <div key={cameraId} className="info-item">
                  <span className="info-label">{camera?.name || `Camera ${cameraId}`}</span>
                  <span className="info-value">{processorType}</span>
                </div>
              );
            })
          ) : (
            <div className="info-item">
              <span className="info-label">Active Processors</span>
              <span className="info-value">None</span>
            </div>
          )}
          
          <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #4a4a4a' }}>
            <div className="info-item">
              <span className="info-label">Available Processors</span>
              <span className="info-value">
                Motion Detection, Face Detection, Edge Detection, Color Filter
              </span>
            </div>
          </div>
        </div>

        {/* API Information */}
        <div className="info-card">
          <h3 className="info-title">
            <Server size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            API Information
          </h3>
          <div className="info-item">
            <span className="info-label">API Version</span>
            <span className="info-value">v1.0.0</span>
          </div>
          <div className="info-item">
            <span className="info-label">Backend</span>
            <span className="info-value">FastAPI</span>
          </div>
          <div className="info-item">
            <span className="info-label">Frontend</span>
            <span className="info-value">React</span>
          </div>
          <div className="info-item">
            <span className="info-label">Video Processing</span>
            <span className="info-value">OpenCV</span>
          </div>
          <div className="info-item">
            <span className="info-label">WebSocket</span>
            <span className="info-value status status-online">Connected</span>
          </div>
        </div>

        {/* Configuration */}
        <div className="info-card">
          <h3 className="info-title">
            <Clock size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
            Configuration
          </h3>
          <div className="info-item">
            <span className="info-label">Recordings Directory</span>
            <span className="info-value">./recordings</span>
          </div>
          <div className="info-item">
            <span className="info-label">Default Resolution</span>
            <span className="info-value">1920x1080</span>
          </div>
          <div className="info-item">
            <span className="info-label">Default FPS</span>
            <span className="info-value">30</span>
          </div>
          <div className="info-item">
            <span className="info-label">Video Format</span>
            <span className="info-value">MP4 (H.264)</span>
          </div>
          <div className="info-item">
            <span className="info-label">Streaming Format</span>
            <span className="info-value">MJPEG</span>
          </div>
        </div>
      </div>

      <div className="content-section">
        <div className="section-header">
          <h2 className="section-title">System Logs</h2>
        </div>
        
        <div className="card">
          <div style={{ 
            backgroundColor: '#1a1a1a', 
            color: '#00ff00', 
            padding: '16px', 
            fontFamily: 'monospace',
            fontSize: '12px',
            borderRadius: '4px',
            height: '300px',
            overflowY: 'scroll'
          }}>
            <div>[{new Date().toISOString()}] INFO: NVR Server started successfully</div>
            <div>[{new Date().toISOString()}] INFO: FastAPI server running on http://0.0.0.0:8000</div>
            <div>[{new Date().toISOString()}] INFO: WebSocket server initialized</div>
            <div>[{new Date().toISOString()}] INFO: Recording service initialized</div>
            <div>[{new Date().toISOString()}] INFO: Streaming service initialized</div>
            <div>[{new Date().toISOString()}] INFO: Processing service initialized</div>
            <div>[{new Date().toISOString()}] INFO: System ready for camera connections</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemSettings;