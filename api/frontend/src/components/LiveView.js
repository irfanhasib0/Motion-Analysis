import React, { useState, useRef } from 'react';
import { Camera, Play, Clock, Activity, BarChart3, ChevronLeft, ChevronRight, Pause } from 'lucide-react';
import { api } from '../api';
import './LiveView.css';

const LiveView = ({ recordings = [], cameras = [] }) => {
  const validRecordings = Array.isArray(recordings) ? recordings : [];
  const completedRecordings = validRecordings.filter(
    (recording) => (recording?.status || '').toLowerCase() === 'completed'
  );
  const validCameras = Array.isArray(cameras) ? cameras : [];
  const [playingId, setPlayingId] = useState(null);
  const videoRefs = useRef({});
  const rowScrollRefs = useRef({});

  const recordingsByCamera = completedRecordings
    .slice()
    .sort((a, b) => new Date(b.start_time) - new Date(a.start_time))
    .reduce((acc, recording) => {
      const key = recording.camera_id || 'unknown_camera';
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(recording);
      return acc;
    }, {});

  const cameraRows = Object.keys(recordingsByCamera)
    .map((cameraId) => {
      const cameraInfo = validCameras.find((cam) => cam.id === cameraId) || { id: cameraId, name: 'Unknown Camera' };
      return {
        cameraId,
        cameraName: cameraInfo.name,
        recordings: recordingsByCamera[cameraId],
      };
    })
    .sort((a, b) => a.cameraName.localeCompare(b.cameraName));

  const getRecordingMetadata = (recording) => {
    const metadata = recording?.metadata;
    if (!metadata) return {};

    if (typeof metadata === 'string') {
      try {
        return JSON.parse(metadata);
      } catch (_error) {
        return {};
      }
    }

    return typeof metadata === 'object' ? metadata : {};
  };

  const formatTimestampParts = (value) => {
    if (!value) {
      return { date: 'N/A', time: '' };
    }

    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return { date: String(value), time: '' };
    }

    return {
      date: parsed.toLocaleDateString(undefined, { day: '2-digit', month: 'short' }),
      time: parsed.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }),
    };
  };

  const formatDuration = (value) => {
    const numericValue = Number(value);
    if (Number.isFinite(numericValue)) {
      const totalSeconds = Math.max(0, Math.round(numericValue));
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      return `${minutes}m ${seconds}s`;
    }
    return 'N/A';
  };

  const scrollRow = (cameraId, direction) => {
    const rowElement = rowScrollRefs.current[cameraId];
    if (!rowElement) return;

    const firstCard = rowElement.querySelector('.reel-card');
    const cardWidth = firstCard ? firstCard.offsetWidth : 320;
    const scrollAmount = (cardWidth + 20) * 2;

    rowElement.scrollBy({
      left: direction * scrollAmount,
      behavior: 'smooth',
    });
  };

  const handleMouseEnter = (recordingId) => {
    // For MJPEG streams, we don't need play/pause control
    // The stream is always playing once the img src is set
  };

  const handleMouseLeave = (recordingId) => {
    // For MJPEG streams, we don't need play/pause control
  };

  const handleClick = (recordingId) => {
    console.log('Click on recording:', recordingId);
    // For MJPEG streams, toggle visual indicator only
    if (playingId === recordingId) {
      setPlayingId(null);
    } else {
      setPlayingId(recordingId);
    }
  };

  if (completedRecordings.length === 0) {
    return (
      <div className="reels-container">
        <div className="empty-reels-state">
          <Camera size={64} />
          <h3>No Completed Recordings</h3>
          <p>Completed recordings will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="reels-container">
      <div className="reels-header">
        <h2>Recordings</h2>
        <div className="header-controls">
          <span className="recordings-count">{completedRecordings.length} videos</span>
          <span className="page-indicator">{cameraRows.length} camera rows</span>
        </div>
      </div>

      <div className="camera-rows">
        {cameraRows.map((row) => {
          return (
            <div key={row.cameraId} className="camera-row-card">
              <div className="camera-row-header">
                <h3>{row.cameraName}</h3>
                <div className="camera-row-meta">
                  <span>{row.recordings.length} videos</span>
                </div>
              </div>

              <div className="reels-carousel-wrapper">
                <button
                  className="carousel-nav-button prev"
                  onClick={() => scrollRow(row.cameraId, -1)}
                  title="Scroll left"
                >
                  <ChevronLeft size={32} />
                </button>

                <div
                  className="reels-grid row-grid"
                  ref={(el) => {
                    rowScrollRefs.current[row.cameraId] = el;
                  }}
                >
                  {row.recordings.map((recording) => {
                    const isPlaying = playingId === recording.id;
                    const metadata = getRecordingMetadata(recording);
                    const timestampValue = metadata.time_stamp || metadata.timestamp || recording.start_time || recording.started_at || recording.created_at;
                    const timestampParts = formatTimestampParts(timestampValue);
                    const durationValue = metadata.duration ?? recording.duration;
                    const velValue = metadata.vel;
                    const diffValue = metadata.diff;

                    return (
                      <div
                        key={recording.id}
                        className="reel-card"
                        onMouseEnter={() => handleMouseEnter(recording.id)}
                        onMouseLeave={() => handleMouseLeave(recording.id)}
                        onClick={() => handleClick(recording.id)}
                      >
                        <div className="reel-timestamp">
                          <span className="reel-date">{timestampParts.date}</span>
                          {timestampParts.time && <span className="reel-time">{timestampParts.time}</span>}
                        </div>

                        <div className="reel-thumbnail">
                          <img
                            ref={(el) => (videoRefs.current[recording.id] = el)}
                            className="reel-video"
                            src={api.appendQueryParams(api.getRecordingStreamUrl(recording.id), {
                              ts: Date.now(),
                            })}
                            alt={`Recording ${recording.id}`}
                            onLoad={() => console.log('Stream loaded:', recording.id)}
                            onError={(e) => console.error('Stream error:', recording.id, e)}
                          />

                          {!isPlaying && (
                            <div className="play-overlay">
                              <Play size={48} />
                            </div>
                          )}

                          {isPlaying && (
                            <div className="pause-indicator">
                              <Pause size={24} />
                            </div>
                          )}
                        </div>

                        <div className="reel-info">
                          <div className="recording-meta">
                            <div className="meta-item">
                              <Clock size={12} />
                              <span>{formatDuration(durationValue)}</span>
                            </div>
                            <div className="meta-item">
                              <Activity size={12} />
                              <span>{velValue ?? 'N/A'}</span>
                            </div>
                            <div className="meta-item">
                              <BarChart3 size={12} />
                              <span>{diffValue ?? 'N/A'}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <button
                  className="carousel-nav-button next"
                  onClick={() => scrollRow(row.cameraId, 1)}
                  title="Scroll right"
                >
                  <ChevronRight size={32} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default LiveView;
