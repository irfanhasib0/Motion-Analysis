import React, { useState, useRef, useEffect } from 'react';
import { Camera, Play, Calendar, Clock, ChevronLeft, ChevronRight, Pause } from 'lucide-react';
import toast from 'react-hot-toast';
import './LiveView.css';

const VIDEOS_PER_PAGE = 4;

const LiveView = ({ recordings = [], cameras = [] }) => {
  const validRecordings = Array.isArray(recordings) ? recordings : [];
  const validCameras = Array.isArray(cameras) ? cameras : [];
  const [currentPage, setCurrentPage] = useState(0);
  const [playingId, setPlayingId] = useState(null);
  const videoRefs = useRef({});

  // Calculate pagination
  const totalPages = Math.ceil(validRecordings.length / VIDEOS_PER_PAGE);
  const startIdx = currentPage * VIDEOS_PER_PAGE;
  const endIdx = startIdx + VIDEOS_PER_PAGE;
  const currentRecordings = validRecordings.slice(startIdx, endIdx);

  // Find camera info for recording
  const getCameraInfo = (cameraId) => {
    return validCameras.find(cam => cam.id === cameraId) || { name: 'Unknown Camera' };
  };

  // Stop all videos when page changes
  useEffect(() => {
    // Clear playing state when page changes
    setPlayingId(null);
  }, [currentPage]);

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

  const handlePrevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  if (validRecordings.length === 0) {
    return (
      <div className="reels-container">
        <div className="empty-reels-state">
          <Camera size={64} />
          <h3>No Recordings Available</h3>
          <p>Start recording from cameras to see videos here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="reels-container">
      <div className="reels-header">
        <h2>Recordings</h2>
        <div className="header-controls">
          <span className="recordings-count">{validRecordings.length} videos</span>
          <span className="page-indicator">
            Page {currentPage + 1} of {totalPages}
          </span>
        </div>
      </div>

      <div className="reels-carousel-wrapper">
        {/* Previous Button */}
        {currentPage > 0 && (
          <button className="carousel-nav-button prev" onClick={handlePrevPage}>
            <ChevronLeft size={32} />
          </button>
        )}

        {/* Video Grid */}
        <div className="reels-grid">
          {currentRecordings.map((recording) => {
            const cameraInfo = getCameraInfo(recording.camera_id);
            const isPlaying = playingId === recording.id;

            return (
              <div
                key={recording.id}
                className="reel-card"
                onMouseEnter={() => handleMouseEnter(recording.id)}
                onMouseLeave={() => handleMouseLeave(recording.id)}
                onClick={() => handleClick(recording.id)}
              >
                <div className="reel-thumbnail">
                  <img
                    ref={(el) => (videoRefs.current[recording.id] = el)}
                    className="reel-video"
                    src={`/api/recordings/${recording.id}/stream`}
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

                  <div className="reel-duration">
                    {recording.duration}s
                  </div>
                </div>

                <div className="reel-info">
                  <div className="camera-name">
                    <Camera size={14} />
                    <span>{cameraInfo.name}</span>
                  </div>
                  <div className="recording-meta">
                    <div className="meta-item">
                      <Calendar size={12} />
                      <span>{new Date(recording.start_time).toLocaleDateString()}</span>
                    </div>
                    <div className="meta-item">
                      <Clock size={12} />
                      <span>{new Date(recording.start_time).toLocaleTimeString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Next Button */}
        {currentPage < totalPages - 1 && (
          <button className="carousel-nav-button next" onClick={handleNextPage}>
            <ChevronRight size={32} />
          </button>
        )}
      </div>

      {/* Pagination Dots */}
      {totalPages > 1 && (
        <div className="pagination-dots">
          {Array.from({ length: totalPages }).map((_, idx) => (
            <button
              key={idx}
              className={`dot ${idx === currentPage ? 'active' : ''}`}
              onClick={() => setCurrentPage(idx)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default LiveView;
