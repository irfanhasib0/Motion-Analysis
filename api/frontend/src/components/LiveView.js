import React, { useEffect, useState, useRef } from 'react';
import { Camera, Play, Clock, Activity, BarChart3, ChevronLeft, ChevronRight, Pause, Image, Maximize2, X } from 'lucide-react';
import { api } from '../api';
import './LiveView.css';

const LiveView = ({ recordings = [], cameras = [] }) => {
  const MOUSE_DRAG_SENSITIVITY = 2.2;
  const TOUCH_DRAG_SENSITIVITY = 1.8;

  const validRecordings = Array.isArray(recordings) ? recordings : [];
  const completedRecordings = validRecordings.filter(
    (recording) => (recording?.status || '').toLowerCase() === 'completed'
  );
  const validCameras = Array.isArray(cameras) ? cameras : [];
  const [playingId, setPlayingId] = useState(null);
  const [hoveredId, setHoveredId] = useState(null);
  const [playbackMode, setPlaybackMode] = useState(api.getRecordingPlaybackMode());
  const [playbackStatsById, setPlaybackStatsById] = useState({});
  const [expandedContext, setExpandedContext] = useState(null);
  const videoRefs = useRef({});
  const expandedVideoRefs = useRef({});
  const rowScrollRefs = useRef({});
  const seekStateRef = useRef({});
  const rowDragStateRef = useRef({
    isDragging: false,
    cameraId: null,
    pointerId: null,
    startX: 0,
    startScrollLeft: 0,
    moved: false,
  });
  const touchDragStateRef = useRef({
    active: false,
    cameraId: null,
    startX: 0,
    startY: 0,
    startScrollLeft: 0,
    horizontalLocked: false,
    moved: false,
  });
  const suppressClickUntilRef = useRef(0);

  const snapRowToNearestCard = (cameraId) => {
    const rowElement = rowScrollRefs.current[cameraId];
    if (!rowElement) {
      return;
    }

    const cards = Array.from(rowElement.querySelectorAll('.reel-card'));
    if (cards.length === 0) {
      return;
    }

    const targetScrollLeft = rowElement.scrollLeft;
    let nearestCard = cards[0];
    let nearestDistance = Math.abs(cards[0].offsetLeft - targetScrollLeft);

    for (let idx = 1; idx < cards.length; idx += 1) {
      const distance = Math.abs(cards[idx].offsetLeft - targetScrollLeft);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestCard = cards[idx];
      }
    }

    rowElement.scrollTo({
      left: nearestCard.offsetLeft,
      behavior: 'smooth',
    });
  };

  const stopInlineMediaPlayback = () => {
    Object.values(videoRefs.current).forEach((element) => {
      if (!element) {
        return;
      }

      const tag = String(element.tagName || '').toUpperCase();

      if (tag === 'VIDEO') {
        try {
          element.pause();
          element.removeAttribute('src');
          element.load();
        } catch (_error) {
        }
      }

      if (tag === 'IMG') {
        try {
          element.src = '';
        } catch (_error) {
        }
      }
    });
  };

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

  const formatPlaybackTime = (value) => {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue) || numericValue < 0) {
      return '00:00.00';
    }
    const totalSeconds = Math.floor(numericValue);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const centiseconds = Math.floor((numericValue - totalSeconds) * 100);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${centiseconds.toString().padStart(2, '0')}`;
  };

  const updatePlaybackStats = (recordingId, patch) => {
    setPlaybackStatsById((current) => ({
      ...current,
      [recordingId]: {
        ...(current[recordingId] || { currentTime: 0, duration: 0 }),
        ...patch,
      },
    }));
  };

  const handleVideoTimeUpdate = (recordingId, event) => {
    const seekState = seekStateRef.current[recordingId];
    if (seekState?.isSeeking) {
      return;
    }
    const currentTime = Number(event?.target?.currentTime) || 0;
    const duration = Number(event?.target?.duration) || 0;
    updatePlaybackStats(recordingId, { currentTime, duration });
  };

  const handleVideoLoadedMetadata = (recordingId, event) => {
    const currentTime = Number(event?.target?.currentTime) || 0;
    const duration = Number(event?.target?.duration) || 0;
    updatePlaybackStats(recordingId, { currentTime, duration });
  };

  const handleSeekChange = (recordingId, value, useExpandedRef = false) => {
    setHoveredId(recordingId);
    const refMap = useExpandedRef ? expandedVideoRefs.current : videoRefs.current;
    const video = refMap[recordingId];
    if (!video) {
      return;
    }
    const nextTime = Number(value);
    if (!Number.isFinite(nextTime)) {
      return;
    }
    video.currentTime = nextTime;
    updatePlaybackStats(recordingId, {
      currentTime: nextTime,
      duration: Number(video.duration) || 0,
    });
  };

  const handleSeekStart = (recordingId, useExpandedRef = false) => {
    const refMap = useExpandedRef ? expandedVideoRefs.current : videoRefs.current;
    const video = refMap[recordingId];

    const wasPlaying = !!(video && !video.paused);
    if (wasPlaying) {
      video.pause();
    }

    seekStateRef.current[recordingId] = {
      isSeeking: true,
      wasPlaying,
      useExpandedRef,
    };
  };

  const handleSeekEnd = (recordingId) => {
    const seekState = seekStateRef.current[recordingId];
    if (!seekState) {
      return;
    }

    seekStateRef.current[recordingId] = {
      ...seekState,
      isSeeking: false,
    };

    const refMap = seekState.useExpandedRef ? expandedVideoRefs.current : videoRefs.current;
    const video = refMap[recordingId];
    if (seekState.wasPlaying && video && typeof video.play === 'function') {
      video.play().catch(() => {});
      setPlayingId(recordingId);
    }
  };

  const stepFrame = (recording, direction, useExpandedRef = false) => {
    if (playbackMode !== 'play') {
      return;
    }

    const recordingId = recording.id;
    setHoveredId(recordingId);

    const refMap = useExpandedRef ? expandedVideoRefs.current : videoRefs.current;
    const video = refMap[recordingId];
    if (!video) {
      return;
    }

    if (!video.paused) {
      video.pause();
      setPlayingId(null);
    }

    const metadata = getRecordingMetadata(recording);
    const fpsFromRecording = Number(recording.fps);
    const fpsFromMetadata = Number(metadata.fps);
    const fps = Number.isFinite(fpsFromRecording) && fpsFromRecording > 0
      ? fpsFromRecording
      : (Number.isFinite(fpsFromMetadata) && fpsFromMetadata > 0 ? fpsFromMetadata : 30);

    const frameSeconds = 1 / fps;
    const duration = Number(video.duration) || Number(playbackStatsById[recordingId]?.duration) || 0;
    const maxTime = Math.max(0, duration - frameSeconds);
    const nextTime = Math.min(maxTime, Math.max(0, (Number(video.currentTime) || 0) + (direction * frameSeconds)));

    video.currentTime = nextTime;
    updatePlaybackStats(recordingId, {
      currentTime: nextTime,
      duration,
    });
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
    setHoveredId(recordingId);
    const video = videoRefs.current[recordingId];
    if (video && typeof video.play === 'function') {
      video.play().catch(() => {});
    }
  };

  const handleMouseLeave = (recordingId) => {
    setHoveredId((current) => (current === recordingId ? null : current));
    const video = videoRefs.current[recordingId];
    if (recordingId !== playingId && video && typeof video.pause === 'function') {
      video.pause();
    }
  };

  const handleClick = (recordingId) => {
    if (Date.now() < suppressClickUntilRef.current) {
      return;
    }

    if (playbackMode === 'stream') {
      setPlayingId((current) => (current === recordingId ? null : recordingId));
      return;
    }

    const video = videoRefs.current[recordingId];
    if (!video) {
      setPlayingId((current) => (current === recordingId ? null : recordingId));
      return;
    }

    if (video.paused) {
      video.play().catch(() => {});
      setPlayingId(recordingId);
    } else {
      video.pause();
      setPlayingId(null);
    }
  };

  const handlePlaybackModeChange = (event) => {
    const mode = event.target.value === 'stream' ? 'stream' : 'play';
    api.setRecordingPlaybackMode(mode);
    setPlaybackMode(mode);
    setPlayingId(null);
    setHoveredId(null);
  };

  const handleRowPointerDown = (cameraId, event) => {
    if (event.pointerType !== 'mouse') {
      return;
    }

    const target = event.target;
    if (
      target instanceof Element
      && target.closest('button, input, select, textarea')
    ) {
      return;
    }

    if (event.pointerType === 'mouse' && event.button !== 0) {
      return;
    }

    const rowElement = rowScrollRefs.current[cameraId];
    if (!rowElement) {
      return;
    }

    rowDragStateRef.current = {
      isDragging: true,
      cameraId,
      pointerId: event.pointerId,
      startX: event.clientX,
      startScrollLeft: rowElement.scrollLeft,
      moved: false,
    };

    rowElement.classList.add('dragging');
    if (rowElement.setPointerCapture) {
      rowElement.setPointerCapture(event.pointerId);
    }
  };

  const handleRowPointerMove = (cameraId, event) => {
    const dragState = rowDragStateRef.current;
    if (!dragState.isDragging || dragState.cameraId !== cameraId) {
      return;
    }

    const rowElement = rowScrollRefs.current[cameraId];
    if (!rowElement) {
      return;
    }

    const deltaX = event.clientX - dragState.startX;
    if (Math.abs(deltaX) > 4) {
      rowDragStateRef.current.moved = true;
      event.preventDefault();
    }
    rowElement.scrollLeft = dragState.startScrollLeft - (deltaX * MOUSE_DRAG_SENSITIVITY);
  };

  const endRowDrag = (cameraId, event) => {
    const dragState = rowDragStateRef.current;
    if (!dragState.isDragging || dragState.cameraId !== cameraId) {
      return;
    }

    const rowElement = rowScrollRefs.current[cameraId];
    if (rowElement) {
      rowElement.classList.remove('dragging');
      if (rowElement.releasePointerCapture && dragState.pointerId !== null) {
        try {
          rowElement.releasePointerCapture(dragState.pointerId);
        } catch (_error) {
        }
      }
    }

    if (dragState.moved) {
      suppressClickUntilRef.current = Date.now() + 180;
      snapRowToNearestCard(cameraId);
    }

    rowDragStateRef.current = {
      isDragging: false,
      cameraId: null,
      pointerId: null,
      startX: 0,
      startScrollLeft: 0,
      moved: false,
    };
  };

  const handleRowTouchStart = (cameraId, event) => {
    if (!event.touches || event.touches.length === 0) {
      return;
    }

    const target = event.target;
    if (
      target instanceof Element
      && target.closest('button, input, select, textarea')
    ) {
      return;
    }

    const rowElement = rowScrollRefs.current[cameraId];
    if (!rowElement) {
      return;
    }

    const touch = event.touches[0];
    touchDragStateRef.current = {
      active: true,
      cameraId,
      startX: touch.clientX,
      startY: touch.clientY,
      startScrollLeft: rowElement.scrollLeft,
      horizontalLocked: false,
      moved: false,
    };
  };

  const handleRowTouchMove = (cameraId, event) => {
    const dragState = touchDragStateRef.current;
    if (!dragState.active || dragState.cameraId !== cameraId || !event.touches || event.touches.length === 0) {
      return;
    }

    const rowElement = rowScrollRefs.current[cameraId];
    if (!rowElement) {
      return;
    }

    const touch = event.touches[0];
    const dx = touch.clientX - dragState.startX;
    const dy = touch.clientY - dragState.startY;

    if (!dragState.horizontalLocked) {
      if (Math.abs(dx) > 8 && Math.abs(dx) > Math.abs(dy)) {
        dragState.horizontalLocked = true;
      } else if (Math.abs(dy) > 8 && Math.abs(dy) >= Math.abs(dx)) {
        dragState.active = false;
        return;
      }
    }

    if (!dragState.horizontalLocked) {
      return;
    }

    if (Math.abs(dx) > 4) {
      dragState.moved = true;
    }

    event.preventDefault();
    rowElement.scrollLeft = dragState.startScrollLeft - (dx * TOUCH_DRAG_SENSITIVITY);
  };

  const handleRowTouchEnd = (cameraId) => {
    const dragState = touchDragStateRef.current;
    if (!dragState.active || dragState.cameraId !== cameraId) {
      touchDragStateRef.current = {
        active: false,
        cameraId: null,
        startX: 0,
        startY: 0,
        startScrollLeft: 0,
        horizontalLocked: false,
        moved: false,
      };
      return;
    }

    if (dragState.moved) {
      suppressClickUntilRef.current = Date.now() + 220;
      snapRowToNearestCard(cameraId);
    }

    touchDragStateRef.current = {
      active: false,
      cameraId: null,
      startX: 0,
      startY: 0,
      startScrollLeft: 0,
      horizontalLocked: false,
      moved: false,
    };
  };

  const handleOpenExpanded = (recording, cameraId, index) => {
    stopInlineMediaPlayback();
    setExpandedContext({ cameraId, index });
    setHoveredId(recording.id);
    setPlayingId(null);
  };

  const handleCloseExpanded = () => {
    setExpandedContext(null);
  };

  const handleExpandedNavigate = (direction) => {
    setExpandedContext((current) => {
      if (!current) {
        return current;
      }
      const rowRecordings = recordingsByCamera[current.cameraId] || [];
      if (rowRecordings.length === 0) {
        return current;
      }

      const nextIndex = Math.min(
        rowRecordings.length - 1,
        Math.max(0, current.index + direction)
      );
      const nextRecording = rowRecordings[nextIndex];
      if (nextRecording) {
        stopInlineMediaPlayback();
        setHoveredId(nextRecording.id);
        setPlayingId(null);
      }
      return { ...current, index: nextIndex };
    });
  };

  useEffect(() => {
    if (expandedContext) {
      stopInlineMediaPlayback();
      setPlayingId(null);
    }
  }, [expandedContext]);

  const expandedRowRecordings = expandedContext
    ? (recordingsByCamera[expandedContext.cameraId] || [])
    : [];
  const expandedRecording = expandedContext
    ? expandedRowRecordings[expandedContext.index] || null
    : null;
  const canNavigatePrev = !!expandedContext && expandedContext.index > 0;
  const canNavigateNext = !!expandedContext && expandedContext.index < (expandedRowRecordings.length - 1);

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
          <select
            className="form-control form-select"
            value={playbackMode}
            onChange={handlePlaybackModeChange}
            style={{ width: '170px' }}
          >
            <option value="play">File Playback</option>
            <option value="stream">Legacy Stream</option>
          </select>
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
                  onPointerDown={(event) => handleRowPointerDown(row.cameraId, event)}
                  onPointerMove={(event) => handleRowPointerMove(row.cameraId, event)}
                  onPointerUp={(event) => endRowDrag(row.cameraId, event)}
                  onPointerCancel={(event) => endRowDrag(row.cameraId, event)}
                  onTouchStart={(event) => handleRowTouchStart(row.cameraId, event)}
                  onTouchMove={(event) => handleRowTouchMove(row.cameraId, event)}
                  onTouchEnd={() => handleRowTouchEnd(row.cameraId)}
                  onTouchCancel={() => handleRowTouchEnd(row.cameraId)}
                >
                  {row.recordings.map((recording, recordingIndex) => {
                    const isPlaying = playingId === recording.id;
                    const isHovered = hoveredId === recording.id;
                    const shouldLoadVideo = !expandedContext && (isPlaying || isHovered);
                    const playbackStats = playbackStatsById[recording.id] || { currentTime: 0, duration: 0 };
                    const playbackFpsFromRecording = Number(recording.fps);
                    const playbackDuration = playbackStats.duration > 0
                      ? playbackStats.duration
                      : (Number(recording.duration) || 0);
                    const playbackProgress = playbackDuration > 0
                      ? Math.min(100, Math.max(0, (playbackStats.currentTime / playbackDuration) * 100))
                      : 0;
                    const metadata = getRecordingMetadata(recording);
                    const playbackFpsFromMetadata = Number(metadata.fps);
                    const playbackFps = Number.isFinite(playbackFpsFromRecording) && playbackFpsFromRecording > 0
                      ? playbackFpsFromRecording
                      : (Number.isFinite(playbackFpsFromMetadata) && playbackFpsFromMetadata > 0 ? playbackFpsFromMetadata : 30);
                    const playbackFrame = Math.max(0, Math.floor(playbackStats.currentTime * playbackFps));
                    const totalFrames = Math.max(0, Math.floor(playbackDuration * playbackFps));
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
                          <button
                            type="button"
                            className="enlarge-btn"
                            title="Enlarge playback"
                            onClick={(event) => {
                              event.stopPropagation();
                              handleOpenExpanded(recording, row.cameraId, recordingIndex);
                            }}
                          >
                            <Maximize2 size={14} />
                          </button>

                          {playbackMode === 'stream' ? (
                            <img
                              ref={(el) => (videoRefs.current[recording.id] = el)}
                              className="reel-video"
                              src={!expandedContext ? api.appendQueryParams(api.getRecordingStreamUrl(recording.id, 'stream'), {
                                ts: Date.now(),
                              }) : undefined}
                              alt={`Recording ${recording.id}`}
                              onLoad={() => console.log('Stream loaded:', recording.id)}
                              onError={(e) => console.error('Stream error:', recording.id, e)}
                            />
                          ) : (
                            <video
                              ref={(el) => (videoRefs.current[recording.id] = el)}
                              className="reel-video"
                              src={shouldLoadVideo ? api.getRecordingStreamUrl(recording.id, 'play') : undefined}
                              muted
                              loop
                              playsInline
                              preload="none"
                              autoPlay={shouldLoadVideo}
                              onLoadedMetadata={(event) => handleVideoLoadedMetadata(recording.id, event)}
                              onTimeUpdate={(event) => handleVideoTimeUpdate(recording.id, event)}
                              onLoadedData={() => console.log('Video loaded:', recording.id)}
                              onError={(e) => console.error('Video error:', recording.id, e)}
                            />
                          )}

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
                          <div className="reel-playback-controls" onClick={(event) => event.stopPropagation()}>
                            <div className="reel-progress-header">
                              <div className="reel-progress-left">
                                <span className="reel-time-frame-badge">
                                  <span>{formatPlaybackTime(playbackStats.currentTime)}</span>
                                  <span className="reel-time-frame-divider" />
                                  <span className="reel-frame-inline">
                                    <Image size={10} />
                                    <span>{playbackFrame}</span>
                                  </span>
                                </span>
                              </div>
                              <div className="reel-frame-controls">
                                <button
                                  type="button"
                                  className="frame-step-btn"
                                  disabled={playbackMode !== 'play'}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    stepFrame(recording, -1);
                                  }}
                                >
                                  <ChevronLeft size={12} />
                                </button>
                                <button
                                  type="button"
                                  className="frame-step-btn"
                                  disabled={playbackMode !== 'play'}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    stepFrame(recording, 1);
                                  }}
                                >
                                  <ChevronRight size={12} />
                                </button>
                              </div>
                              <div className="reel-progress-right">
                                <span className="reel-time-frame-badge">
                                  <span>{formatPlaybackTime(playbackDuration)}</span>
                                  <span className="reel-time-frame-divider" />
                                  <span className="reel-frame-inline">
                                    <Image size={10} />
                                    <span>{totalFrames}</span>
                                  </span>
                                </span>
                              </div>
                            </div>
                            <input
                              type="range"
                              min={0}
                              max={Math.max(playbackDuration, 0.01)}
                              step={0.01}
                              value={Math.min(playbackStats.currentTime, Math.max(playbackDuration, 0.01))}
                              disabled={playbackMode !== 'play'}
                              onMouseDown={() => {
                                setHoveredId(recording.id);
                                handleSeekStart(recording.id, false);
                              }}
                              onMouseUp={() => handleSeekEnd(recording.id)}
                              onTouchStart={() => {
                                setHoveredId(recording.id);
                                handleSeekStart(recording.id, false);
                              }}
                              onTouchEnd={() => handleSeekEnd(recording.id)}
                              onChange={(event) => handleSeekChange(recording.id, event.target.value)}
                              className="reel-progress-slider"
                              style={{ '--progress': `${playbackProgress}%` }}
                            />
                          </div>
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

      {expandedRecording && (
        <div className="enlarged-overlay" onClick={handleCloseExpanded}>
          <div className="enlarged-content" onClick={(event) => event.stopPropagation()}>
            <div className="enlarged-header">
              <span className="enlarged-title">{expandedRecording.filename || expandedRecording.id}</span>
              <button
                type="button"
                className="enlarged-close"
                onClick={handleCloseExpanded}
                title="Close enlarged playback"
              >
                <X size={16} />
              </button>
            </div>
            <div className="enlarged-card-stage">
              <div className="reel-card enlarged-reel-card">
                {(() => {
                  const metadata = getRecordingMetadata(expandedRecording);
                  const timestampValue = metadata.time_stamp || metadata.timestamp || expandedRecording.start_time || expandedRecording.started_at || expandedRecording.created_at;
                  const timestampParts = formatTimestampParts(timestampValue);
                  const durationValue = metadata.duration ?? expandedRecording.duration;
                  const velValue = metadata.vel;
                  const diffValue = metadata.diff;
                  const playbackStats = playbackStatsById[expandedRecording.id] || { currentTime: 0, duration: 0 };
                  const playbackFpsFromRecording = Number(expandedRecording.fps);
                  const playbackDuration = playbackStats.duration > 0
                    ? playbackStats.duration
                    : (Number(expandedRecording.duration) || 0);
                  const playbackProgress = playbackDuration > 0
                    ? Math.min(100, Math.max(0, (playbackStats.currentTime / playbackDuration) * 100))
                    : 0;
                  const playbackFpsFromMetadata = Number(metadata.fps);
                  const playbackFps = Number.isFinite(playbackFpsFromRecording) && playbackFpsFromRecording > 0
                    ? playbackFpsFromRecording
                    : (Number.isFinite(playbackFpsFromMetadata) && playbackFpsFromMetadata > 0 ? playbackFpsFromMetadata : 30);
                  const playbackFrame = Math.max(0, Math.floor(playbackStats.currentTime * playbackFps));
                  const totalFrames = Math.max(0, Math.floor(playbackDuration * playbackFps));

                  return (
                    <>
                      <div className="reel-timestamp">
                        <span className="reel-date">{timestampParts.date}</span>
                        {timestampParts.time && <span className="reel-time">{timestampParts.time}</span>}
                      </div>

                      <div className="reel-thumbnail">
                        <button
                          type="button"
                          className="enlarged-inline-nav prev"
                          disabled={!canNavigatePrev}
                          onClick={(event) => {
                            event.stopPropagation();
                            handleExpandedNavigate(-1);
                          }}
                          title="Previous"
                        >
                          <ChevronLeft size={20} />
                        </button>

                        {playbackMode === 'stream' ? (
                          <img
                            ref={(el) => (expandedVideoRefs.current[expandedRecording.id] = el)}
                            className="reel-video"
                            src={api.appendQueryParams(api.getRecordingStreamUrl(expandedRecording.id, 'stream'), {
                              ts: Date.now(),
                            })}
                            alt={`Recording ${expandedRecording.id}`}
                          />
                        ) : (
                          <video
                            ref={(el) => (expandedVideoRefs.current[expandedRecording.id] = el)}
                            className="reel-video"
                            src={api.getRecordingStreamUrl(expandedRecording.id, 'play')}
                            muted
                            loop={false}
                            playsInline
                            controls
                            autoPlay
                            onLoadedMetadata={(event) => handleVideoLoadedMetadata(expandedRecording.id, event)}
                            onTimeUpdate={(event) => handleVideoTimeUpdate(expandedRecording.id, event)}
                          />
                        )}

                        <button
                          type="button"
                          className="enlarged-inline-nav next"
                          disabled={!canNavigateNext}
                          onClick={(event) => {
                            event.stopPropagation();
                            handleExpandedNavigate(1);
                          }}
                          title="Next"
                        >
                          <ChevronRight size={20} />
                        </button>
                      </div>

                      <div className="reel-info">
                        <div className="reel-playback-controls">
                          <div className="reel-progress-header">
                            <div className="reel-progress-left">
                              <span className="reel-time-frame-badge">
                                <span>{formatPlaybackTime(playbackStats.currentTime)}</span>
                                <span className="reel-time-frame-divider" />
                                <span className="reel-frame-inline">
                                  <Image size={10} />
                                  <span>{playbackFrame}</span>
                                </span>
                              </span>
                            </div>
                            <div className="reel-frame-controls">
                              <button
                                type="button"
                                className="frame-step-btn"
                                disabled={playbackMode !== 'play'}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  stepFrame(expandedRecording, -1, true);
                                }}
                              >
                                <ChevronLeft size={12} />
                              </button>
                              <button
                                type="button"
                                className="frame-step-btn"
                                disabled={playbackMode !== 'play'}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  stepFrame(expandedRecording, 1, true);
                                }}
                              >
                                <ChevronRight size={12} />
                              </button>
                            </div>
                            <div className="reel-progress-right">
                              <span className="reel-time-frame-badge">
                                <span>{formatPlaybackTime(playbackDuration)}</span>
                                <span className="reel-time-frame-divider" />
                                <span className="reel-frame-inline">
                                  <Image size={10} />
                                  <span>{totalFrames}</span>
                                </span>
                              </span>
                            </div>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={Math.max(playbackDuration, 0.01)}
                            step={0.01}
                            value={Math.min(playbackStats.currentTime, Math.max(playbackDuration, 0.01))}
                            disabled={playbackMode !== 'play'}
                            onMouseDown={() => handleSeekStart(expandedRecording.id, true)}
                            onMouseUp={() => handleSeekEnd(expandedRecording.id)}
                            onTouchStart={() => handleSeekStart(expandedRecording.id, true)}
                            onTouchEnd={() => handleSeekEnd(expandedRecording.id)}
                            onChange={(event) => handleSeekChange(expandedRecording.id, event.target.value, true)}
                            className="reel-progress-slider"
                            style={{ '--progress': `${playbackProgress}%` }}
                          />
                        </div>
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
                    </>
                  );
                })()}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveView;
