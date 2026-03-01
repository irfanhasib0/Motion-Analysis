import React, { useEffect, useState, useRef } from 'react';
import { Camera, Play, ChevronLeft, ChevronRight, Pause, Maximize2, X, Download, Trash2 } from 'lucide-react';
import { api } from '../api';
import {
  TimeFrameBadge,
  FrameStepButtons,
  RecordingMetaInfo,
  getRecordingMetadata,
  formatDuration,
  formatPlaybackTime,
  resolvePlaybackFps,
  getRecordingPlaybackViewData,
  buildRecordingsByCamera,
  buildCameraRows,
  buildRowMetricsData,
} from './EventClipsUtils';
import './LiveView.css';

const EventClips = ({ recordings = [], cameras = [] }) => {
  // ===== Tunable interaction parameters =====
  const MOUSE_DRAG_SENSITIVITY = 2.2;
  const TOUCH_DRAG_SENSITIVITY = 1.8;

  // ===== Source data normalization =====
  const validRecordings = Array.isArray(recordings) ? recordings : [];
  const [removedRecordingIds, setRemovedRecordingIds] = useState({});
  const completedRecordings = validRecordings.filter(
    (recording) => (recording?.status || '').toLowerCase() === 'completed' && !removedRecordingIds[recording.id]
  );
  const validCameras = Array.isArray(cameras) ? cameras : [];

  // ===== UI state =====
  const [playingId, setPlayingId] = useState(null);
  const [hoveredId, setHoveredId] = useState(null);
  const [highlightedRecordingId, setHighlightedRecordingId] = useState(null);
  const [playbackMode, setPlaybackMode] = useState(api.getRecordingPlaybackMode());
  const [playbackStatsById, setPlaybackStatsById] = useState({});
  const [expandedContext, setExpandedContext] = useState(null);

  // ===== Mutable refs for media, drag, and seeking =====
  const videoRefs = useRef({});
  const expandedVideoRefs = useRef({});
  const reelCardRefs = useRef({});
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

  // ===== Utility: snap row scroll to closest card =====
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

  // ===== Utility: stop/unload inline media when expanded modal is active =====
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

  // ===== Derived structure: recordings grouped by camera =====
  const recordingsByCamera = buildRecordingsByCamera(completedRecordings);
  const cameraRows = buildCameraRows(recordingsByCamera, validCameras);

  // ===== Playback stat and seek handlers =====
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
    const fps = resolvePlaybackFps(recording, metadata);

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

  // ===== Row and card interactions =====
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

  const handleTimelinePointClick = (cameraId, recordingId) => {
    setHighlightedRecordingId(recordingId);
    setHoveredId(recordingId);

    const rowElement = rowScrollRefs.current[cameraId];
    const cardElement = reelCardRefs.current[recordingId];
    if (rowElement && cardElement) {
      const targetLeft = Math.max(0, cardElement.offsetLeft - 8);
      rowElement.scrollTo({
        left: targetLeft,
        behavior: 'smooth',
      });
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

  const handleDeleteRecording = async (recordingId) => {
    const confirmed = window.confirm('Delete this recording?');
    if (!confirmed) {
      return;
    }

    try {
      await api.deleteRecording(recordingId);
      setRemovedRecordingIds((current) => ({ ...current, [recordingId]: true }));
      if (highlightedRecordingId === recordingId) {
        setHighlightedRecordingId(null);
      }
      if (hoveredId === recordingId) {
        setHoveredId(null);
      }
      if (playingId === recordingId) {
        setPlayingId(null);
      }
    } catch (error) {
      console.error('Failed to delete recording:', error);
      window.alert(error?.response?.data?.detail || 'Failed to delete recording');
    }
  };

  // ===== Desktop drag-to-scroll handlers =====
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

  // ===== Touch drag-to-scroll handlers =====
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

  // ===== Expanded card modal handlers =====
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

  // Keep inline card playback stopped while expanded view is open.
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
          const {
            chartMetrics,
            chartMaxVelocity,
            chartMaxBgDiff,
            axisTicks,
          } = buildRowMetricsData(row.recordings);

          return (
            <div key={row.cameraId} className="camera-row-card">
              <div className="camera-row-header">
                <h3>{row.cameraName}</h3>
                <div className="camera-row-meta">
                  <span>{row.recordings.length} videos</span>
                </div>
              </div>

              <div className="camera-row-metrics-card">
                {/* Mini analytics card for each camera row */}
                <div className="camera-row-metrics-header">
                  <span>Timestamp vs Velocity / bg_diff</span>
                  <div className="metrics-legend">
                    <span><span className="legend-dot velocity" /> Velocity</span>
                    <span><span className="legend-dot bgdiff" /> bg_diff</span>
                  </div>
                </div>

                <div className="metrics-bar-chart" role="img" aria-label="Velocity and bg_diff bar plot by timestamp">
                  {chartMetrics.map((metric) => {
                    if (metric.xPercent === null) {
                      return null;
                    }
                    const velocityHeight = Math.max(2, (metric.velocity / chartMaxVelocity) * 100);
                    const bgDiffHeight = Math.max(2, (metric.bgDiff / chartMaxBgDiff) * 100);
                    const markerHeight = Math.max(velocityHeight, bgDiffHeight);
                    return (
                      <React.Fragment key={metric.id}>
                        <div
                          className="metrics-bar-group"
                          style={{ left: `${metric.xPercent}%` }}
                        >
                          <span
                            className="metrics-bar velocity"
                            style={{ height: `${velocityHeight}%` }}
                            title={`Velocity: ${metric.velocity.toFixed(3)}`}
                          />
                          <span
                            className="metrics-bar bgdiff"
                            style={{ height: `${bgDiffHeight}%` }}
                            title={`bg_diff: ${metric.bgDiff.toFixed(0)}`}
                          />
                        </div>

                        <button
                          type="button"
                          className={`motion-point-btn${highlightedRecordingId === metric.id ? ' active' : ''}`}
                          style={{
                            left: `${metric.xPercent}%`,
                            bottom: `${Math.min(96, markerHeight + 2)}%`,
                          }}
                          onClick={() => handleTimelinePointClick(row.cameraId, metric.id)}
                          title={`${metric.timeLabel} | vel: ${metric.velocity.toFixed(3)} | bg_diff: ${metric.bgDiff.toFixed(0)}`}
                        >
                          <span className="point-dot" />
                        </button>
                      </React.Fragment>
                    );
                  })}
                </div>
                <div className="metrics-time-axis">
                  {axisTicks.map((tick, index) => (
                    <div key={`${row.cameraId}-tick-${index}`} className="metrics-axis-tick" style={{ left: `${tick.xPercent}%` }}>
                      <span className="metrics-axis-mark" />
                      {tick.showLabel && <span className="metrics-axis-label">{tick.label}</span>}
                    </div>
                  ))}
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
                    const {
                      timestampParts,
                      durationValue,
                      velValue,
                      diffValue,
                      playbackStats,
                      playbackDuration,
                      playbackProgress,
                      playbackFrame,
                      totalFrames,
                    } = getRecordingPlaybackViewData(recording, playbackStatsById[recording.id]);

                    return (
                      <div
                        key={recording.id}
                        ref={(el) => {
                          reelCardRefs.current[recording.id] = el;
                        }}
                        className={`reel-card${highlightedRecordingId === recording.id ? ' reel-card-highlighted' : ''}`}
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
                          {/* Per-card playback controls */}
                          <div className="reel-playback-controls" onClick={(event) => event.stopPropagation()}>
                            <div className="reel-progress-header">
                              <div className="reel-progress-left">
                                <TimeFrameBadge timeText={formatPlaybackTime(playbackStats.currentTime)} frame={playbackFrame} />
                              </div>
                              <FrameStepButtons
                                disabled={playbackMode !== 'play'}
                                onStepBack={(event) => {
                                  event.stopPropagation();
                                  stepFrame(recording, -1);
                                }}
                                onStepForward={(event) => {
                                  event.stopPropagation();
                                  stepFrame(recording, 1);
                                }}
                              />
                              <div className="reel-progress-right">
                                <TimeFrameBadge timeText={formatPlaybackTime(playbackDuration)} frame={totalFrames} />
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
                          <div className="reel-meta-row">
                            <RecordingMetaInfo durationText={formatDuration(durationValue)} velValue={velValue} diffValue={diffValue} />
                            <div className="reel-card-actions" onClick={(event) => event.stopPropagation()}>
                              <button
                                type="button"
                                className="reel-action-btn"
                                title="Download"
                                onClick={() => handleDownloadRecording(recording.id)}
                              >
                                <Download size={12} />
                              </button>
                              <button
                                type="button"
                                className="reel-action-btn danger"
                                title="Delete"
                                onClick={() => handleDeleteRecording(recording.id)}
                              >
                                <Trash2 size={12} />
                              </button>
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
        // Expanded modal view for focused reel playback.
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
                  const {
                    timestampParts,
                    durationValue,
                    velValue,
                    diffValue,
                    playbackStats,
                    playbackDuration,
                    playbackProgress,
                    playbackFrame,
                    totalFrames,
                  } = getRecordingPlaybackViewData(expandedRecording, playbackStatsById[expandedRecording.id]);

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
                              <TimeFrameBadge timeText={formatPlaybackTime(playbackStats.currentTime)} frame={playbackFrame} />
                            </div>
                            <FrameStepButtons
                              disabled={playbackMode !== 'play'}
                              onStepBack={(event) => {
                                event.stopPropagation();
                                stepFrame(expandedRecording, -1, true);
                              }}
                              onStepForward={(event) => {
                                event.stopPropagation();
                                stepFrame(expandedRecording, 1, true);
                              }}
                            />
                            <div className="reel-progress-right">
                              <TimeFrameBadge timeText={formatPlaybackTime(playbackDuration)} frame={totalFrames} />
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
                        <RecordingMetaInfo durationText={formatDuration(durationValue)} velValue={velValue} diffValue={diffValue} />
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

export default EventClips;
