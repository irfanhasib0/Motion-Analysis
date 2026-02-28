import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? window.location.origin 
  : 'http://localhost:9001';

const TOKEN_STORAGE_KEY = 'nvr_access_token';
const RECORDING_PLAYBACK_MODE_KEY = 'nvr_recording_playback_mode';

const getStoredToken = () => localStorage.getItem(TOKEN_STORAGE_KEY);

const getStoredRecordingPlaybackMode = () => {
  const mode = localStorage.getItem(RECORDING_PLAYBACK_MODE_KEY);
  return mode === 'stream' ? 'stream' : 'play';
};

const buildUrlWithToken = (path) => {
  const token = getStoredToken();
  const base = `${API_BASE_URL}${path}`;
  if (!token) {
    return base;
  }
  const separator = base.includes('?') ? '&' : '?';
  return `${base}${separator}access_token=${encodeURIComponent(token)}`;
};

const appendQueryParams = (url, params = {}) => {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== '');
  if (entries.length === 0) {
    return url;
  }
  const separator = url.includes('?') ? '&' : '?';
  const query = entries
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
    .join('&');
  return `${url}${separator}${query}`;
};

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 10000,
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    const token = getStoredToken();
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const api = {
  // Auth endpoints
  login: (password) => apiClient.post('/auth/login', { password }),
  setAccessToken: (token) => {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    }
  },
  getAccessToken: () => getStoredToken(),
  clearAccessToken: () => localStorage.removeItem(TOKEN_STORAGE_KEY),
  getRecordingPlaybackMode: () => getStoredRecordingPlaybackMode(),
  setRecordingPlaybackMode: (mode) => {
    const normalized = mode === 'stream' ? 'stream' : 'play';
    localStorage.setItem(RECORDING_PLAYBACK_MODE_KEY, normalized);
  },
  appendQueryParams,

  // Camera endpoints
  getCameras: () => apiClient.get('/cameras'),
  createCamera: (camera) => apiClient.post('/cameras', camera),
  updateCamera: (cameraId, updates) => apiClient.put(`/cameras/${cameraId}`, updates),
  deleteCamera: (cameraId) => apiClient.delete(`/cameras/${cameraId}`),
  startCamera: (cameraId) => apiClient.post(`/cameras/${cameraId}/start`),
  stopCamera: (cameraId) => apiClient.post(`/cameras/${cameraId}/stop`),
  
  // Recording endpoints
  startRecording: (cameraId) => {
    console.log('API startRecording called for camera:', cameraId);
    return apiClient.post(`/cameras/${cameraId}/start-recording`);
  },
  stopRecording: (cameraId) => apiClient.post(`/cameras/${cameraId}/stop-recording`),
  getRecordings: (cameraId = null) => {
    const params = cameraId ? { camera_id: cameraId } : {};
    return apiClient.get('/recordings', { params });
  },
  getRecordingStorageInfo: () => apiClient.get('/recordings/storage'),
  deleteRecording: (recordingId) => apiClient.delete(`/recordings/${recordingId}`),
  downloadRecording: (recordingId) => {
    return buildUrlWithToken(`/api/recordings/${recordingId}/download`);
  },
  
  // Streaming endpoints
  getCameraStreamUrl: (cameraId) => buildUrlWithToken(`/api/cameras/${cameraId}/stream`),
  closeCameraStream: (cameraId) => apiClient.post(`/cameras/${cameraId}/stream/close`),
  getBlankStreamUrl: (cameraId) => buildUrlWithToken(`/api/cameras/${cameraId}/stream/blank`),
  getRecordingStreamUrl: (recordingId, mode = getStoredRecordingPlaybackMode()) => {
    const endpoint = mode === 'stream'
      ? `/api/recordings/${recordingId}/stream`
      : `/api/recordings/${recordingId}/play`;
    return buildUrlWithToken(endpoint);
  },
  getProcessingStreamUrl: (cameraId) => buildUrlWithToken(`/api/cameras/${cameraId}/processing_stream`),
  getResultStreamUrl: (cameraId) => buildUrlWithToken(`/api/cameras/${cameraId}/result_stream`),
  // Processing endpoints
  getProcessingTypes: () => apiClient.get('/processing/types'),
  startProcessing: (cameraId, processorType, params = {}) => 
    apiClient.post(`/cameras/${cameraId}/processing/${processorType}/start`, params),
  stopProcessing: (cameraId) => apiClient.post(`/cameras/${cameraId}/processing/stop`),
  
  // System endpoints
  getSystemInfo: () => apiClient.get('/system/info'),
};

export default api;