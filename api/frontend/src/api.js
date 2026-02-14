import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? window.location.origin 
  : 'http://localhost:9001';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 10000,
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
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
  deleteRecording: (recordingId) => apiClient.delete(`/recordings/${recordingId}`),
  downloadRecording: (recordingId) => {
    return `${API_BASE_URL}/api/recordings/${recordingId}/download`;
  },
  
  // Streaming endpoints
  getCameraStreamUrl: (cameraId) => `${API_BASE_URL}/api/cameras/${cameraId}/stream`,
  closeCameraStream: (cameraId) => apiClient.post(`/cameras/${cameraId}/stream/close`),
  getBlankStreamUrl: (cameraId) => `${API_BASE_URL}/api/cameras/${cameraId}/stream/blank`,
  getRecordingStreamUrl: (recordingId) => `${API_BASE_URL}/api/recordings/${recordingId}/stream`,
  getProcessingStreamUrl: (cameraId) => `${API_BASE_URL}/api/cameras/${cameraId}/processing_stream`,
  // Processing endpoints
  getProcessingTypes: () => apiClient.get('/processing/types'),
  startProcessing: (cameraId, processorType, params = {}) => 
    apiClient.post(`/cameras/${cameraId}/processing/${processorType}/start`, params),
  stopProcessing: (cameraId) => apiClient.post(`/cameras/${cameraId}/processing/stop`),
  
  // System endpoints
  getSystemInfo: () => apiClient.get('/system/info'),
};

export default api;