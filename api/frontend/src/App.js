import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Camera, Video, Settings, BarChart3, Monitor } from 'lucide-react';

import Dashboard from './components/Dashboard';
import CameraList from './components/CameraList';
import RecordingList from './components/RecordingList';
import EventView from './components/EventView';
import SystemSettings from './components/SystemSettings';
import LoginScreen from './components/LoginScreen';
import { api } from './api';

import './App.css';

function App() {
  const [cameras, setCameras] = useState([]);
  const [recordings, setRecordings] = useState([]);
  const [systemInfo, setSystemInfo] = useState({});
  const [loading, setLoading] = useState(true);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(true);

  useEffect(() => {
    const bootstrap = async () => {
      setLoading(true);
      try {
        await loadInitialData();
        setIsAuthenticated(true);
      } catch (error) {
        const statusCode = error?.response?.status;
        if (statusCode === 401) {
          setIsAuthenticated(false);
          api.clearAccessToken();
        } else {
          setIsAuthenticated(true);
        }
      } finally {
        setLoading(false);
      }
    };

    bootstrap();
  }, []);

  useEffect(() => {
    if (!isAuthenticated) {
      return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/main`);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      switch (message.type) {
        case 'camera_added':
          setCameras(prev => [...prev, message.camera]);
          break;
        case 'camera_updated':
          setCameras(prev => prev.map(c => c.id === message.camera.id ? message.camera : c));
          break;
        case 'camera_deleted':
          setCameras(prev => prev.filter(c => c.id !== message.camera_id));
          break;
        case 'recording_started':
          setRecordings(prev => [...prev, message.recording]);
          setCameras(prev => prev.map(c =>
            c.id === message.camera_id
              ? { ...c, status: 'recording', recording_id: message.recording.id }
              : c
          ));
          break;
        case 'recording_stopped':
          setCameras(prev => prev.map(c =>
            c.id === message.camera_id
              ? { ...c, status: 'online', recording_id: null }
              : c
          ));
          loadInitialData().catch((error) => {
            console.error('Failed to refresh data after recording stopped:', error);
          });
          break;
        default:
          break;
      }
    };

    return () => {
      ws.close();
    };
  }, [isAuthenticated]);

  const handleLogin = async (password) => {
    setAuthLoading(true);
    setAuthError('');
    try {
      const response = await api.login(password);
      const token = response?.data?.access_token;
      if (!token) {
        setAuthError('Login failed: no access token returned.');
        return;
      }
      api.setAccessToken(token);
      setIsAuthenticated(true);
      setLoading(true);
      await loadInitialData();
      setLoading(false);
    } catch (error) {
      setAuthError(error?.response?.data?.detail || 'Invalid password');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    api.clearAccessToken();
    setIsAuthenticated(false);
    setCameras([]);
    setRecordings([]);
    setSystemInfo({});
  };

  const loadInitialData = async () => {
    const [camerasRes, recordingsRes, systemRes] = await Promise.all([
      api.getCameras(),
      api.getRecordings(),
      api.getSystemInfo()
    ]);
    console.log('Initial data loaded:', camerasRes.data, recordingsRes.data, systemRes.data);
    setCameras(camerasRes.data);
    setRecordings(recordingsRes.data);
    setSystemInfo(systemRes.data);
  };

  if (loading) {
    return (
      <div className="app-loading">
        <div className="loading"></div>
        <p>Loading NVR Server...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginScreen onLogin={handleLogin} loading={authLoading} error={authError} />;
  }

  return (
    <Router>
      <div className="app">
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#2a2a2a',
              color: '#ffffff',
              border: '1px solid #4a4a4a'
            }
          }}
        />
        
        <nav className="sidebar">
          <div className="sidebar-header">
            <h2 className="sidebar-title">NVR Server</h2>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ marginTop: 12, width: '100%' }}
              onClick={handleLogout}
            >
              Logout
            </button>
          </div>
          
          <div className="sidebar-nav">
            <NavLink to="/" className="nav-link" end>
              <BarChart3 size={20} />
              Dashboard
            </NavLink>
            <NavLink to="/cameras" className="nav-link">
              <Camera size={20} />
              Cameras
            </NavLink>
            <NavLink to="/live" className="nav-link">
              <Monitor size={20} />
              Event View
            </NavLink>
            <NavLink to="/recordings" className="nav-link">
              <Video size={20} />
              Recordings
            </NavLink>
            <NavLink to="/settings" className="nav-link">
              <Settings size={20} />
              Settings
            </NavLink>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route 
              path="/" 
              element={
                <Dashboard 
                  cameras={cameras}
                  recordings={recordings}
                  systemInfo={systemInfo}
                />
              } 
            />
            <Route 
              path="/cameras" 
              element={
                <CameraList 
                  cameras={cameras}
                  setCameras={setCameras}
                />
              } 
            />
            <Route 
              path="/live" 
              element={<EventView cameras={cameras} recordings={recordings} />} 
            />
            <Route 
              path="/recordings" 
              element={
                <RecordingList 
                  recordings={recordings}
                  setRecordings={setRecordings}
                  cameras={cameras}
                />
              } 
            />
            <Route 
              path="/settings" 
              element={<SystemSettings systemInfo={systemInfo} />} 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;