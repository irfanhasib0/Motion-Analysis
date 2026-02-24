import React, { useState } from 'react';

function LoginScreen({ onLogin, loading, error }) {
  const [password, setPassword] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    await onLogin(password);
  };

  return (
    <div className="app-loading" style={{ minHeight: '100vh' }}>
      <div
        style={{
          width: '100%',
          maxWidth: 420,
          background: 'rgba(255,255,255,0.85)',
          border: '1px solid rgba(0, 137, 123, 0.15)',
          borderRadius: 16,
          padding: 24,
          boxShadow: '0 12px 36px rgba(0, 150, 136, 0.2)',
        }}
      >
        <h2 className="page-title" style={{ fontSize: 26, marginBottom: 8 }}>NVR Login</h2>
        <p className="page-subtitle" style={{ marginBottom: 16 }}>Enter API password to continue.</p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              className="form-control"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
              required
            />
          </div>

          {error ? (
            <p style={{ color: '#d32f2f', marginBottom: 12 }}>{error}</p>
          ) : null}

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default LoginScreen;
