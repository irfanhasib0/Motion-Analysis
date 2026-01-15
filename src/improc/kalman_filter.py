import cv2
import numpy as np

'''
def _init_kalmans(self):
        if self.prev_pts is not None:
            if self.kalman_type == "opencv":
                self.kalmans = [CvKalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.prev_pts]
            else:
                self.kalmans = [KalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.prev_pts]
        
    def _update_kalman(self, good_new, good_old, val_pts):
        # Update Kalman filters for valid points
        fused_positions = []
        new_kalmans = []
        if self.kalmans is None:
            if self.kalman_type == "opencv":
                self.kalmans = [CvKalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.prev_pts]
            else:
                self.kalmans = [KalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.prev_pts]

        indices = np.where(val_pts)[0]
        for idx_i, i in enumerate(indices):
            if self.kalman_type == "opencv":
                kf = self.kalmans[i] if i < len(self.kalmans) else CvKalmanPoint(float(good_old[idx_i][0]), float(good_old[idx_i][1]))
            else:
                kf = self.kalmans[i] if i < len(self.kalmans) else KalmanPoint(float(good_old[idx_i][0]), float(good_old[idx_i][1]))
            # Predict
            kf.predict(dt=1.0)
            # Update with LK position using available API
            meas_pos = good_new[idx_i]
            if hasattr(kf, 'update_pos'):
                kf.update_pos(meas_pos)
            elif hasattr(kf, 'correct'):
                kf.correct(meas_pos)
            else:
                kf.update(meas_pos)
            # Optional velocity update only for custom filter
            if self.kalman_type != "opencv":
                disp = good_new[idx_i] - good_old[idx_i]
                kf.update_vel(disp)
            #fused_positions.append([float(kf.x[0,0]), float(kf.x[1,0])])
            good_new[idx_i] = [kf.x[0,0], kf.x[1,0]]
            new_kalmans.append(kf)
        self.kalmans = new_kalmans
        return good_new
'''

class KalmanPoint:
    """Constant-velocity Kalman filter for 2D points."""
    def __init__(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * 1e-2
        self.Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3]).astype(np.float32)
        self.R_pos = np.diag([1e-2, 1e-2]).astype(np.float32)
        self.R_vel = np.diag([1e-2, 1e-2]).astype(np.float32)
        self.H_pos = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=np.float32)
        self.H_vel = np.array([[0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)

    def _A(self, dt):
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

    def predict(self, dt=1.0):
        A = self._A(dt)
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q
        return self.x

    def update_pos(self, z):
        z = np.asarray(z, dtype=np.float32).reshape(2, 1)
        y = z - self.H_pos @ self.x
        S = self.H_pos @ self.P @ self.H_pos.T + self.R_pos
        K = self.P @ self.H_pos.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H_pos) @ self.P
        return self.x

    # Aliases for compatibility with different calling code
    def correct(self, z):
        return self.update_pos(z)

    def update(self, z):
        return self.update_pos(z)

    def update_vel(self, v):
        v = np.asarray(v, dtype=np.float32).reshape(2, 1)
        y = v - self.H_vel @ self.x
        S = self.H_vel @ self.P @ self.H_vel.T + self.R_vel
        K = self.P @ self.H_vel.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H_vel) @ self.P
        return self.x

class CvKalmanPoint:
    """OpenCV cv2.KalmanFilter wrapper for constant-velocity 2D points."""
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.diag([1e-4, 1e-4, 1e-3, 1e-3]).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([1e-2, 1e-2]).astype(np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.x = self.kf.statePost.copy()

    def _set_transition(self, dt):
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)

    def predict(self, dt=1.0):
        self._set_transition(dt)
        _ = self.kf.predict()
        self.x = self.kf.statePre.copy()
        return self.x

    def update_pos(self, z):
        meas = np.asarray(z, dtype=np.float32).reshape(2, 1)
        _ = self.kf.correct(meas)
        self.x = self.kf.statePost.copy()
        return self.x

    def update_vel(self, v):
        # Position-only measurement in OpenCV model; ignore explicit velocity updates
        return self.x

    # Aliases for compatibility with different calling code
    def correct(self, z):
        return self.update_pos(z)

    def update(self, z):
        return self.update_pos(z)