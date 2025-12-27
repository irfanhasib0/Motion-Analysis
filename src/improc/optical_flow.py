import cv2
import numpy as np
import threading
from matplotlib import pyplot as plt
from trackers.trackers import SimpleTracker

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

class OpticalFlowTracker:
    def __init__(self, flow_mode="sparse", kalman_type="custom"):
        self.prev_gray = None
        self.prev_kp   = None
        self.prev_des  = None
        self.mask      = None

        colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        self.colors = []
        for _ in range(10):
            self.colors += colors

        self.n_colors = len(self.colors)
        self.bgsub    = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.gftt     = cv2.GFTTDetector_create(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=10) # 0.01, 7, 7
        self.orb      = cv2.ORB_create(nfeatures=500)
        self.tracker  = SimpleTracker(max_disappeared=10, max_distance=50)
        #self.sift     = cv2.SIFT_create(nfeatures=500,  contrastThreshold=0.04, edgeThreshold=10,  sigma=1.6, nOctaveLayers=3, firstOctave=0, scoreType=cv2.SIFT_FAST_SCORE,  patchSize=31, WTA_K=2,  useHarrisDetector=False,  k=0.04, upright=False,  scaleFactor=1.2)
        self.flow_params = dict(winSize=(15, 15), 
                                maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.kalmans = None
        self.kalman_type = kalman_type  # "custom" or "opencv"

    def _compute_dense_flow(self, prev_gray, gray):
            """Compute dense optical flow using Farneback method"""
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Convert flow to HSV visualization
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = (ang / 2).astype(np.uint8)  # Hue from angle
            hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Saturation from magnitude
            hsv[..., 2] = 255  # Full value
            
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return rgb
    
    def _init_kalmans(self):
        if self.p0 is not None:
            if self.kalman_type == "opencv":
                self.kalmans = [CvKalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.p0]
            else:
                self.kalmans = [KalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.p0]
        
    def _update_kalman(self, good_new, good_old, val_pts):
        # Update Kalman filters for valid points
        fused_positions = []
        new_kalmans = []
        if self.kalmans is None:
            if self.kalman_type == "opencv":
                self.kalmans = [CvKalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.p0]
            else:
                self.kalmans = [KalmanPoint(float(pt[0,0]), float(pt[0,1])) for pt in self.p0]

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
    
    def _detect_forground_bboxes(self, gray):
        # Apply background subtraction
        fg_mask = self.bgsub.apply(gray)
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours of the foreground objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                _mask = np.zeros_like(fg_mask)
                _mask[y:y+h, x:x+w] = fg_mask[y:y+h, x:x+w]
                results.append({'bbox': [x, y, x + w, y + h],
                                'centroid': [x + w / 2, y + h / 2],
                                'mask': _mask})
        return results
    
    def _detect_pts(self, _gray):
         # Detect corners to track
        results = self._detect_forground_bboxes(_gray)
        del_idx = []
        for i in range(len(results)):
            mask = results[i]['mask']
            p0  = self.gftt.detect(_gray, mask=mask)
            p0  = cv2.KeyPoint_convert(p0)
            if len(p0) == 0:
                del_idx.append(i)
                continue
            results[i]['keypoints_1']  = p0.reshape(-1, 1, 2)
            results[i]['keypoints_2']  = results[i]['keypoints_1'].copy()
        #self.npts = len(p0)
        for i in del_idx:
            del results[i]
        self.tracker.update(results)
        return results
    
    def match_keypoints_by_distance(self, kp1, kp2, max_distance=25.0):
        """Match keypoints based on Euclidean distance with a threshold"""
        if len(kp1) == 0 or len(kp2) == 0:
            return kp2
        
        matched_kp2 = []
        for i, point1 in enumerate(kp1):
            best_match_idx = -1
            best_dist = float('inf')
            for j, point2 in enumerate(kp2):
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                if dist < best_dist:# and dist < max_distance:
                    best_dist = dist
                    best_match_idx = j
            
            if best_match_idx >= 0 and best_dist < max_distance:
                matched_kp2.append(kp2[best_match_idx])
            else:
                matched_kp2.append(point1)
        return np.array(matched_kp2)
    
    def _compute_sparse_flow(self, frame):
        """Compute sparse optical flow using Lucas-Kanade method"""
        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.p0 = self._detect_pts(self.prev_gray)
            #self._init_kalmans()
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(len(self.p0)):
            # Calculate optical flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                gray,
                                                self.p0[i]['keypoints_1'], None,
                                                **self.flow_params)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(gray,
                                                self.prev_gray,
                                                p1, None,
                                                **self.flow_params)
            d = abs(self.p0[i]['keypoints_1'] - p0r).reshape(-1, 2).max(-1)
            good_mask = d < 1.0
        
            # Select good points
            val_pts  = (st.flatten() == 1) & good_mask
            self.p0[i]['keypoints_2'] = p1[val_pts]
            self.p0[i]['keypoints_1'] = self.p0[i]['keypoints_1'][val_pts]
            #good_new = self.match_keypoints_by_distance(good_old, good_new, max_distance=5.0)
            #update_kalman
            #good_new = self._update_kalman(good_new, good_old, val_pts)
            
        viz_frame = self._draw_pts_flow(frame)

        self.prev_gray = gray
        # Use fused positions for next iteration
        if 0:#len(good_new) > int(0.9 * self.npts):
            self.p0 = good_new.reshape(-1, 1, 2)
        else:
            # Fallback to redetection when no valid points
           self.p0 = self._detect_pts(gray)
           for i in range(len(self.p0)):
               self.p0[i]['keypoints_1'] = self.match_keypoints_by_distance(self.p0[i]['keypoints_1'], self.p0[i]['keypoints_2'], max_distance=5.0)
        return viz_frame
    
    def _draw_pts_flow(self, frame):
        viz_frame = frame.copy()
        for i in range(len(self.p0)):
            good_new = self.p0[i]['keypoints_2']
            good_old = self.p0[i]['keypoints_1']
            if self.mask is None:
                self.mask = np.zeros_like(viz_frame)
            # Draw tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                # Draw line for trajectory
                self.mask = cv2.line(self.mask, (a, b), (c, d), self.colors[j % self.n_colors], 1)
                # Draw point``
                viz_frame = cv2.circle(viz_frame, (a, b), 3, self.colors[j % self.n_colors], -1)
            
            # Combine visualization
            viz_frame = cv2.add(viz_frame, self.mask)
            cv2.rectangle(viz_frame, (int(self.p0[i]['bbox'][0]), int(self.p0[i]['bbox'][1])),
                        (int(self.p0[i]['bbox'][2]), int(self.p0[i]['bbox'][3])), (0, 255, 0), 2)
        return viz_frame
    
    def _compute_flann_match(self, desc1, desc2):
        """Compute feature matches using FLANN-based matcher"""
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test (robust to single-match cases)
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches
    
    def _compute_keypoint_flow(self, frame):
        """Compute optical flow based on ORB keypoints and FLANN matching"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.bgsub.apply(gray)
         # First frame initialization
        if self.prev_kp is None or self.prev_des is None:
            self.prev_kp, self.prev_des = self.orb.detectAndCompute(gray, mask)
            return frame
        
        kp, des = self.orb.detectAndCompute(gray, mask)
        
        # Match descriptors using FLANN
        good_matches = self._compute_flann_match(self.prev_des, des)
        # Draw matches
        kp1 = np.array([self.prev_kp[m.queryIdx].pt for m in good_matches])
        kp2 = np.array([kp[m.trainIdx].pt for m in good_matches])
        d = abs(kp1 - kp2).reshape(-1, 2).max(-1)
        good_mask = d < 25.0
        kp1 = kp1[good_mask]
        kp2 = kp2[good_mask]
        viz_frame = self._draw_pts_flow(frame.copy(), np.array(kp2), np.array(kp1))
        self.prev_kp = kp
        self.prev_des = des
        return viz_frame

    def detect(self, frame):
        #viz = self._compute_keypoint_flow(frame)
        viz = self._compute_sparse_flow(frame)
        return viz