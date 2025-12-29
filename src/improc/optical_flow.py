import cv2
import numpy as np
import threading
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from trackers.trackers import SimpleTracker
from collections import deque
from improc.kalman_filter import KalmanPoint, CvKalmanPoint

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
        self.tracker  = SimpleTracker(max_disappeared=30, max_distance=100)
        #self.sift     = cv2.SIFT_create(nfeatures=500,  contrastThreshold=0.04, edgeThreshold=10,  sigma=1.6, nOctaveLayers=3, firstOctave=0, scoreType=cv2.SIFT_FAST_SCORE,  patchSize=31, WTA_K=2,  useHarrisDetector=False,  k=0.04, upright=False,  scaleFactor=1.2)
        self.flow_params = dict(winSize=(15, 15), 
                                maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.kalmans = None
        self.kalman_type = kalman_type  # "custom" or "opencv"
        self.keypoint_queue = deque(maxlen=5)

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
                                'centroid': [y + h / 2, x + w / 2],
                                'mask': _mask})
        return results
    
    def _detect_pts(self, _gray):
         # Detect corners to track
        results = self._detect_forground_bboxes(_gray)
        del_idx = []
        for i in range(len(results)):
            mask = results[i]['mask']

            # Keep GFTT keypoints to access detection scores (kp.response)
            kps = self.gftt.detect(_gray, mask=mask)
            pts = cv2.KeyPoint_convert(kps)
            if len(pts) == 0:
                del_idx.append(i)
                continue
            
            # First sort by y-coordinate, then by x-coordinate within each row 
            indices = np.lexsort((pts[:, 0], pts[:, 1]))
            pts = pts[indices]
            scores = np.array([kps[j].response for j in indices], dtype=np.float32)
            
            # Divide bbox into grid and keep highest confidence point per cell
            bbox = results[i]['bbox']
            grid_n, grid_m = 3, 2  # Grid dimensions (rows, cols)
            w = (bbox[2] - bbox[0]) / grid_m
            h = (bbox[3] - bbox[1]) / grid_n

            filtered_pts = []
            filtered_scores = []

            for row in range(grid_n):
                for col in range(grid_m):
                    # Define grid cell boundaries
                    cell_x1 = bbox[0] + col * w
                    cell_y1 = bbox[1] + row * h
                    cell_x2 = cell_x1 + w
                    cell_y2 = cell_y1 + h
                    
                    # Find points within this cell
                    in_cell = (pts[:, 0] >= cell_x1) & (pts[:, 0] < cell_x2) & \
                              (pts[:, 1] >= cell_y1) & (pts[:, 1] < cell_y2)
                    
                    if np.any(in_cell):
                        # Keep point with highest confidence in this cell
                        cell_indices = np.where(in_cell)[0]
                        best_idx = cell_indices[scores[cell_indices].argmax()]
                        filtered_pts.append(pts[best_idx])
                        filtered_scores.append(scores[best_idx])
                    elif self.prev_pts is not None and i in self.prev_pts.keys():
                        # Use previous frame's point if available             
                        pt = self.prev_pts[i]['keypoints_2'][row * grid_m + col][0]
                        filtered_pts.append(pt)
                        filtered_scores.append(0.0)  # Low confidence for centroid
                    else:
                        # If no point in this cell, use cell center as fallback
                        centroid = np.array([cell_x1 + w/2, cell_y1 + h/2])
                        filtered_pts.append(centroid)
                        filtered_scores.append(0.0)  # Low confidence for centroid

            pts = np.array(filtered_pts, dtype=np.float32)
            scores = np.array(filtered_scores, dtype=np.float32)

            results[i]['keypoints_1']  = pts.reshape(-1, 1, 2)
            results[i]['keypoints_2']  = results[i]['keypoints_1'].copy()
            results[i]['keypoint_scores'] = scores.reshape(-1, 1)  # per-point detection score

        del_idx = sorted(del_idx, reverse=True)
        for i in del_idx:
            del results[i]
        results = self.tracker.update(results)
        return results
    
    def match_keypoints_by_distance(self, kp1, kp2, matcher='hungarian'):
        # Handle empty inputs gracefully
        if kp1 is None or kp2 is None:
            return kp1 if kp2 is None else kp2
        if kp1.size == 0:
            return kp2
        if kp2.size == 0:
            return kp1

        _kp1 = kp1.reshape(-1, 2)[:, None, :]
        _kp2 = kp2.reshape(-1, 2)[None, :, :]
        dists = ((_kp1 - _kp2) ** 2).sum(axis=2).astype(np.float64)
        # Sanitize cost matrix to avoid infeasible assignment (NaN/Inf -> large cost)
        dists = np.nan_to_num(dists, nan=1e9, posinf=1e9, neginf=1e9)
        if matcher == 'hungarian':
            kp2s = np.empty_like(kp1)
            row_ind, col_ind = linear_sum_assignment(dists)
            kp2s[row_ind] = kp2[col_ind]
        else:
            # Fallback: greedy nearest neighbor
            idx = np.argmin(dists, axis=1)
            kp2s = kp2[idx]
            
        return np.nan_to_num(kp2s)
    
    def _compute_sparse_flow(self, frame):
        """Compute sparse optical flow using Lucas-Kanade method"""
        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_pts = self._detect_pts(self.prev_gray)
            #self._init_kalmans()
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for i in self.prev_pts.keys():
            # Calculate optical flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                gray,
                                                self.prev_pts[i]['keypoints_1'], None,
                                                **self.flow_params)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(gray,
                                                self.prev_gray,
                                                p1, None,
                                                **self.flow_params)
            d = abs(self.prev_pts[i]['keypoints_1'] - p0r)
            good_mask = d.reshape(-1, 2).max(-1) < 1.0
            val_pts   = (st.flatten() == 1) & good_mask
            self.prev_pts[i]['keypoints_2'][val_pts] = p1[val_pts]
            #good_new = self._update_kalman(good_new, good_old, val_pts)
            
        viz_frame = self._draw_pts_flow(frame, self.prev_pts)
        self.prev_gray = gray
        curr_pts = self._detect_pts(gray)
        
        for i in curr_pts.keys():
            if i in self.prev_pts.keys() and len(self.prev_pts[i]['keypoints_1']) >= curr_pts[i]['keypoints_1'].shape[0]:
                prev_kpts = self.prev_pts[i]['keypoints_2']
                self.prev_pts[i] = curr_pts[i]
                self.prev_pts[i]['keypoints_1'] = self.match_keypoints_by_distance(prev_kpts, curr_pts[i]['keypoints_1'])
                self.prev_pts[i]['keypoints_2'] = self.prev_pts[i]['keypoints_1'].copy()
            else:
                self.prev_pts[i] = curr_pts[i]
        
        for i in list(self.prev_pts.keys()):
            if i not in curr_pts.keys():
                del self.prev_pts[i]

        return viz_frame
    
    def _draw_pts_flow(self, frame, _pts):
        viz_frame = frame.copy()
        for i in _pts.keys():
            good_new = _pts[i]['keypoints_2']
            good_old = _pts[i]['keypoints_1']
            
            if self.mask is None:
                self.mask = np.zeros_like(viz_frame)
            else:
                self.mask = (0.98 * self.mask).astype(np.uint8)

            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                if min(a,b,c,d) < 0:
                    continue
                try:
                    self.mask = cv2.line(self.mask, (a, b), (c, d), self.colors[j % self.n_colors], 2)
                    viz_frame = cv2.circle(viz_frame, (a, b), 3, self.colors[j % self.n_colors], -1)
                except:
                    print(a,b,c,d)
                    
            viz_frame = cv2.add(viz_frame, self.mask)
            cv2.rectangle(viz_frame, (int(_pts[i]['bbox'][0]), int(_pts[i]['bbox'][1])),
                        (int(_pts[i]['bbox'][2]), int(_pts[i]['bbox'][3])), (0, 255, 0), 2)
            cv2.putText(viz_frame, f'ID: {i}',
                        (int(_pts[i]['bbox'][0]), int(_pts[i]['bbox'][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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