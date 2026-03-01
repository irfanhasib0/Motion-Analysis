"""
Optical Flow Tracker with Background Subtraction

This module provides an OpticalFlowTracker class that uses background subtraction (MOG2)
for motion detection combined with optical flow for tracking.

Features:
- Background subtraction using MOG2 for foreground detection
- Keypoint detection and tracking within detected regions
- Multiple detection modes: 'fast', 'accurate'
- Trajectory memory and coreset-based motion analysis

Example Usage:
    # Create tracker
    tracker = OpticalFlowTracker()
    
    # Set detection mode
    tracker.set_detection_method('fast')
    
    # Process frame
    viz_frame, pts, viz_mem1, viz_mem2 = tracker.detect(frame)
    
    # Switch to accurate mode
    tracker.set_detection_method('accurate')

Detection Modes:
- 'fast': Quick detection using only centroids, good for real-time
- 'accurate': Full keypoint detection and optical flow tracking
"""

import cv2
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from trackers.trackers import SimpleTracker #, ByteTracker
from improc.kalman_filter import KalmanPoint, CvKalmanPoint
from improc.memory import FlowMemory, CoresetMemory

'''
# Optional C++ acceleration via pybind11 module
cpp_lib_path = os.path.join('../', "cpp", "build")
if os.path.isdir(cpp_lib_path) and cpp_lib_path not in sys.path:
    sys.path.append(cpp_lib_path)

import motionflow_cpp
MOTIONFLOW_CPP_AVAILABLE = False
'''
class OpticalFlowTracker:
    def __init__(self, 
                 min_traj_len=10,
                 max_traj_len = 100, 
                 max_pid=2500, 
                 coreset_k =2, 
                 matcher_mode="hungarian", 
                 det_method="fast"):
        
        self.prev_gray = None
        self.prev_pts  = None
        self.mask      = None
        self.viz_pos   = None
        self.viz_vel   = None
        self.fg_mask   = 0
        
        self.det_method = det_method  # 'fast', 'accurate'
        self.kpt_det_freq = 1
        self.bg_min_bbox_area = 500
        self.bg_min_pix_thr = 200
        self.bg_mask_dilate_ksize = (3, 3)
        self.bg_hist = 50
        self.mtc_max_cost_thr = 50
        self.kpt_max_kpts = 5
        self.kpt_det_idx = 1  # 0: FAST, 1: SIFT, 2: ORB, 3: GFTT
        self.num_traj_viz = 5  # Number of trajectories to visualize

        colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        self.colors = []
        for _ in range(10):
            self.colors += colors
        self.n_colors = len(self.colors)
        self.count = 0
        
        self.mog_params = dict(history=self.bg_hist,
                               varThreshold=16, # 16
                               detectShadows=False)
        self.kpt_params = dict(threshold=25,
                               nonmaxSuppression=True)
        self.flow_params = dict(winSize=(9, 9), # 15 
                                maxLevel=1, # 2 
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 7, 0.03)) # 10 , 0.03
        
        self.bgsub    = cv2.createBackgroundSubtractorMOG2(**self.mog_params) # 500, 16, True
        self.gftt     = cv2.GFTTDetector_create(maxCorners=100, qualityLevel=0.1, minDistance=10, blockSize=10) # 200, 0.01, 10, 10
        self.orb      = cv2.ORB_create(nfeatures=24)
        self.fast     = cv2.FastFeatureDetector_create(**self.kpt_params)
        self.sift     = cv2.SIFT_create(nfeatures=100,  contrastThreshold=0.04, edgeThreshold=10,  sigma=1.6, nOctaveLayers=3)
        self.tracker  = SimpleTracker(max_disappeared=10, max_distance=400)
        #self.tracker  = ByteTracker()

        self.matcher_mode = matcher_mode  # 'hungarian' (robust) or 'greedy' (faster, C++ accelerated)
        self.memory = FlowMemory(maxpid=max_pid, min_traj_len=min_traj_len, max_traj_len=max_traj_len)
        # Coreset memory (PatchCore-like) for representative motion trajectories
        self.coreset = CoresetMemory(sample_len=max_traj_len, max_items=max_traj_len)
        self.coreset_k = coreset_k

        self.viz_div_h = None

    def restart(self, matcher_mode="hungarian"):
        self.__init__(matcher_mode=matcher_mode)
        print("OpticalFlowTracker restarted.")
    
    def set_detection_method(self, mode):
        self.det_method = mode  # 'fast', 'accurate'
    
    def get_detection_method(self):
        return self.det_method
    
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
    
    def _detect_forground_bboxes(self, gray):
        prev_fg_mask = deepcopy(self.fg_mask)
        self.fg_mask = np.uint8(0.5 * self.fg_mask) + np.uint8(0.5 * self.bgsub.apply(gray))
        if type(prev_fg_mask) == int:
            prev_fg_mask = self.fg_mask.copy()
        diff_mask    = np.abs(self.fg_mask - prev_fg_mask)
        diff_mask = diff_mask[diff_mask > 0]
        if len(diff_mask) > 0:
            self.bg_diff = np.mean(diff_mask)
        else:
            self.bg_diff = 0
        
        if self.bg_mask_dilate_ksize[0] > 1:
            kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.bg_mask_dilate_ksize)
            #self.fg_mask = cv2.morphologyEx(self.fg_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            self.fg_mask = cv2.erode(self.fg_mask, kernel, iterations=1)
            self.fg_mask = cv2.dilate(self.fg_mask, kernel, iterations=1)
        
        self.fg_mask[self.fg_mask > self.bg_min_pix_thr] = 255
        contours, _ = cv2.findContours(self.fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        indx  = np.argsort(areas)[::-1].astype(np.int32)
        areas = areas[indx]
        contours = [contours[i] for i in indx]
        for cnt, area in zip(contours, areas):
            if area > self.bg_min_bbox_area:
                x, y, w, h = cv2.boundingRect(cnt)
                _mask = np.zeros_like(self.fg_mask)
                _mask[y:y+h, x:x+w] = self.fg_mask[y:y+h, x:x+w]
                results.append({'bbox': [x, y, x + w, y + h],
                                'bbox_xywh': [int(x + w/2), int(y + h/2), int(w), int(h)],
                                'centroid': [y + h / 2, x + w / 2],
                                'mask': _mask})
        
        results = self.tracker.update(results)
        for i in results.keys():
            ret, track_window = cv2.CamShift(results[i]['mask'], results[i]['bbox_xywh'], (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
            results[i]['bbox_xywh'] = track_window
            x, y, w, h = track_window
            results[i]['bbox'] = np.array([x, y, x + w, y + h])

        return results
    
    def pts_in_bbox(self, pts, bbox):
        x1, y1, x2, y2 = bbox
        in_bbox = (pts[:, 0, 0] >= x1) & (pts[:, 0, 0] <= x2) & (pts[:, 0, 1] >= y1) & (pts[:, 0, 1] <= y2)
        return in_bbox
    
    def _detect_pts(self, _gray, det_feat_pts=True):
        '''
        Original detection method using background subtraction
        '''
        # Detect corners to track (Python path; with per-bbox C++ accel when available)
        results = self._detect_forground_bboxes(_gray)
        for i in results.keys():
            bbox = results[i]['bbox']
            
            pts = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], dtype=np.float32).reshape(-1,2)
            scores = np.array([1.0], dtype=np.float32).reshape(-1,1)
            if det_feat_pts:
                roi = _gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                kpt_det = [self.fast, self.sift, self.orb, self.gftt][self.kpt_det_idx]
                kps = kpt_det.detect(roi)
                det_pts = cv2.KeyPoint_convert(kps)
                if len(det_pts) > 0:
                    det_pts += np.array([bbox[0], bbox[1]], dtype=np.float32)
                    pts = np.concatenate([pts, det_pts], axis=0)
                    
                    scores  = np.array([1.0]+[kp.response for kp in kps], dtype=np.float32)
                    inds = np.argsort(scores)[::-1]
                    pts  = pts[inds][:self.kpt_max_kpts].reshape(-1,2)
                    scores = scores[inds][:self.kpt_max_kpts].reshape(-1,1)
                
            results[i]['keypoints_1']  = pts
            results[i]['keypoints_2']  = pts.copy()
            results[i]['keypoint_scores'] = scores
        
        return results
    
    def _update_init_pts(self, gray):
        if self.count % self.kpt_det_freq == 0:
            det_pts = self._detect_pts(gray)
            for i in det_pts.keys():
                if i in self.prev_pts.keys() and len(self.prev_pts[i]['keypoints_1']) >= det_pts[i]['keypoints_1'].shape[0]:
                    flow_kpts = self.prev_pts[i]['keypoints_2'].copy()
                    self.prev_pts[i] = det_pts[i]
                    self.prev_pts[i]['keypoints_1'] = self.match_keypoints_by_distance(flow_kpts, det_pts[i]['keypoints_1'], matcher=self.matcher_mode)
                    self.prev_pts[i]['keypoints_2'] = self.prev_pts[i]['keypoints_1'].copy()
                else:
                    self.prev_pts[i] = det_pts[i]
            
            for i in list(self.prev_pts.keys()):
                if i not in det_pts.keys():
                    del self.prev_pts[i]
        else:
            for i in self.prev_pts.keys():
                self.prev_pts[i]['keypoints_1'] = self.prev_pts[i]['keypoints_2'].copy()
                vel = self.prev_pts[i].get('vel', np.array([0.0, 0.0]))
                vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
                self.prev_pts[i]['bbox'] = self.prev_pts[i]['bbox'] + np.array([vel[0], vel[1], vel[0], vel[1]])
                
        for i in self.prev_pts.keys():
            center = np.array([[(self.prev_pts[i]['bbox'][0]+self.prev_pts[i]['bbox'][2])/2,
                                (self.prev_pts[i]['bbox'][1]+self.prev_pts[i]['bbox'][3])/2]], dtype=np.float32)
            self.prev_pts[i]['keypoints_1'][0] = center.reshape(1,2)
            self.prev_pts[i]['keypoints_2'][0] = center.reshape(1,2)

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
        dists = ((_kp1 - _kp2) ** 2).sum(axis=2).astype(np.float32)
        
        # Sanitize cost matrix to avoid infeasible assignment (NaN/Inf -> large cost)
        dists = np.nan_to_num(dists, nan=0.0, posinf=1e3, neginf=0.0)
        
        kp2s = np.empty_like(kp1)
        row_ind, col_ind = linear_sum_assignment(dists)
        # Filter out high-cost matches
        valid_matches = dists[row_ind, col_ind] < self.mtc_max_cost_thr
        
        row_ind = row_ind[valid_matches]
        col_ind = col_ind[valid_matches]
        kp2s[row_ind] = kp2[col_ind]
        
        return kp2s
    
    def _compute_sparse_flow(self, gray):
        for i in self.prev_pts.keys():
            if len(self.prev_pts[i]['keypoints_1']) == 0:
                continue
                
            # Ensure keypoints are in correct format
            kp1 = self.prev_pts[i]['keypoints_1'].astype(np.float32)
            kp1 = kp1.reshape(-1, 1, 2).astype(np.float32)
            
            kp1 = np.nan_to_num(kp1, nan=0.0, posinf=1e3, neginf=0.0)
            kp1[:,0,0] = np.clip(kp1[:,0,0], 0, gray.shape[1]-1)
            kp1[:,0,1] = np.clip(kp1[:,0,1], 0, gray.shape[0]-1)

            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                        gray,
                                                        kp1, None,
                                                        **self.flow_params)
            
            p1 = np.nan_to_num(p1, nan=0.0, posinf=1e3, neginf=0.0)
            p1[:,0,0] = np.clip(p1[:,0,0], 0, gray.shape[1]-1)
            p1[:,0,1] = np.clip(p1[:,0,1], 0, gray.shape[0]-1)

            p0r, _, _ = cv2.calcOpticalFlowPyrLK(gray,
                                                 self.prev_gray,
                                                 p1, None,
                                                 **self.flow_params)
            
            p0r = np.nan_to_num(p0r, nan=0.0, posinf=1e3, neginf=0.0)
            p0r[:,0,0] = np.clip(p0r[:,0,0], 0, gray.shape[1]-1)
            p0r[:,0,1] = np.clip(p0r[:,0,1], 0, gray.shape[0]-1)
            
            d = abs(kp1 - p0r)
            good_mask = d.reshape(-1, 2).max(-1) < 1.0
            val_pts   = (st.flatten() == 1) & good_mask
            bbox = self.prev_pts[i]['bbox']
            new_center = np.array([[[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]]], dtype=np.float32)
            self.prev_pts[i]['keypoints_2'][val_pts] = p1[val_pts].reshape(-1,2)
            
            if ((kp1[0] - new_center)**2).sum() < self.mtc_max_cost_thr:
                self.prev_pts[i]['keypoints_2'][0] = new_center
                self.prev_pts[i]['vel'] = np.mean(p1[val_pts] - kp1[val_pts], axis=0)
                self.prev_pts[i]['keypoints_2'][~val_pts] += (self.prev_pts[i]['vel']).reshape(1,2)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_pts  = self._detect_pts(gray, det_feat_pts= not (self.det_method == 'fast'))
            self.prev_gray = gray
            self.viz_div_h = int(gray.shape[0] // (self.num_traj_viz+1))
            self.viz_div_w = int(gray.shape[1])
            return frame, frame,  {}
        
        if self.det_method == 'fast':
            for i in self.prev_pts.keys():
                self.prev_pts[i]['keypoints_2'] = self.prev_pts[i]['keypoints_1'].copy()
        elif self.det_method == 'accurate':
            self._compute_sparse_flow(gray)
        else:  # balanced
            raise ValueError(f"Unknown detection method: {self.det_method}")

        self.prev_gray = gray

        # Visualization
        self.memory._viz_pos = None
        self.memory._viz_vel = None
        pts = deepcopy(self.prev_pts)
        self.memory.add(pts)
        sorted_ids  = self.memory.get_sorted_traj_ids(curr_pids=pts.keys(), num_of_kpts=self.num_traj_viz)
        
        points_dict = {}
        for _ch,_id in enumerate(sorted_ids):
            _ch += 1
            vel = 0.1*self.memory.get_traj_velocities(traj_id=_id)#[-200:]
            if len(vel) > 0:
                points = np.array([[i + 300, self.viz_div_h*_ch + 3*v] for i, v in enumerate(vel)], dtype=np.int32)
                points = points.reshape((-1, 1, 2))  # Reshape for cv2.polylines
                points = points[-self.viz_div_w:]  # Keep only the last 'viz_div_w' points
                points_dict[_id] = [points, np.mean(vel)]
        
        viz_frame   = self._draw_pts_flow(frame, self.prev_pts)
        plot_array, res = self.plot_velocities(frame, points_dict)
        
        if self.det_method == 'fast':
            self.prev_pts = self._detect_pts(gray, det_feat_pts=False)
            return viz_frame, plot_array, res
        
        self._update_init_pts(gray)
        self.count += 1
        return viz_frame, plot_array, res
    
    def plot_velocities(self, plot_array, points_dict):
        plot_array = 0* plot_array
        _ch = 0
        res = {}
        for _id, (points, mean_vel) in points_dict.items():
            y_pos = self.viz_div_h*_ch
            cv2.line(plot_array, 
                    (0, y_pos), (plot_array.shape[1], y_pos),  
                    (150, 150, 150), 1)
            cv2.polylines(plot_array,
                            pts=[points],
                            isClosed=False,
                            color=self.colors[_ch],
                            thickness=2)
            y_pos = y_pos - self.viz_div_h + 25
            for elem in [f'id: {_id}', f'vel: {mean_vel:.2f} diff: {int(self.bg_diff)}']:
                cv2.putText(plot_array, elem,
                            (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors[_ch], 2)
                y_pos += 25
            res[_id] = {'vel': round(mean_vel,2), 'bg_diff': int(self.bg_diff)}
            _ch += 1

        return plot_array, res
        
    def get_coreset_prototypes(self, k=32):
        """Return selected representative trajectory ids and embeddings."""
        if self.coreset is None:
            return [], np.empty((0, 0), dtype=np.float32)
        return self.coreset.select_kcenter(k)
    
    def get_plot_arrays(self):
        """Return dictionary of plot arrays"""
        return getattr(self, 'plot_arrays', {})
    
    def save_plot_arrays(self, save_path):
        """Save all plot arrays to disk"""
        if hasattr(self, 'plot_arrays'):
            for plot_name, plot_array in self.plot_arrays.items():
                filename = f"{save_path}/{plot_name}_frame_{self.count}.npy"
                np.save(filename, plot_array)
                print(f"Saved plot array: {filename}")
    
    def clear_plot_arrays(self):
        """Clear stored plot arrays to free memory"""
        if hasattr(self, 'plot_arrays'):
            self.plot_arrays.clear()
    
    def _draw_pts_flow(self, frame, _pts):
        viz_frame = frame.copy()
        for i in _pts.keys():
            if self.memory.pid_hist[i]['total_seen'] < 3:
                continue
            good_new = _pts[i]['keypoints_2']
            good_old = _pts[i]['keypoints_1']
            
            if self.mask is None:
                self.mask = np.zeros_like(viz_frame)
            else:
                self.mask = (0.99 * self.mask).astype(np.uint8)

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
                    
            try:
                viz_frame = cv2.add(viz_frame, self.mask)
            except:
                print(viz_frame.shape, self.mask.shape)
                print(viz_frame.dtype, self.mask.dtype)
            cv2.rectangle(viz_frame, (int(_pts[i]['bbox'][0]), int(_pts[i]['bbox'][1])),
                        (int(_pts[i]['bbox'][2]), int(_pts[i]['bbox'][3])), (0, 255, 0), 2)
            cv2.putText(viz_frame, f'ID: {i}',
                        (int(_pts[i]['bbox'][0]), int(_pts[i]['bbox'][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return viz_frame