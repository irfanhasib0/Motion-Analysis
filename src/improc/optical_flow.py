import cv2
import numpy as np
import threading
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from trackers.trackers import SimpleTracker
from collections import deque
from improc.kalman_filter import KalmanPoint, CvKalmanPoint
from improc.memory import FlowMemory, CoresetMemory
import cProfile, pstats, io
import os, sys

# Optional C++ acceleration via pybind11 module
cpp_lib_path = os.path.join('../', "cpp", "build")
if os.path.isdir(cpp_lib_path) and cpp_lib_path not in sys.path:
    sys.path.append(cpp_lib_path)

import motionflow_cpp
MOTIONFLOW_CPP_AVAILABLE = False

class OpticalFlowTracker:
    def __init__(self, max_traj_len = 100, max_pid=10, coreset_k =2, matcher_mode="hungarian"):
        self.prev_gray = None
        self.prev_pts  = None
        self.prev_des  = None
        self.mask      = None
        self.viz_pos   = None
        self.viz_vel   = None

        colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        self.colors = []
        for _ in range(10):
            self.colors += colors
        self.n_colors = len(self.colors)
        self.count = 0
        self.mog_params = dict(history=100,
                               varThreshold=16,
                               detectShadows=True)
        self.kpt_params = dict(threshold=25,
                               nonmaxSuppression=True)
        self.flow_params = dict(winSize=(9, 9), # 15 
                                maxLevel=1, # 2 
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 7, 0.03)) # 10 , 0.03
        
        self.bgsub    = cv2.createBackgroundSubtractorMOG2(**self.mog_params) # 500, 16, True
        #self.gftt    = cv2.GFTTDetector_create(maxCorners=100, qualityLevel=0.1, minDistance=10, blockSize=10) # 200, 0.01, 10, 10
        #self.gftt    = cv2.ORB_create(nfeatures=24)
        self.gftt     = cv2.FastFeatureDetector_create(**self.kpt_params)
        #self.gftt    = cv2.SIFT_create(nfeatures=100,  contrastThreshold=0.04, edgeThreshold=10,  sigma=1.6, nOctaveLayers=3)
        self.tracker  = SimpleTracker(max_disappeared=30, max_distance=100)
        
        self.kalmans = None
        self.matcher_mode = matcher_mode  # 'hungarian' (robust) or 'greedy' (faster, C++ accelerated)
        self.keypoint_queue = deque(maxlen=5)
        self.memory = FlowMemory(maxlen=max_traj_len, maxpid=max_pid)
        # Coreset memory (PatchCore-like) for representative motion trajectories
        self.coreset = CoresetMemory(sample_len=self.memory.maxlen, max_items=max_traj_len)
        self.coreset_k = coreset_k
    
    def restart(self, matcher_mode="hungarian"):
        self.__init__(matcher_mode=matcher_mode)
        print("OpticalFlowTracker restarted.")

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
        if MOTIONFLOW_CPP_AVAILABLE:
            res_list = motionflow_cpp.detect_foreground_bboxes(gray.astype(np.uint8), 500, 2, 2, 3)
            # Convert to Python-native structures if needed
            py_results = []
            for item in res_list:
                bbox = list(item['bbox']) if isinstance(item['bbox'], tuple) else item['bbox']
                centroid = list(item['centroid']) if isinstance(item['centroid'], tuple) else item['centroid']
                _mask = item['mask']
                py_results.append({'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                                    'centroid': [float(centroid[0]), float(centroid[1])],
                                    'mask': _mask})
            return py_results
        
        # Fallback to Python implementation
        fg_mask = self.bgsub.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                _mask = np.zeros_like(fg_mask)
                _mask[y:y+h, x:x+w] = fg_mask[y:y+h, x:x+w]
                results.append({'bbox': [x, y, x + w, y + h],
                                'centroid': [y + h / 2, x + w / 2],
                                'mask': _mask})
        return results
    
    def _detect_pts(self, _gray):
        '''
        if getattr(self, 'prof', None) is None:
            self.prof = cProfile.Profile()
            self.prof.enable()
        '''
        # Fast path: full C++ pipeline (bboxes + per-bbox points)
        if MOTIONFLOW_CPP_AVAILABLE:
            res_list = motionflow_cpp.detect_pts(_gray.astype(np.uint8), 3, 2, 25, 500, 2, 2, 3)
            results = []
            for item in res_list:
                bbox = list(item['bbox']) if isinstance(item['bbox'], tuple) else item['bbox']
                centroid = list(item['centroid']) if isinstance(item['centroid'], tuple) else item['centroid']
                kp1 = np.array(item['keypoints_1'], dtype=np.float32)
                kp2 = np.array(item['keypoints_2'], dtype=np.float32)
                scores = np.array(item['keypoint_scores'], dtype=np.float32)
                results.append({'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                                'centroid': [float(centroid[0]), float(centroid[1])],
                                'mask': item['mask'],
                                'keypoints_1': kp1,
                                'keypoints_2': kp2,
                                'keypoint_scores': scores})
            return self.tracker.update(results)
            
        # Detect corners to track (Python path; with per-bbox C++ accel when available)
        results = self._detect_forground_bboxes(_gray)
        del_idx = []
        for i in range(len(results)):
            bbox = results[i]['bbox']
            grid_n, grid_m = 3, 2
            pts = None
            scores = None
            
            if pts is None:
                # Fallback to Python FAST + grid selection
                roi = _gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                kps = self.gftt.detect(roi)
                pts0 = cv2.KeyPoint_convert(kps)
                if len(pts0) == 0:
                    del_idx.append(i)
                    continue
                pts0 += np.array([bbox[0], bbox[1]], dtype=np.float32)
                indices = np.lexsort((pts0[:, 0], pts0[:, 1]))
                pts0 = pts0[indices]
                sc0 = np.array([kps[j].response for j in indices], dtype=np.float32)
                w = (bbox[2] - bbox[0]) / grid_m
                h = (bbox[3] - bbox[1]) / grid_n
                filtered_pts = []
                filtered_scores = []
                rows = ((pts0[:, 1] - bbox[1]) / h).astype(np.int32)
                cols = ((pts0[:, 0] - bbox[0]) / w).astype(np.int32)
                valid = (rows >= 0) & (rows < grid_n) & (cols >= 0) & (cols < grid_m)
                best_idx_per_cell = np.full(grid_n * grid_m, -1, dtype=np.int32)
                best_score_per_cell = np.full(grid_n * grid_m, -np.inf, dtype=np.float32)
                if np.any(valid):
                    rows_v = rows[valid]
                    cols_v = cols[valid]
                    pidx_v = np.nonzero(valid)[0]
                    sc_v = sc0[valid]
                    cell_ids = rows_v * grid_m + cols_v
                    for k in range(cell_ids.shape[0]):
                        cid = int(cell_ids[k])
                        s = float(sc_v[k])
                        if s > best_score_per_cell[cid]:
                            best_score_per_cell[cid] = s
                            best_idx_per_cell[cid] = pidx_v[k]
                for row in range(grid_n):
                    for col in range(grid_m):
                        cid = row * grid_m + col
                        if best_idx_per_cell[cid] != -1:
                            idx = int(best_idx_per_cell[cid])
                            filtered_pts.append(pts0[idx])
                            filtered_scores.append(sc0[idx])
                        elif self.prev_pts is not None and i in self.prev_pts.keys() and \
                             self.prev_pts[i]['keypoints_2'].shape[0] > cid:
                            pt = self.prev_pts[i]['keypoints_2'][cid][0].astype(np.float32)
                            filtered_pts.append(pt)
                            filtered_scores.append(0.0)
                        else:
                            cx = bbox[0] + (col + 0.5) * w
                            cy = bbox[1] + (row + 0.5) * h
                            filtered_pts.append(np.array([cx, cy], dtype=np.float32))
                            filtered_scores.append(0.0)
                pts = np.array(filtered_pts, dtype=np.float32).reshape(-1,1,2)
                scores = np.array(filtered_scores, dtype=np.float32)

            results[i]['keypoints_1']  = pts
            results[i]['keypoints_2']  = pts.copy()
            results[i]['keypoint_scores'] = scores.reshape(-1, 1)

        del_idx = sorted(del_idx, reverse=True)
        for i in del_idx:
            del results[i]
        results = self.tracker.update(results)
        '''
        self.count = getattr(self, 'count', 0) + 1
        if self.count % 50 == 0:
            s = io.StringIO()
            ps = pstats.Stats(self.prof, stream=s).sort_stats('cumtime').print_stats(20)
            ps.print_stats('cv2')
            print(s.getvalue())
            self.prof.disable()
            self.prof = None
            import pdb; pdb.set_trace()
        '''
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
            if MOTIONFLOW_CPP_AVAILABLE:
                kp2s = motionflow_cpp.match_greedy(kp1.reshape(-1,1,2).astype(np.float32),
                                                       kp2.reshape(-1,1,2).astype(np.float32))
            else:
                idx = np.argmin(dists, axis=1)
                kp2s = kp2[idx]
        return np.nan_to_num(kp2s)
    
    def _compute_sparse_flow(self, gray):
        for i in self.prev_pts.keys():
            if len(self.prev_pts[i]['keypoints_1']) == 0:
                continue
                
            try:
                # Ensure keypoints are in correct format
                kp1 = self.prev_pts[i]['keypoints_1'].astype(np.float32)
                if len(kp1.shape) != 3 or kp1.shape[1] != 1 or kp1.shape[2] != 2:
                    kp1 = kp1.reshape(-1, 1, 2).astype(np.float32)
                    
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                            gray,
                                                            kp1, None,
                                                            **self.flow_params)
                p0r, _, _ = cv2.calcOpticalFlowPyrLK(gray,
                                                        self.prev_gray,
                                                        p1, None,
                                                        **self.flow_params)
                d = abs(kp1 - p0r)
                good_mask = d.reshape(-1, 2).max(-1) < 1.0
                val_pts   = (st.flatten() == 1) & good_mask
                self.prev_pts[i]['keypoints_2'][val_pts] = p1[val_pts]
            except cv2.error as e:
                print(f"OpenCV error in optical flow for object {i}: {e}")
                # Reset keypoints for this object on error
                self.prev_pts[i]['keypoints_2'] = self.prev_pts[i]['keypoints_1'].copy()
                continue
            except Exception as e:
                print(f"Unexpected error in optical flow for object {i}: {e}")
                self.prev_pts[i]['keypoints_2'] = self.prev_pts[i]['keypoints_1'].copy()
                continue
    
    def _update_init_pts(self, gray):
        if self.count % 5 == 0:
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
                vel = np.mean(self.prev_pts[i]['keypoints_2'] - self.prev_pts[i]['keypoints_1'], axis=0)[0]
                self.prev_pts[i]['bbox'] = self.prev_pts[i]['bbox'] + np.array([vel[0], vel[1], vel[0], vel[1]])

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_pts  = self._detect_pts(gray)
            self.prev_gray = gray
            return frame, {}, frame, frame
        
        self._compute_sparse_flow(gray)
        self.prev_gray = gray

        # Visualization
        viz_frame = self._draw_pts_flow(frame, self.prev_pts)
        self.memory._viz_pos = None
        self.memory._viz_vel = None
        pts = deepcopy(self.prev_pts)
        # Accumulate motion trajectories in simple memory
        self.memory.add(pts)#self.prev_pts)

        # Snapshot full-length trajectories into coreset when available
        self.coreset.add_n_traj(self.memory.motion_trajs)
            
        self._update_init_pts(gray)
        self.count += 1
        if self.count % 100 == 0:
            self.viz_pos, self.viz_vel = self.coreset.viz_k_centers(self.viz_pos, k=self.coreset_k)
        else:
            if self.viz_pos is None or self.viz_vel is None:
                self.viz_pos, self.viz_vel = np.zeros_like(frame), np.zeros_like(frame)
        viz_mem1 = self.viz_pos.copy()
        viz_mem2 = self.viz_vel.copy()
        return viz_frame, pts, viz_mem1, viz_mem2

    def get_coreset_prototypes(self, k=32):
        """Return selected representative trajectory ids and embeddings."""
        if self.coreset is None:
            return [], np.empty((0, 0), dtype=np.float32)
        return self.coreset.select_kcenter(k)
    
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