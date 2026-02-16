import cv2
import numpy as np
from collections import deque, defaultdict
from matplotlib import pyplot as plt

class FlowMemory:
    def __init__(self, maxpid=25, min_traj_len=3, max_traj_len=150, keep_last_seen = 90):
        self.motion_trajs  = defaultdict(lambda: np.ndarray(shape=(0,2), dtype=np.float32))
        self.motion_queue  = defaultdict(lambda: deque(maxlen=max_traj_len))
        self.pid_hist      = {}  # pid -> {'total_seen': int, 'last_seen': int}

        self.traj_len = {}
        self.maxpid = maxpid
        self.maxkpid = 6  # fixed for now
        self.rolling_win = 1
        self.min_traj_len = min_traj_len
        self.keep_last_seen = keep_last_seen

    def fill_neg1(self,a):
        a = a.astype(float)
        x = np.arange(a.size)
        m = a != -1
        if m.sum() == 0:  # no known values
            return a
        a[~m] = np.interp(x[~m], x[m], a[m])
        return a
    
    def fill_array_neg1(self,your_2d_array):
       filled = np.apply_along_axis(self.fill_neg1, axis=1, arr=your_2d_array)
       filled = np.apply_along_axis(self.fill_neg1, axis=0, arr=filled)
       return filled

    def add(self, pts):
        for pid in self.pid_hist.keys():
            self.pid_hist[pid]['last_seen'] += 1

        for pid in pts.keys():
            pid = pid % self.maxpid
            if pid not in self.pid_hist:
                self.pid_hist[pid] = {'last_seen': 0, 'total_seen': 0}
            self.pid_hist[pid]['last_seen']   = 0
            self.pid_hist[pid]['total_seen'] += 1
            for kpid in range(len(pts[pid]['keypoints_1'])):
                curr_pts = pts[pid]['keypoints_1'][kpid].reshape(1,2)
                
                self.motion_queue[f'{pid}|{kpid}'].append(list(curr_pts[0]))
                traj = np.array(self.motion_queue[f'{pid}|{kpid}'])
                
                if traj.shape[0] >= self.min_traj_len:
                    kernel = np.ones(self.rolling_win) / self.rolling_win
                    smoothed_x = np.convolve(traj[:, 0], kernel, mode='same')
                    smoothed_y = np.convolve(traj[:, 1], kernel, mode='same')
                    self.motion_trajs[f'{pid}|{kpid}'] = np.column_stack((smoothed_x, smoothed_y))
                    self.traj_len[f'{pid}|{kpid}'] = traj.shape[0]
    
    def get_sorted_traj_ids(self, curr_pids = [], num_of_kpts = 5):
        _ids = sorted(self.traj_len.keys(), key=lambda x: self.traj_len[x], reverse=True)
        if len(curr_pids) == 0:
            valid_ids = []
            for tid in _ids:
                pid = int(tid.split('|')[0])
                if self.pid_hist[pid]['last_seen'] <= self.keep_last_seen:  # recently seen, keep as candidate
                    valid_ids.append(tid)
            return sorted(valid_ids[:num_of_kpts]) # sort alphabetically for consistency
            
        sorted_ids =  []    
        extra_sorted_ids = []
        for tid in _ids:
            pid = int(tid.split('|')[0])
            if pid in curr_pids:
                sorted_ids.append(tid)
            else:
                if self.pid_hist[pid]['last_seen'] <= self.keep_last_seen:  # recently seen, keep as extra candidate
                    extra_sorted_ids.append(tid)
        if len(sorted_ids) < num_of_kpts:
            sorted_ids += extra_sorted_ids[:num_of_kpts - len(sorted_ids)]
        return sorted(sorted_ids) # sort alphabetically for consistency
    
    def get_traj_velocities(self, traj_id = None, traj = None):
        if traj is None:
            assert traj_id is not None, "Either traj_id or traj must be provided"
            traj = self.motion_trajs.get(traj_id, None)
        velocities = np.sum(np.diff(traj, axis=0)**2, axis=1)**0.5
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        return velocities
                
    def save(self):
        pass

class CoresetMemory:
    """
    Coreset-style memory for motion trajectories.

    Stores fixed-length trajectories for keypoints and selects
    representative prototypes using a greedy k-center algorithm.

    Trajectory embedding: by default, uses normalized deltas of positions
    over the trajectory window to be translation-invariant and roughly
    scale-aware.
    """
    def __init__(self, sample_len=25, normalize=False, max_items=5000):
        self.sample_len = int(sample_len)
        self.normalize = bool(normalize)
        self.max_items = int(max_items)
        # Raw stored trajectories: dict traj_id -> np.ndarray [L, 2]
        self.trajectories = {}
        # Cached embeddings: dict traj_id -> np.ndarray [D]
        self.embeddings = {}
        # Incremental buffers per key: dict traj_key -> deque of points
        self.buffers = defaultdict(lambda: deque(maxlen=self.sample_len))
        self._viz_pos = None
        self._viz_vel = None

    def reset(self):
        self.trajectories.clear()
        self.embeddings.clear()
        self.buffers.clear()

    def add_point(self, traj_key, point):
        """
        Incrementally add a point to the buffer for `traj_key`.
        When the buffer reaches `sample_len`, snapshot it into memory.
        traj_key: any hashable id (e.g., f"obj:{obj_id}|kp:{kp_idx}")
        point: (x, y)
        """
        buf = self.buffers[traj_key]
        buf.append((float(point[0]), float(point[1])))
        if len(buf) >= self.sample_len:
            traj = np.array(buf, dtype=np.float32)
            traj_id = f"{traj_key}|start:{len(self.trajectories)}"
            self._store_trajectory(traj_id, traj)
    
    def add_n_traj(self, trajs):
        for traj_id, traj in trajs.items():
            dist = np.abs(traj[:-1] - traj[1:]).max(-1)
            dist = np.concatenate([dist, np.array([dist[-1]])], axis=0)
            val  = (dist < 10.0) & (traj.min(axis=-1).flatten() > 5.0)
            
            count = 0
            max_count = 0
            best_traj = traj[:0] 
            for _ind  in range(0,len(val)):
                if val[_ind]:
                    count +=1
                else:
                    count =0
                if count > max_count:
                    max_count = count
                    best_traj = traj[_ind - count +1: _ind +1]
            traj = best_traj
            #if traj.shape[0] < 5:
            #    continue
            #import pdb; pdb.set_trace()
            self.add_traj(traj_id, traj)

    def add_traj(self, traj_id, traj_points):
        """
        Directly add a trajectory array [L, 2] with a specific id.
        """
        traj = np.asarray(traj_points, dtype=np.float32)
        if traj.shape[0] < self.sample_len:
            # Pad by repeating last point to reach sample_len
            pad_len = self.sample_len - traj.shape[0]
            if traj.shape[0] == 0:
                return
            pad = np.repeat(traj[-1:], pad_len, axis=0)
            traj = np.vstack([traj, pad])
        elif traj.shape[0] > self.sample_len:
            # Uniformly sample to sample_len
            idx = np.linspace(0, traj.shape[0]-1, self.sample_len).astype(int)
            traj = traj[idx]
        self._store_trajectory(traj_id, traj)

    def _store_trajectory(self, traj_id, traj):
        if len(self.trajectories) >= self.max_items:
            # Drop oldest item to bound memory
            oldest_id = next(iter(self.trajectories))
            self.trajectories.pop(oldest_id, None)
            self.embeddings.pop(oldest_id, None)
        self.trajectories[traj_id] = traj
        self.embeddings[traj_id] = self._embed(traj)

    def _embed(self, traj):
        """
        Convert trajectory [L,2] to embedding [D]. Default: normalized deltas.
        - Subtract first point (translation invariance)
        - Optionally scale by path length (approximate scale invariance)
        - Flatten to 2*L vector
        """
        deltas = traj - traj[0]
        if self.normalize:
            # Scale by total displacement magnitude (avoid divide-by-zero)
            scale = np.linalg.norm(deltas[-1]) + 1e-6
            deltas = deltas / scale
        return deltas.reshape(-1)

    def _draw_gradient_polyline(self, canvas, pts, color_start, color_end, thickness=2, lineType=cv2.LINE_AA):
        """Draw a polyline with a color gradient from start to end.
        - canvas: image to draw on
        - pts: np.ndarray of shape [N, 2] with integer coordinates
        - color_start/color_end: BGR tuples (ints 0-255)
        - thickness: line thickness
        - lineType: cv2 line type (default anti-aliased)
        """
        if pts is None or len(pts) < 2:
            return
        # Ensure integer tuples for cv2.line
        pts = np.asarray(pts, dtype=np.int32)
        n_seg = len(pts) - 1
        if n_seg <= 0:
            return
        c0 = np.array(color_start, dtype=np.float32)
        c1 = np.array(color_end, dtype=np.float32)
        for i in range(n_seg):
            # t ranges 0..1 across segments
            t = 0.0 if n_seg == 1 else (i / float(n_seg - 1))
            c = (1.0 - t) * c0 + t * c1
            color = tuple(int(round(v)) for v in c.tolist())
            p0 = (int(pts[i, 0]), int(pts[i, 1]))
            p1 = (int(pts[i+1, 0]), int(pts[i+1, 1]))
            cv2.line(canvas, p0, p1, color, thickness=thickness, lineType=lineType)

    def select_kcenter(self, k=32):
        """
        Greedy k-center selection over current embeddings.
        Returns (selected_ids, centers_embeddings).
        """
        k = int(max(1, k))
        ids = list(self.embeddings.keys())
        if not ids:
            return [], np.empty((0, 0), dtype=np.float32)
        X = np.stack([self.embeddings[i] for i in ids], axis=0)  # [N, D]
        N = X.shape[0]
        # Start with the point farthest from the mean
        centroid = X.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(X - centroid, axis=1)
        first = int(np.argmax(dists))
        selected_idx = [first]
        # Maintain min distance to current selected set
        min_d = np.linalg.norm(X - X[first], axis=1)
        while len(selected_idx) < min(k, N):
            nxt = int(np.argmax(min_d))
            selected_idx.append(nxt)
            # Update min distances
            min_d = np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1))
        sel_ids = [ids[i] for i in selected_idx]
        centers = X[selected_idx]
        return sel_ids, centers
    
    def clear_viz(self):
        """Clear visualization buffers."""
        self._viz_pos = None
        self._viz_vel = None

    def viz_k_centers(self, _viz_frame, k=2):
        """Return trajectories of selected k-center prototypes."""
        if self._viz_pos is None or self._viz_vel is None:
            self._viz_pos = np.zeros_like(_viz_frame)
            self._viz_vel = np.zeros_like(_viz_frame)
        sel_ids, _ = self.select_kcenter(k)
        H, W = _viz_frame.shape[:2]
        for _id, traj in self.trajectories.items():
        #for _id, traj in self.embeddings.items():#self.trajectories.items():
            traj =  traj.reshape(self.sample_len, 2)
            #traj -= traj.min(axis=0)
            traj *= np.array([1,1])
            color = [0, 255, 0] if _id in sel_ids else [0, 0, 255]

            pts = traj.astype(np.int32)
            
            idx = (pts.min(-1)>5).flatten() & (pts[:,0]<W-5) & (pts[:,1]<H-5)
            pts = pts[idx]
            # Draw polyline and small circles along the trajectory
            if _id in sel_ids:
                # Gradient from red (start) to green (end) in BGR
                self._draw_gradient_polyline(self._viz_pos, pts[1:-1], (255, 0, 0), (0, 255, 0), thickness=4)
            else:
                # Thin gray gradient for non-selected (kept for completeness)
                self._draw_gradient_polyline(self._viz_vel, pts[1:-1], (255, 0, 0), (0, 255, 0), thickness=2)
            '''
            vel = np.diff(pts, axis=0)
            vel = (vel[:,0]**2 + vel[:,1]**2)
            vel = np.sqrt(vel)
            vel = np.concatenate([8*np.arange(vel.shape[0])[:, None], 20*vel[:, None]], axis=1).astype(np.int32)
            cv2.polylines(self._viz_vel, [vel.reshape(-1, 1, 2)], False, color, 1)
            '''
            '''
            vel_pts = pts[:-1] + vel // 2
            for (x, y), (dx, dy) in zip(vel_pts, vel):
                vel = 2*np.sqrt(dx*dx + dy*dy) + 1e-6
                cv2.circle(self._viz_vel, (x, y), int(vel), color, 1)
            '''
            #cv2.polylines(_viz_vel, [pts.reshape(-1, 1, 2)], False, color, 1)
            #for x, y in pts:
            #    cv2.circle(_viz_frame, (int(x), int(y)), 2, color, -1)
        #import pdb; pdb.set_trace()
        return self._viz_pos, self._viz_vel
    
    def get_items(self):
        """Return (ids, trajectories, embeddings) for all stored items."""
        ids = list(self.trajectories.keys())
        trajs = [self.trajectories[i] for i in ids]
        embs = [self.embeddings[i] for i in ids]
        return ids, trajs, embs
    
    def best_matching_trajectory(memory, query, metric="euclidean"):
        """
        Find the best matching stored trajectory in `memory` for the given `query`.
        - memory: CoresetMemory instance
        - query: np.ndarray trajectory [L,2] or embedding vector [D]
        - metric: "euclidean" or "cosine"
        Returns (traj_id, trajectory, distance). If no items, returns (None, None, np.inf).
        """
        ids = list(memory.embeddings.keys())
        if not ids:
            return None, None, float("inf")

        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 2 and q.shape[1] == 2:
            # Normalize length to memory.sample_len
            L = memory.sample_len
            if q.shape[0] < L:
                if q.shape[0] == 0:
                    return None, None, float("inf")
                pad = np.repeat(q[-1:], L - q.shape[0], axis=0)
                q = np.vstack([q, pad])
            elif q.shape[0] > L:
                idx = np.linspace(0, q.shape[0] - 1, L).astype(int)
                q = q[idx]
            q_emb = memory._embed(q)
        else:
            q_emb = q.reshape(-1)

        X = np.stack([memory.embeddings[i] for i in ids], axis=0)  # [N, D]
        if metric == "cosine":
            qn = np.linalg.norm(q_emb) + 1e-8
            Xn = np.linalg.norm(X, axis=1) + 1e-8
            sims = (X @ q_emb) / (Xn * qn)
            best_idx = int(np.argmax(sims))
            dist = 1.0 - float(sims[best_idx])
        else:
            diffs = X - q_emb
            dists = np.linalg.norm(diffs, axis=1)
            best_idx = int(np.argmin(dists))
            dist = float(dists[best_idx])

        best_id = ids[best_idx]
        return best_id, memory.trajectories[best_id], dist