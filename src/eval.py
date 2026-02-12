import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from improc.optical_flow import OpticalFlowTracker
from matplotlib import pyplot as plt
'''
for split in ['training', 'testing']:
    files = sorted(glob.glob(f'/media/irfan/TRANSCEND/action_data/stech/{split}/frames/*'))
    #if split == 'testing':
    #    continue
    for idx, file in enumerate(files):
        print(split, idx, file)
        tracker = OpticalFlowTracker(flow_mode='sparse')
        for path in tqdm(sorted(glob.glob(file + '/*.jpg'))):
            temp = np.zeros((50, 1 + 2 + 6 + 6, 2), dtype=np.float32)
            frame = cv2.imread(path)
            _, pts, _, _ =tracker.detect(frame)
            count = 0
            for pid in sorted(pts.keys()):
                bbox = pts[pid]['bbox']
                kpts1 = pts[pid]['keypoints_1']
                kpts2 = pts[pid]['keypoints_2']
                temp[count, 0:1, :] = [pid, count]
                temp[count, 1:3, :] = np.array(bbox).reshape(-1, 2)
                temp[count, 3:9, :] = kpts1[:,0,:]
                temp[count, 9:15, :] = kpts2[:,0,:]
                count += 1
            temp = temp[:count]
            save_path = path.replace('.jpg', '.npy')
            np.save(save_path, temp)

'''
# Removed SciPy signal usage; autocorrelation uses NumPy-based helpers below

frame = np.zeros((800, 800, 3), dtype=np.uint8)

# --- Autocorrelation-based periodicity helpers ---
def _estimate_dominant_lag_complex(sig_c, lag_min=6, lag_max=60):
    sig = np.asarray(sig_c)
    N = sig.shape[0]
    if N < lag_min + 2:
        return None, 0.0, np.arange(0), np.zeros((0,), dtype=float)
    lag_max = min(lag_max, N - 2)
    lags = np.arange(lag_min, lag_max + 1)
    s = sig - np.mean(sig)
    denom = float(np.sum(np.abs(s) ** 2) + 1e-8)
    r = np.empty_like(lags, dtype=float)
    for i, k in enumerate(lags):
        r[i] = np.abs(np.vdot(s[k:], s[:-k])) / denom
    if r.size == 0:
        return None, 0.0, lags, r
    bi = int(np.argmax(r))
    return int(lags[bi]), float(r[bi]), lags, r

def _windowed_corr_at_lag(sig_c, lag, win_len, corr_thr=0.7):
    s = np.asarray(sig_c)
    N = s.shape[0]
    if lag is None or N < lag + 2 or win_len < 3:
        return np.zeros((0,), dtype=int)
    half = win_len // 2
    idxs = []
    for c in range(lag + half, N - half):
        x1 = s[c - half : c + half]
        x2 = s[c - lag - half : c - lag + half]
        x1c = x1 - np.mean(x1)
        x2c = x2 - np.mean(x2)
        num = np.abs(np.vdot(x1c, x2c))
        den = float(np.linalg.norm(x1c) * np.linalg.norm(x2c) + 1e-8)
        rho = num / den
        if rho >= corr_thr:
            idxs.append(c)
    if not idxs:
        return np.zeros((0,), dtype=int)
    return np.unique(np.array(idxs, dtype=int))

def _segments_from_indices(idxs, min_len=25):
    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0:
        return []
    idxs = np.sort(np.unique(idxs))
    segments = []
    start = idxs[0]
    prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
        else:
            if (prev - start + 1) >= min_len:
                segments.append((start, prev))
            start = i
            prev = i
    if (prev - start + 1) >= min_len:
        segments.append((start, prev))
    return segments
for split in ['testing']:
    files = sorted(glob.glob(f'/media/irfan/TRANSCEND/action_data/stech/{split}/frames/*'))
    for file in files[:10]:
        print(split, file)
        tracker = OpticalFlowTracker(max_traj_len=100, max_pid=25)
        for i, path in tqdm(enumerate(sorted(glob.glob(file + '/*.npy')))):
            pts_arr = np.load(path)
            #frame = cv2.imread(path.replace('.npy', '.jpg'))
            #_, pts1, _, _ =tracker.detect(frame)
            pts = {}
            for row in range(pts_arr.shape[0]):
                pid  = int(pts_arr[row, 0, 0])
                bbox = pts_arr[row, 1:3, :].reshape(1,2,2)
                kpts1 = pts_arr[row, 3:9, :].reshape(6,1,2)
                kpts2 = pts_arr[row, 9:15, :].reshape(6,1,2)
                pts[pid] = {'bbox': bbox, 'keypoints_1': kpts1, 'keypoints_2': kpts2}
            tracker.memory.add(pts)
            '''
            # Snapshot full-length trajectories into coreset when available
            tracker.coreset.add_n_traj(tracker.memory.motion_trajs)
                

            if i > 0 and i % 300 == 0:
                _viz_pos, _viz_vel = tracker.coreset.viz_k_centers(frame, k=2)
                fig, ax = plt.subplots(1,2, figsize=(12,6))
                ax[0].set_title('Position Coreset Visualization')
                ax[0].imshow(_viz_pos)
                ax[1].set_title('Velocity Coreset Visualization')
                ax[1].imshow(_viz_vel)
                tracker.coreset._viz_pos = None
                tracker.coreset._viz_vel = None
                #plt.show()
                if not os.path.exists(f'./figs/stech/{split}/'):
                    os.makedirs(f'./figs/stech/{split}/')
                plt.savefig(f'./figs/stech/{split}/coreset_viz_{file.split("/")[-1]}_{i}_a.png')
                plt.show()
                plt.close()
            '''
        trajs = tracker.memory.motion_trajs
        kernel = np.ones((5,)) / 5.0
        for traj_id, traj in trajs.items():
            if len(traj) < 10:
                continue
            traj_xy = np.array(traj)
            traj_xm = np.convolve(traj_xy[:,0], kernel, mode='valid')
            traj_ym = np.convolve(traj_xy[:,1], kernel, mode='valid')
            traj_xym = np.stack([traj_xm, traj_ym], axis=1)
            # Build complex trajectory for autocorrelation (mean removed)
            traj_ab = traj_xm + 1j * traj_ym
            traj_ab = traj_ab - np.mean(traj_ab)

            # Autocorrelation-based periodic pattern detection (dense segments > 25)
            lag_min, lag_max = 6, 60
            best_lag, best_score, _, _ = _estimate_dominant_lag_complex(traj_ab, lag_min=lag_min, lag_max=lag_max)
            win_len = max(25, int(best_lag) if best_lag is not None else 25)
            corr_thr = 0.7
            ac_idx = _windowed_corr_at_lag(traj_ab, best_lag, win_len, corr_thr=corr_thr)
            segments = _segments_from_indices(ac_idx, min_len=25)

            # 5. Visualization: base trajectory and dense autocorr segments
            plt.figure(figsize=(6, 6))
            plt.plot(traj_xym[:, 0], traj_xym[:, 1], '-', color='k', alpha=0.5, linewidth=1, label='Smoothed')
            for (s, e) in segments:
                seg = traj_xym[s:e+1, :]
                plt.plot(seg[:, 0], seg[:, 1], '-', color='g', linewidth=2)
            title = "Trajectory with dense autocorrelated segments (len≥25)"
            if best_lag is not None:
                title += f" | lag={best_lag}, r~{best_score:.2f}"
            plt.title(title)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            #plt.figure(figsize=(6,6))
            #plt.plot(traj_xy[:,0], traj_xy[:,1], '-o', markersize=2)
            #plt.plot(traj_xc, traj_yc, '-x', markersize=2)
            #plt.show()
        import pdb; pdb.set_trace()
