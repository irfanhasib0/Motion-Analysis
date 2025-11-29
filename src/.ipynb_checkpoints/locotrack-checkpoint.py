import os
import cv2
import uuid
import time
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt

# --- Repo imports (same as the official demo) ---
import sys
import sys
sys.path.append("locotrack/locotrack_pytorch")
from models.locotrack_model import load_model, FeatureGrids  # repo API

# ===================== User knobs =====================
SOURCE = "0"              # "0" for webcam, or "/path/to/video.mp4"
MODEL_SIZE = "small"      # "small" or "base"
CVIDEO_PATH = "/media/irfan/TRANSCEND/camera_data/stech/shanghaitech/training/videos/01_001.avi"               # <- change me
CKPT = f"./locotrack/weights/locotrack_small.ckpt"

PREVIEW_WIDTH = 768       # draw/output width
INPUT_HW = (256, 256)     # model input (H, W); keep default
BUFFER_T = 32             # window size (temporal context)
UPDATE_STRIDE = 1         # run model every N frames (1 = every frame)
DRAW_TRAIL = True
SAVE_MP4 = False          # set True to record an mp4 in ./outputs

# Hard-coded points (in PREVIEW pixels) to start tracking at FIRST frame
# e.g., [(x,y), (x,y), ...]
HARDCODED_POINTS = [(320, 180), (420, 200), (250, 300)]
# ======================================================

def _make_colors(n):
    # fixed color wheel (BGR)
    import math
    cols = []
    for i in range(n):
        a = i / max(1, n)
        b = int(127.5 * (1 + math.sin(2*np.pi*a)))
        g = int(127.5 * (1 + math.sin(2*np.pi*(a + 1/3))))
        r = int(127.5 * (1 + math.sin(2*np.pi*(a + 2/3))))
        cols.append((b, g, r))
    return cols

class OnlineLocoTracker:
    def __init__(self, model, device="cuda", dtype=torch.bfloat16,
                 buffer_T=32, input_hw=(256,256), preview_wh=(None, None)):
        self.model = model.eval()
        self.device = device
        self.dtype = dtype
        self.buffer_T = buffer_T
        self.in_h, self.in_w = input_hw
        self.prev_w, self.prev_h = preview_wh  # prev_h resolved after first frame

        # rolling buffers
        self.buf_input = []   # list of (H,W,3) uint8 resized to input
        self.buf_preview = [] # list of (H,W,3) uint8 resized to PREVIEW
        self.frame_count = 0

        # per-point state
        self.points = []      # list of dicts: {'anchor_t': int, 'x_m': float, 'y_m': float, 'color': (b,g,r)}
        self.history = []     # list of list of (x_prev, y_prev, visible_bool)

        self.colors = []

    def _resize_frames(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        if self.prev_h is None:
            self.prev_h, self.prev_w = H , W
        preview_frame = cv2.resize(frame_bgr, (self.prev_w, self.prev_h), interpolation=cv2.INTER_AREA)
        input_frame   = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        return preview_frame, input_frame

    def add_points_at_current(self, pts_preview):
        # called at t = last frame in buffer
        if not pts_preview:
            return
        if not self.colors:
            self.colors = _make_colors(max(3, len(pts_preview)))

        sx = self.in_w / self.prev_w
        sy = self.in_h / self.prev_h
        t_anchor = len(self.buf_input) - 1  # last frame in buffer

        for i, (x, y) in enumerate(pts_preview):
            self.points.append({
                'anchor_t': t_anchor,
                'x_m': float(x * sx),
                'y_m': float(y * sy),
                'color': self.colors[(len(self.points)+i) % len(self.colors)],
            })
            self.history.append([])

    def _maybe_pop_front(self):
        popped = False
        if len(self.buf_input) > self.buffer_T:
            self.buf_input.pop(0)
            self.buf_preview.pop(0)
            popped = True
            # shift anchor_t for all points
            for p in self.points:
                p['anchor_t'] -= 1
        return popped

    @torch.inference_mode()
    def _run_model(self):
        # Build FeatureGrids as in the demo (normalize to [-1,1], last-level grids)
        vid_np = np.stack(self.buf_input, axis=0)  # (T,H,W,3) uint8
        x = (vid_np / 255.0) * 2 - 1
        x = torch.tensor(x).unsqueeze(0).to(self.device, dtype=self.dtype)  # (1,T,H,W,3)

        # no autocast on CPU
        if self.device == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            feats = self.model.get_feature_grids(x)
        feats = FeatureGrids(
            lowres=(feats.lowres[-1].to(self.device, self.dtype),),
            hires=(feats.hires[-1].to(self.device, self.dtype),),
            highest=(feats.highest[-1].to(self.device, self.dtype),),
            resolutions=(feats.resolutions[-1],),
        )

        # Build query tensor from anchors (flip xyt->tyx like the demo)
        if not self.points:
            return None

        q = []
        for p in self.points:
            q.append([p['x_m'], p['y_m'], p['anchor_t']])  # x,y,t
        q = torch.tensor(q, dtype=torch.float32)[None].flip(-1).to(self.device, self.dtype)  # (1,N,3) tyx

        with autocast_ctx:
            out = self.model(x, q, feature_grids=feats)

        # tracks: (N, T, 2) in model coordinates
        tracks = out['tracks'][0].float().cpu().numpy()
        # occlusion -> visible mask; combine with expected_dist like demo
        occ = torch.sigmoid(out['occlusion'])
        if 'expected_dist' in out:
            ed = torch.sigmoid(out['expected_dist'])
            occ = 1 - (1 - occ) * (1 - ed)
        visible = (occ[0].cpu().numpy() <= 0.5)  # True = visible

        return tracks, visible

    def step(self, frame_bgr, run_model=True):
        # 1) add resized frames to buffers
        prev, inp = self._resize_frames(frame_bgr)
        self.buf_preview.append(prev)
        self.buf_input.append(inp)
        self.frame_count += 1

        # 2) drop oldest and adjust anchors
        dropped = self._maybe_pop_front()

        # 3) if any anchor got dropped (anchor_t < 0), re-anchor that point to last known pred
        #    (we do this AFTER a model run when we have a last prediction; on the very first drop
        #     we just clamp to t = last frame and keep xy as-is until next run)
        for p in self.points:
            if p['anchor_t'] < 0:
                p['anchor_t'] = len(self.buf_input) - 1  # re-anchor at current last frame

        # 4) run model (optionally)
        tracks, visible = (None, None)
        if run_model and self.points:
            res = self._run_model()
            if res is not None:
                tracks, visible = res
                # update each point’s last position (model coords) and append to history
                T = tracks.shape[1]
                for i, p in enumerate(self.points):
                    x_m, y_m = tracks[i, T-1]
                    p['x_m'], p['y_m'] = float(x_m), float(y_m)

                    # to preview coords
                    x_p = x_m * (self.prev_w / self.in_w)
                    y_p = y_m * (self.prev_h / self.in_h)
                    vis = bool(visible[i, T-1])
                    self.history[i].append((float(x_p), float(y_p), vis))

                    # if anchor_t got negative earlier, ensure it sits at last frame after update
                    if p['anchor_t'] < 0:
                        p['anchor_t'] = T - 1

        # 5) draw current frame
        canvas = prev.copy()
        for i, p in enumerate(self.points):
            # Draw last observation if available, else draw anchor
            if self.history[i]:
                x_p, y_p, vis = self.history[i][-1]
                if vis:
                    cv2.circle(canvas, (int(round(x_p)), int(round(y_p))), 4, p['color'], -1, lineType=cv2.LINE_AA)
            else:
                # draw the anchor location
                x_p = p['x_m'] * (self.prev_w / self.in_w)
                y_p = p['y_m'] * (self.prev_h / self.in_h)
                cv2.circle(canvas, (int(round(x_p)), int(round(y_p))), 4, p['color'], -1, lineType=cv2.LINE_AA)

            # Optional trail
            if DRAW_TRAIL and len(self.history[i]) >= 2:
                pts = [(int(round(x)), int(round(y))) for (x, y, v) in self.history[i] if v]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, a, b, p['color'], 2, lineType=cv2.LINE_AA)

        return canvas

def main():
    assert Path(CKPT).exists(), f"Checkpoint not found at {CKPT}"
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model = load_model(CKPT, model_size=MODEL_SIZE).to(device)

    # open source
    if CVIDEO_PATH == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(CVIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {SOURCE}")

    # Resolve preview height from first frame
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No frames available from source.")
    tracker = OnlineLocoTracker(model, device=device, dtype=dtype,
                                buffer_T=BUFFER_T, input_hw=INPUT_HW,
                                preview_wh=(PREVIEW_WIDTH, None))

    # push first frame, then add points at t=0
    canvas = tracker.step(frame, run_model=False)
    tracker.add_points_at_current(HARDCODED_POINTS)

    # writer
    writer = None
    if SAVE_MP4:
        Path("outputs").mkdir(parents=True, exist_ok=True)
        out_path = f"outputs/locotrack_stream_{uuid.uuid4().hex}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30, (canvas.shape[1], canvas.shape[0]))

    # Display loop
    frame_idx = 1
    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_now = (frame_idx % UPDATE_STRIDE == 0)
        canvas = tracker.step(frame, run_model=run_now)

        # FPS text
        now = time.time()
        fps = 1.0 / max(1e-6, now - last_t)
        last_t = now
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 200, 40), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(canvas)

        #cv2.imshow("LocoTrack (Online)", canvas)
        #key = cv2.waitKey(1) & 0xFF
        #if key == 27 or key == ord('q'):
        #    break
        plt.imshow(canvas)
        plt.show()
        frame_idx += 1

    #cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {out_path}")
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
