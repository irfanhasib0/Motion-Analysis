"""
Single-file deployer for TAPNext (PyTorch) based on DeepMind's Colab
`colabs/torch_tapnext_demo.ipynb`.

Goal
-----
Import one class and call a couple of functions to:
  1) Fetch the pretrained checkpoint if missing
  2) Build the TAPNext (PyTorch) model and load weights
  3) Run point tracking on a video or a folder of frames
  4) Save tracks (JSON/NPZ) and an overlaid visualization video

Usage
-----
from tapnext_deploy import TapNextDeployer

runner = TapNextDeployer(device="cuda", image_size=(256, 256))
runner.ensure_assets()   # downloads ckpt
runner.load_model()      # builds and restores model

# Example 1: track a grid of points across the whole video
tracks = runner.track_video(
    video_path="/path/to/video.mp4",
    query_mode="grid",            # "grid" | "manual" | "file"
    grid_stride=32,                # pixels between grid points
    out_video="tracks.mp4",
    out_points_json="tracks.json",
)

# Example 2: track user-specified points defined at given frames
# queries is a list of dicts: {"t": frame_idx, "x": float, "y": float}
queries = [{"t":0, "x": 120.0, "y": 200.0}, {"t":0, "x": 300.0, "y": 150.0}]
tracks = runner.track_video(
    video_path="/path/to/video.mp4",
    query_mode="manual",
    manual_queries=queries,
)

Notes
-----
* This file targets the official `tapnet` package layout. If import paths
  change upstream, the fallback import logic will try common alternatives,
  and raise a clear error if none are found.
* Visualization uses OpenCV; install with `pip install opencv-python`.
* Checkpoint path defaults to a BootsTAPNext JAX ckpt restored into the
  PyTorch model via `restore_model_from_jax_checkpoint()` per the Colab.

"""
from __future__ import annotations

import dataclasses
import json
import os
import cv2
import sys
sys.path.append('tapnet')
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------
# TapNet / TAPNext imports
# ---------------------------
# The Colab snippets show these symbols:
#   - TAPNext(image_size=(256, 256))
#   - restore_model_from_jax_checkpoint(model, ckpt_path)
# We attempt multiple import paths to be robust to small repo refactors.

_restore_jax_ckpt = None

_import_errors: List[str] = []

from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty  # type: ignore

# Common places TAPNext might live
from tapnet.tapnext.tapnext_torch import TAPNext

def _torch_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _resize_frames(frames: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Resize frames (T, H, W, 3) to (T, h, w, 3) using area interpolation."""
    import cv2
    T, H, W, C = frames.shape
    h, w = image_size
    out = np.empty((T, h, w, C), dtype=frames.dtype)
    for i in range(T):
        out[i] = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_AREA)
    return out

def _read_video(path: str) -> np.ndarray:
    """Read a video with OpenCV as uint8 array of shape (T, H, W, 3) in RGB."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {path}")
    return np.stack(frames, axis=0)


def _to_torch_video(frames_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 -> float32 tensor in [0,1], shape (1, T, 3, H, W)."""
    assert frames_uint8.ndim == 4 and frames_uint8.shape[-1] == 3
    vid = torch.from_numpy(frames_uint8).to(device)  # (T, H, W, 3), uint8
    vid = vid.contiguous()#.permute(0, 3, 1, 2).contiguous()       # (T, 3, H, W)
    vid = vid.float() / 255.0
    vid = vid.unsqueeze(0)                           # (1, T, 3, H, W)
    return 2 * (vid - 0.5)


def _save_video_overlay(
    out_path: str,
    frames_rgb: np.ndarray,  # (T, H, W, 3) uint8
    tracks: Dict[str, Any],
    radius: int = 3,
    thickness: int = 2,
    fps: Optional[float] = None,
) -> None:
    """Render tracks onto frames and save as mp4 using OpenCV.

    `tracks` format expected from `TapNextDeployer.track_video`, namely:
      {
        "points": [ {"id": int, "t0": int, "xy": [[x_t, y_t, vis_t], ...]} , ... ]
      }
    where each point has one trajectory value per frame (vis_t is 0/1).
    """
    import cv2

    T, H, W, _ = frames_rgb.shape
    if fps is None:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _ensure_dir(Path(out_path))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    COLORS = [(237, 102, 99), (255, 158, 74), (103, 191, 92), (130, 176, 210),
              (144, 103, 167), (171, 104, 87), (114, 158, 206), (82, 84, 163)]

    for t in range(T):
        frame = frames_rgb[t].copy()  # RGB uint8
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for i, pt in enumerate(tracks.get("points", [])):
            xy_list = pt["xy"]
            if t < len(xy_list):
                x, y, vis = xy_list[t]
                color = COLORS[i % len(COLORS)]
                if vis > 0.5:
                    cv2.circle(bgr, (int(round(x)), int(round(y))), radius, color, -1)
        writer.write(bgr)

    writer.release()


# ---------------------------
# Public API
# ---------------------------

@dataclasses.dataclass
class TapNextDeployer:
    device: str | torch.device = "cuda"
    image_size: Tuple[int, int] = (256, 256)  # (h, w)
    #ckpt_url: str = "https://storage.googleapis.com/dm-tapnet/bootstapnext_ckpt.npz"
    #ckpt_url: str = "https://storage.googleapis.com/dm-tapnet/tapnext/tapnext_ckpt.npz"
    ckpt_url: str = "https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz"
    ckpt_path: str = "checkpoints/bootstapnext_ckpt.npz" #bootstapnext_ckpt.npz

    _model: Optional[nn.Module] = dataclasses.field(default=None, init=False)
    _device: torch.device = dataclasses.field(default=torch.device("cpu"), init=False)

    def ensure_assets(self) -> None:
        """Download the checkpoint if not present (idempotent)."""
        ckpt = Path(self.ckpt_path)
        if ckpt.exists():
            return
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading TAPNext checkpoint to {ckpt} ...")
        urllib.request.urlretrieve(self.ckpt_url, str(ckpt))
        print("Done.")

    def load_model(self) -> nn.Module:
        """Create TAPNext(image_size) and restore from JAX checkpoint into PyTorch.
        Returns the loaded model in eval() mode on the selected device.
        """
        self._device = _torch_device(self.device)
        # Instantiate TAPNext
        model = TAPNext(image_size=self.image_size)  # type: ignore
        model = restore_model_from_jax_checkpoint(model, self.ckpt_path)
        model.to(self._device).eval()
        self._model = model
        return model

    # ---------- Query preparation helpers ----------
    #@staticmethod
    def make_grid_queries(self,
        T: int,
        H: int,
        W: int,
        stride: int = 32,
        t0: int = 0,
    ) -> List[Dict[str, float]]:
        """Generate a dense grid of queries at frame t0.
        Returns a list of dicts: {"t": t0, "x": x, "y": y}
        """
        xs = list(range(stride // 2, W, stride))
        ys = list(range(stride // 2, H, stride))
        queries = []
        for y in ys:
            for x in xs:
                queries.append([float(t0), float(x), float(y)])
        return torch.tensor(queries)[None]

    def track_init(self, frame0_rgb: np.ndarray, pts_r):
        # Ensure input is uint8 RGB
        if frame0_rgb.dtype != np.uint8:
            frame0_rgb = frame0_rgb.astype(np.uint8)
        
        H0, W0 = frame0_rgb.shape[:2]
        
        h, w = self.image_size
        f0 = cv2.resize(frame0_rgb, (w, h), interpolation=cv2.INTER_AREA)
        sx, sy = w / float(W0), h / float(H0)
        pts_r[:, 0] *= sx
        pts_r[:, 1] *= sy
        
        # Pack into (1, T=2, 3, H, W) float32 [0,1]
        clip = f0[None]
            
        vid = _to_torch_video(clip, self._device)
        query_pts = torch.zeros((1,len(pts_r),3), dtype = torch.float32)
        query_pts[0,:,1:] = torch.tensor(pts_r)
        
        model = self._model
        model.to(self._device)
        with torch.inference_mode():
            pred = None
            #pred = model(vid.to(self._device), query_pts.to(self._device))  # preferred signature
            #from tapnet.tapnext.tapnext_torch_utils import forward_pytorch as _fw  # type: ignore
            #pred = _fw(model, vid, q_t, q_y, q_x)
            #pred = model(vid, {"t": q_t, "x": q_x, "y": q_y})
            
        return
        
            
    def track_step(
        self,
        frame1_rgb: np.ndarray,
        points_xy: np.ndarray | None = None,
        resize_to_model: bool = True,
        mode: str = 'track'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track points from frame0 -> frame1 using TAPNext with a 2-frame clip.

        Parameters
        ----------
        frame0_rgb, frame1_rgb: HxWx3 uint8 RGB arrays (same size).
        points_xy: iterable of (x, y) in pixel coords of *input frames*.
        resize_to_model: if True (default), frames are resized to `self.image_size`
            before inference and point coordinates are scaled accordingly.

        Returns
        -------
        next_points_xy: (N, 2) array of predicted positions in *input frame1* coords.
        visibility: (N,) array in [0,1] for each point at t=1.
        """
        assert self._model is not None, "Call load_model() first."

        # Ensure input is uint8 RGB
        if frame1_rgb.dtype != np.uint8:
            frame1_rgb = frame1_rgb.astype(np.uint8)

        H0, W0 = frame1_rgb.shape[:2]
        
        h, w = self.image_size
        f1 = cv2.resize(frame1_rgb, (w, h), interpolation=cv2.INTER_AREA)
        sw, sh = w / float(W0), h / float(H0)
        
        
        # Pack into (1, T=2, 3, H, W) float32 [0,1]
        clip = f1[None]  # (2, H, W, 3)
            
        vid = _to_torch_video(clip, self._device)
        model = self._model
        model.to(self._device)
        
        with torch.inference_mode():
            if mode == 'init':
                pts_r = np.stack(points_xy, axis=0).astype(np.float32)
                pts_r[:, 0] *= sh
                pts_r[:, 1] *= sw
                query_pts = torch.zeros((1,len(pts_r),3), dtype = torch.float32)
                query_pts[0,:,1:] = torch.tensor(pts_r)
                #query_pts = self.make_grid_queries(0,256,256,stride=8)
                pred_tracks, track_logits, visible_logits, self.tracking_state = model(video=vid.to(self._device), query_points=query_pts.to(self._device))
                pred_visible = visible_logits > 0
                self.pred_tracks, self.pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
                self.pred_track_logits, self.pred_visible_logits = [track_logits.cpu()], [visible_logits.cpu()]
                return
            else:
                curr_tracks,curr_track_logits,curr_visible_logits,self.tracking_state = model(video=vid, state=self.tracking_state)
                curr_visible = curr_visible_logits > 0
                  
                self.pred_tracks.append(curr_tracks.cpu())
                self.pred_visible.append(curr_visible.cpu())
                self.pred_track_logits.append(curr_track_logits.cpu())
                self.pred_visible_logits.append(curr_visible_logits.cpu())
        next_pts = curr_tracks[0,0].detach().cpu().numpy()
        next_pts[:, 0] /= sh
        next_pts[:, 1] /= sw
        return next_pts
        
        tracks = torch.cat(self.pred_tracks, dim=1).transpose(1, 2)
        pred_visible = torch.cat(self.pred_visible, dim=1).transpose(1, 2)
        track_logits = torch.cat(self.pred_track_logits, dim=1).transpose(1, 2)
        visible_logits = torch.cat(self.pred_visible_logits, dim=1).transpose(1, 2)
        import pdb; pdb.set_trace()
        pred_certainty = tracker_certainty(tracks, track_logits, radius)
        pred_visible_and_certain = (
            F.sigmoid(visible_logits) * pred_certainty
        ) > threshold
    
        if use_certainty:
          occluded = ~(pred_visible_and_certain.squeeze(-1))
        else:
          occluded = ~(pred_visible.squeeze(-1))
        '''
        scalars = evaluation_datasets.compute_tapvid_metrics(
          batch['query_points'].cpu().numpy(),
          batch['occluded'].cpu().numpy(),
          batch['target_points'].cpu().numpy(),
          occluded.numpy() + 0.0,
          tracks.numpy()[..., ::-1],
          query_mode='first',
          get_trackwise_metrics=get_trackwise_metrics,
        )
        '''
        # Normalize outputs
        def _as_np(x: Any) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float().numpy()
            return np.asarray(x)

        # Extract t=1 positions
        x1 = pred[0][0, 1,:,0].detach().cpu()
        y1 = pred[0][0, 1,:,1].detach().cpu()
        v1  = pred[2][0, 1,:,0].detach().cpu().numpy()

        next_pts = np.stack([x1, y1], axis=1)  # resized coords

        if resize_to_model:
            # Map back to original frame size
            next_pts[:, 0] /= sx
            next_pts[:, 1] /= sy

        return next_pts.astype(np.float32), v1.astype(np.float32)


# ---------------------------
# Simple CLI
# ---------------------------

def _parse_args(argv: Sequence[str]) -> Any:
    import argparse
    p = argparse.ArgumentParser(description="TAPNext single-file deployer")
    p.add_argument("video", type=str, help="Path to input video")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--h", type=int, default=256)
    p.add_argument("--w", type=int, default=256)
    p.add_argument("--grid_stride", type=int, default=32)
    p.add_argument("--queries", type=str, default=None, help="Path to JSON with {'queries': [{t,x,y}, ...]} ")
    p.add_argument("--out_video", type=str, default="tracks.mp4")
    p.add_argument("--out_json", type=str, default="tracks.json")
    p.add_argument("--ckpt", type=str, default=None, help="Override checkpoint path")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    runner = TapNextDeployer(device=args.device, image_size=(args.h, args.w))
    if args.ckpt:
        runner.ckpt_path = args.ckpt
    runner.ensure_assets()
    runner.load_model()
    manual = None
    qmode = "grid"
    if args.queries:
        with open(args.queries, "r") as f:
            data = json.load(f)
        manual = data.get("queries")
        qmode = "manual"
    runner.track_video(
        video_path=args.video,
        query_mode=qmode,
        grid_stride=args.grid_stride,
        manual_queries=manual,
        out_video=args.out_video,
        out_points_json=args.out_json,
    )


if __name__ == "__main__":
    main()
