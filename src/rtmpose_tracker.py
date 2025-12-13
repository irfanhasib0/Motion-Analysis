#!/usr/bin/env python3
"""
Simple RTMPose + RTMDet Pose Tracking Script

A clean implementation using MMPose's modern inferencer API.
This script provides real-time pose estimation with object tracking.

Requirements:
- mmpose
- mmdetection 
- opencv-python
- numpy

Usage:
    python rtmpose_tracker.py [video_path]
    
    If no video_path is provided, webcam will be used.
    
Controls:
    q - Quit
    space - Pause/Resume
    s - Save current frame
    r - Reset tracker

---

POSE2D INFERENCER FORWARD() CALL CHAIN
=======================================

When self.inferencer(frame) is called, it triggers this chain:

Step 1: Pose2DInferencer.__call__() → forward()
    Location: libs/mmpose/mmpose/apis/inferencers/pose2d_inferencer.py
    - @torch.no_grad() disables gradients
    - Calls super().forward(inputs)
    - Merges data samples for topdown models
    - Filters by bbox threshold
    
Step 2: BaseInferencer.forward()
    Location: libs/mmengine/mmengine/infer/infer.py
    - Delegates to self.model.test_step(inputs)
    
Step 3: BaseModel.test_step()
    Location: libs/mmengine/mmengine/model/base_model/base_model.py
    - Runs data_preprocessor(data, False)
        * Image normalization (mean/std)
        * Resizing to model input size
        * Padding if needed
    - Calls _run_forward(data, mode='predict')
    
Step 4: BaseModel._run_forward()
    Location: libs/mmengine/mmengine/model/base_model/base_model.py
    - Unpacks preprocessed data
    - Calls self(**data, mode='predict')
        This triggers __call__ → forward()
    
Step 5: BasePoseEstimator.forward(mode='predict')
    Location: libs/mmpose/mmpose/models/pose_estimators/base.py
    - Routes based on mode:
        * 'loss' → self.loss() for training
        * 'predict' → self.predict() for inference ← TAKES THIS PATH
        * 'tensor' → self._forward() for raw output
    - Sets metainfo if needed
    - Calls self.predict(inputs, data_samples)
    
Step 6: TopDown/BottomUp.predict()
    Location: libs/mmpose/mmpose/models/pose_estimators/topdown.py
    Pipeline:
    a) Feature Extraction
        - feats = self.extract_feat(inputs)
        - backbone.forward(inputs) → Feature maps (e.g., ResNet, HRNet)
        - neck.forward(feats) → Refined features (optional, e.g., FPN)
    
    b) Keypoint Prediction
        - self.head.predict(feats, data_samples)
        - Generates heatmaps or direct coordinates
        - Returns raw predictions
    
    c) Post-processing
        - Decode heatmaps to coordinates
        - Apply coordinate transforms
        - Create PoseDataSample objects
    
    Output: List[PoseDataSample] with predicted keypoints

Step 7: Back to Pose2DInferencer.forward()
    - Receives List[PoseDataSample]
    - Merges samples if topdown mode
    - Filters by bbox_thr if > 0
    - Returns final predictions

SIMPLIFIED CALL CHAIN:
    Pose2DInferencer.forward()
        → BaseInferencer.forward()
            → BaseModel.test_step()
                → data_preprocessor()  [normalize, resize]
                → BaseModel._run_forward()
                    → BasePoseEstimator.forward(mode='predict')
                        → ConcreteEstimator.predict()
                            → backbone → neck → head
                            → post-processing
                        ← PoseDataSample
                ← predictions
        → filter & merge
    ← final results

DATA FLOW:
    frame (numpy array)
        ↓ [preprocessing]
    normalized tensor
        ↓ [backbone]
    feature maps
        ↓ [neck - optional]
    refined features
        ↓ [head]
    heatmaps/coordinates
        ↓ [post-processing]
    PoseDataSample
        ↓ [filtering]
    final predictions

KEY COMPONENTS:
- Data Preprocessor: Handles image normalization, resizing, padding
- Backbone: Feature extraction (ResNet, HRNet, etc.)
- Neck: Feature refinement (FPN, etc.) - optional
- Head: Keypoint prediction (heatmap-based, regression-based)
- Mode Parameter: Controls behavior ('loss', 'predict', 'tensor')

For detailed flowchart see: docs/pose2d_inferencer_forward_flowchart.md
"""
#xtcocotools==1.14.3
#munkres==1.1.4
#wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
#wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
import cv2
import time
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from scipy.special import softmax
import numpy as np
import sys
import logging
from pathlib import Path
import json
import argparse
from matplotlib import pyplot as plt
sys.path.append(str(Path('../libs') / 'mmengine'))
sys.path.append(str(Path('../libs') / 'mmcv'))
sys.path.append(str(Path('../libs') / 'mmdetection'))
sys.path.append(str(Path('../libs') / 'mmpose'))

from mmpose.apis import MMPoseInferencer
from mmpose.apis.inferencers import Pose2DInferencer
from mmdet.apis import DetInferencer
from mmpose.structures import PoseDataSample
from mmdet.apis import init_detector
from mmpose.apis import init_model
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.utils.tensor_utils import to_numpy       

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

#pose2d = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
#pose_weights = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
#det_model = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
#det_weights = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_model='../configs/openmmlab/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
pose_weights='../models/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth'
#pose2d_weights='./model.onnx'
det_model='../configs/openmmlab/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
det_weights='../models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

for path in sys.path[-4:]+[pose_model, pose_weights, det_model, det_weights]:
    print(f"Resolved {Path(path).exists()}, {path}")

logging.basicConfig(level=logging.INFO)

from mmpose.utils import register_all_modules
register_all_modules()
# COCO keypoint skeleton for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (0, 5), (5, 7), (7, 9),          # Left arm
    (0, 6), (6, 8), (8, 10),         # Right arm
    (5, 6),                          # Shoulders
    (11, 12),                        # Hips
    (11, 13), (13, 15),              # Left leg
    (12, 14), (14, 16)               # Right leg
]

class SimpleTracker:
    """Simple tracking based on position similarity"""
    
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['centroid'])
        else:
            # Compute distances between existing objects and new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                             np.array([d['centroid'] for d in detections]), axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of used row and column indices
            used_rows = set()
            used_cols = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = detections[col]['centroid']
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            if D.shape[0] >= D.shape[1]:
                # More objects than detections
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than objects
                for col in unused_cols:
                    self.register(detections[col]['centroid'])
        
        # Return current tracking assignments
        result = {}
        for i, detection in enumerate(detections):
            # Find the closest tracked object
            min_dist = float('inf')
            best_id = None
            for object_id, centroid in self.objects.items():
                dist = np.linalg.norm(np.array(centroid) - np.array(detection['centroid']))
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = object_id
            
            if best_id is not None:
                result[best_id] = detection
        
        return result


class RTMPoseTracker:
    """Main class for RTMPose tracking"""
    
    def __init__(self, model_alias='human', device='cpu', conf_threshold=0.3, 
                 det_fw='torch', pose_fw='torch'):
        """
        Initialize RTMPose tracker
        
        Args:
            model_alias: Model alias ('human', 'body26', 'wholebody', etc.)
            device: Device to run on ('cpu', 'cuda')
            conf_threshold: Detection confidence threshold
            use_onnx_det: If True, use ONNX detection model instead of PyTorch
            use_onnx_pose: If True, use ONNX pose estimation model instead of PyTorch
            det_onnx_path: Path to detection ONNX model (required if use_onnx_det=True)
            pose_onnx_path: Path to pose estimation ONNX model (required if use_onnx_pose=True)
        """

        self.conf_threshold = conf_threshold
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
        self.device = device
        self.det_fw = det_fw
        self.pose_fw = pose_fw
        self._pycuda_context = pycuda.autoinit.context  # Keep reference to avoid premature cleanup
        self.decoder = SimCCLabel(
                    input_size=(256, 192),
                    sigma=(4.9, 5.66),
                    simcc_split_ratio=2,
                    normalize=False,
                    use_dark=True
                )
        if self.det_fw == 'onnx':
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
            
            self.det_session = ort.InferenceSession(det_weights.replace('.pth', '.onnx'), providers=providers)
            logging.info(f"Loaded detection ONNX model from {det_weights.replace('.pth', '.onnx')}")
            
            self.det_input_name = self.det_session.get_inputs()[0].name
            self.det_input_shape = self.det_session.get_inputs()[0].shape[2:]
            logging.info(f"Detection input: {self.det_input_name}, shape: {self.det_input_shape}")
            
            self.det_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.det_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        elif self.det_fw == 'trt':
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)    
            with open(det_weights.replace('.pth', '.engine'), 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            self.det_engine = runtime.deserialize_cuda_engine(engine_data)
            self.det_context = self.det_engine.create_execution_context()
            
            # Allocate buffers
            self.det_inputs = []
            self.det_outputs = []
            self.det_stream = cuda.Stream()
            
            try:
                for binding in self.det_engine:
                    shape = self.det_engine.get_tensor_shape(binding)
                    dtype = trt.nptype(self.det_engine.get_tensor_dtype(binding))
                    
                    # Calculate buffer size
                    size = trt.volume(shape)
                    logging.info(f"Allocating {binding}: shape={shape}, size={size}, dtype={dtype}")
                    
                    # Allocate host and device buffers
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    # Check if input or output using tensor mode (modern TensorRT API)
                    tensor_mode = self.det_engine.get_tensor_mode(binding)
                    if tensor_mode == trt.TensorIOMode.INPUT:
                        self.det_inputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': shape})
                        self.det_input_shape = shape[2:]
                    else:
                        self.det_outputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': shape})
            except cuda.MemoryError as e:
                logging.error(f"CUDA memory allocation failed: {e}")
                logging.error("Try reducing batch size or freeing GPU memory")
                # Clean up any allocated memory
                for inp in self.det_inputs:
                    if 'device' in inp:
                        inp['device'].free()
                for out in self.det_outputs:
                    if 'device' in out:
                        out['device'].free()
                raise RuntimeError(f"Failed to allocate TensorRT detection buffers: {e}")
            
            # For dynamic shapes, DO NOT set tensor addresses during initialization
            # They will be set per inference call after setting input shapes
            
            self.det_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.det_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            self.det_session = None
            logging.info(f"Loaded detection TensorRT model from {det_weights.replace('.pth', '.engine')}")
            logging.info(f"Detection input shape: {self.det_input_shape}")

        elif self.det_fw == 'torch':
            # Build and load detection model directly from config and weights
            self.det_model = init_detector(
                config=det_model,
                checkpoint=det_weights,
                device=device
            )
            self.det_input_shape = [320, 320]
            self.det_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.det_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            self.det_session = None
        else:
            raise ValueError(f"Unsupported detection framework: {self.det_fw}")

        if self.pose_fw == 'onnx':
            # Setup ONNX Runtime providers
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]

            # Load pose estimation ONNX model
            self.pose_session = ort.InferenceSession(pose_weights.replace('.pth', '.onnx'), providers=providers)
            logging.info(f"Loaded pose estimation ONNX model from {pose_weights.replace('.pth', '.onnx')}")
            self.pose_input_name = self.pose_session.get_inputs()[0].name
            self.pose_input_shape = self.pose_session.get_inputs()[0].shape[2:]  # Get H, W
            logging.info(f"Pose input: {self.pose_input_name}, shape: {self.pose_input_shape}")
            self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        elif self.pose_fw == 'trt':
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)    
            with open(pose_weights.replace('.pth', '.engine'), 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            self.pose_engine = runtime.deserialize_cuda_engine(engine_data)
            self.pose_context = self.pose_engine.create_execution_context()
            
            # Allocate buffers
            self.pose_inputs = []
            self.pose_outputs = []
            self.pose_stream = cuda.Stream()
            
            try:
                for binding in self.pose_engine:
                    shape = self.pose_engine.get_tensor_shape(binding)
                    dtype = trt.nptype(self.pose_engine.get_tensor_dtype(binding))
                    
                    # Replace dynamic dimensions (-1) with batch size 1 for allocation
                    alloc_shape = tuple(4 if dim == -1 else dim for dim in shape)
                    size = trt.volume(alloc_shape)
                    
                    logging.info(f"Allocating {binding}: shape={shape}, alloc_shape={alloc_shape}, size={size}, dtype={dtype}")
                    
                    # Allocate host and device buffers
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    # Check if input or output using tensor mode (modern TensorRT API)
                    tensor_mode = self.pose_engine.get_tensor_mode(binding)
                    if tensor_mode == trt.TensorIOMode.INPUT:
                        self.pose_inputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': alloc_shape})
                        self.pose_input_shape = alloc_shape[2:]
                    else:
                        self.pose_outputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': alloc_shape})
            except cuda.MemoryError as e:
                logging.error(f"CUDA memory allocation failed: {e}")
                logging.error("Try reducing batch size or freeing GPU memory")
                # Clean up any allocated memory
                for inp in self.pose_inputs:
                    if 'device' in inp:
                        inp['device'].free()
                for out in self.pose_outputs:
                    if 'device' in out:
                        out['device'].free()
                raise RuntimeError(f"Failed to allocate TensorRT pose buffers: {e}")
            
            # For dynamic shapes, DO NOT set tensor addresses during initialization
            # They will be set per inference call after setting input shapes
            
            self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            self.pose_session = None
            logging.info(f"Loaded pose estimation TensorRT model from {pose_weights.replace('.pth', '.engine')}")
            logging.info(f"Pose input shape: {self.pose_input_shape}")

        elif self.pose_fw == 'torch':
            # Build and load pose model directly from config and weights
            self.pose_model = init_model(
                config=pose_model,
                checkpoint=pose_weights,
                device=device
            )
            self.pose_input_shape = self.pose_model.head.input_size[::-1]
            self.pose_mean = np.array(self.pose_model.data_preprocessor.mean, dtype=np.float32).reshape(3)
            self.pose_std = np.array(self.pose_model.data_preprocessor.std, dtype=np.float32).reshape(3)
            print(self.pose_mean, self.pose_std)
            logging.info(f"RTMPose inferencer initialized with model: {model_alias}")
            self.pose_session = None
        else:
            raise ValueError(f"Unsupported pose framework: {self.pose_fw}")
    
    def __del__(self):
        """Cleanup TensorRT resources"""
        try:
            # Free detection buffers
            if hasattr(self, 'det_inputs'):
                for inp in self.det_inputs:
                    if 'device' in inp and inp['device']:
                        inp['device'].free()
            if hasattr(self, 'det_outputs'):
                for out in self.det_outputs:
                    if 'device' in out and out['device']:
                        out['device'].free()
            
            # Free pose buffers
            if hasattr(self, 'pose_inputs'):
                for inp in self.pose_inputs:
                    if 'device' in inp and inp['device']:
                        inp['device'].free()
            if hasattr(self, 'pose_outputs'):
                for out in self.pose_outputs:
                    if 'device' in out and out['device']:
                        out['device'].free()
            
            logging.info("TensorRT resources cleaned up")
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")
    
    def preprocess_for_detection(self, frame):
        """Preprocess frame for ONNX detection model"""
        target_h, target_w = self.det_input_shape[0], self.det_input_shape[1]
        img_resized = cv2.resize(frame, (target_w, target_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb - self.det_mean) / self.det_std
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        return img_batch
    
    def preprocess_for_pose(self, img_crops):
        """Preprocess cropped image for ONNX pose model"""
        img_batch = []
        for img_crop in img_crops:
            target_h, target_w = self.pose_input_shape[0], self.pose_input_shape[1]
            img_resized = cv2.resize(img_crop, (target_w, target_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = (img_rgb - self.pose_mean) / self.pose_std
            img_transposed = img_normalized.transpose(2, 0, 1)
            img_batch.append(np.expand_dims(img_transposed, axis=0).astype(np.float32))
        img_batch = np.vstack(img_batch)
        return img_batch
    
    def decode_rtmdet_outputs(self, cls_scores, bbox_preds, img_shape, ori_shape, 
                          num_classes=80, score_thr=0.3, nms_iou_thr=0.65, max_per_img=100):
        """
        Decode RTMDet ONNX outputs to bounding boxes.
        
        Based on BaseDenseHead.predict_by_feat and _predict_by_feat_single from MMDetection.
        
        Args:
            cls_scores: List of classification score tensors, one per FPN level
                    Each has shape [1, num_classes, H, W]
                    For 640x640 input: [[1, 80, 80, 80], [1, 80, 40, 40], [1, 80, 20, 20]]
            bbox_preds: List of bbox prediction tensors, one per FPN level
                    Each has shape [1, 4, H, W] (distance format: l, t, r, b)
            img_shape: Tuple (H, W) of the preprocessed image shape (e.g., (640, 640))
            ori_shape: Tuple (H, W) of the original image shape before resizing
            num_classes: Number of object classes (default 80 for COCO)
            score_thr: Score threshold for filtering predictions
            nms_iou_thr: IoU threshold for NMS
            max_per_img: Maximum number of detections to keep
            
        Returns:
            numpy array of shape [N, 6] where each row is [x1, y1, x2, y2, score, class_id]
            Coordinates are scaled back to original image size
        """
        
        # Strides for each FPN level (RTMDet uses 8, 16, 32)
        strides = [8, 16, 32]
        
        # Convert to torch tensors if numpy
        if isinstance(cls_scores[0], np.ndarray):
            cls_scores = [torch.from_numpy(x) for x in cls_scores]
        if isinstance(bbox_preds[0], np.ndarray):
            bbox_preds = [torch.from_numpy(x) for x in bbox_preds]
        
        all_bboxes = []
        all_scores = []
        all_labels = []
        
        # Process each FPN level
        for level_idx, (cls_score, bbox_pred, stride) in enumerate(zip(cls_scores, bbox_preds, strides)):
            # Remove batch dimension and rearrange
            # cls_score: [1, 80, H, W] -> [H, W, 80] -> [H*W, 80]
            cls_score = cls_score[0].permute(1, 2, 0).reshape(-1, num_classes)
            # bbox_pred: [1, 4, H, W] -> [H, W, 4] -> [H*W, 4]
            bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
            
            # Apply sigmoid to get class scores (RTMDet uses sigmoid, not softmax)
            scores = torch.sigmoid(cls_score)
            
            # Get feature map size
            h, w = cls_scores[level_idx].shape[2:]
            
            # Generate anchor points (priors) for this level
            # Create grid of (x, y) coordinates
            y_coords = torch.arange(0, h, dtype=torch.float32) * stride + stride // 2
            x_coords = torch.arange(0, w, dtype=torch.float32) * stride + stride // 2
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Stack to get anchor points [H*W, 2]
            priors = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
            
            # Filter by score threshold and get top-k
            max_scores, labels = scores.max(dim=1)
            valid_mask = max_scores > score_thr
            
            if valid_mask.sum() == 0:
                continue
                
            # Keep only valid predictions
            bbox_pred = bbox_pred[valid_mask]
            priors = priors[valid_mask]
            scores_valid = scores[valid_mask]
            labels = labels[valid_mask]
            max_scores = max_scores[valid_mask]
            
            # Decode boxes from distance format to xyxy format
            # RTMDet predicts distances [left, top, right, bottom] from anchor point
            # Convert to [x1, y1, x2, y2]
            x1 = priors[:, 0] - bbox_pred[:, 0]
            y1 = priors[:, 1] - bbox_pred[:, 1]
            x2 = priors[:, 0] + bbox_pred[:, 2]
            y2 = priors[:, 1] + bbox_pred[:, 3]
            
            # Stack into [N, 4]
            bboxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # Clip to image boundaries
            bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, img_shape[1])  # x coords
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, img_shape[0])  # y coords
            
            all_bboxes.append(bboxes)
            all_scores.append(max_scores)
            all_labels.append(labels)
        
        if len(all_bboxes) == 0:
            return np.array([]).reshape(0, 6)
        
        # Concatenate all levels
        all_bboxes = torch.cat(all_bboxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Apply NMS
        keep_indices = nms(all_bboxes, all_scores, nms_iou_thr)
        
        # Keep top-k
        if len(keep_indices) > max_per_img:
            # Sort by score and keep top max_per_img
            sorted_scores, sorted_indices = all_scores[keep_indices].sort(descending=True)
            keep_indices = keep_indices[sorted_indices[:max_per_img]]
        
        # Get final detections
        final_bboxes = all_bboxes[keep_indices]
        final_scores = all_scores[keep_indices]
        final_labels = all_labels[keep_indices]
        
        # Scale boxes back to original image size
        scale_x = ori_shape[1] / img_shape[1]
        scale_y = ori_shape[0] / img_shape[0]
        final_bboxes[:, 0::2] *= scale_x
        final_bboxes[:, 1::2] *= scale_y
        
        # Stack into [N, 6] format: [x1, y1, x2, y2, score, class_id]
        results = torch.cat([
            final_bboxes,
            final_scores.unsqueeze(1),
            final_labels.unsqueeze(1).float()
        ], dim=1)
        
        return results.cpu().numpy()

    
    def process_frame(self, frame):
        """Process frame using PyTorch/MMPose models with bare minimum inference"""
        detections = []
        
        # Step 1: Direct detection inference
        # Preprocess frame for detection
        ori_shape = torch.tensor(frame.shape[:2])
        frame_rsz = cv2.resize(frame, (self.det_input_shape[1], self.det_input_shape[0]))
        img_shape = torch.tensor(frame_rsz.shape[:2])

        frame_rsz = self.preprocess_for_detection(frame_rsz)
        if self.det_fw=='torch':
            frame_rsz = torch.tensor(frame_rsz).float()
            with torch.no_grad():
                det_result = self.det_model(inputs= frame_rsz, data_samples= None, mode='tensor')
                class_scores = det_result[0]
                bbox_preds   = det_result[1]

        elif self.det_fw=='trt':
            class_scores = []
            bbox_preds = []
            self._pycuda_context.push()
            # Copy input data to device
            np.copyto(self.det_inputs[0]['host'], frame_rsz.ravel())
            cuda.memcpy_htod_async(self.det_inputs[0]['device'], self.det_inputs[0]['host'], self.det_stream)
            
            # Set input shape for dynamic batch engines
            actual_shape = frame_rsz.shape
            self.det_context.set_input_shape(self.det_inputs[0]['name'], actual_shape)
            
            # Set tensor addresses AFTER setting input shape (required for dynamic shapes)
            for inp in self.det_inputs:
                self.det_context.set_tensor_address(inp['name'], int(inp['device']))
            for out in self.det_outputs:
                self.det_context.set_tensor_address(out['name'], int(out['device']))
            
            # Run inference
            self.det_context.execute_async_v3(stream_handle=self.det_stream.handle)
            
            # Get inferred output shapes
            inferred_output_shapes = []
            for output in self.det_outputs:
                inferred_shape = self.det_context.get_tensor_shape(output['name'])
                inferred_output_shapes.append(inferred_shape)
            
            # Copy output data back to host using inferred shapes
            for output, inferred_shape in zip(self.det_outputs, inferred_output_shapes):
                output_size = np.prod(inferred_shape)
                cuda.memcpy_dtoh_async(output['host'][:int(output_size)], output['device'], self.det_stream)
            
            # Synchronize the stream
            self.det_stream.synchronize()
            
            # Retrieve outputs using inferred shapes
            for output, inferred_shape in zip(self.det_outputs, inferred_output_shapes):
                output_array = output['host'][:np.prod(inferred_shape)].reshape(inferred_shape)
                if output_array.shape[1] == 4:
                    bbox_preds.append(torch.from_numpy(output_array))
                else:
                    class_scores.append(torch.from_numpy(output_array))
            self._pycuda_context.pop()
            
        elif self.det_fw=='onnx':
            det_result = self.det_session.run(None, {self.det_input_name: frame_rsz})
            class_scores = det_result[:3]
            bbox_preds  = det_result[3:]
        
        else:
            raise ValueError(f"Unsupported detection framework: {self.det_fw}")
                
        det_outputs = self.decode_rtmdet_outputs(class_scores, bbox_preds, img_shape.numpy(), ori_shape.numpy(),
                                  score_thr=self.conf_threshold)
        
        person_mask   = det_outputs[:,5] == 0
        person_bboxes = det_outputs[:,:4][person_mask]#.cpu().numpy()
        person_scores = det_outputs[:,4][person_mask]#.cpu().numpy()
        
        person_crops = []
        bboxes = []
        for bbox, score in zip(person_bboxes, person_scores):
            if score < self.conf_threshold:
                continue
            
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            #x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            #x1, y1 = int(x1 * scale_w), int(y1 * scale_h)
            #x2, y2 = int(x2 * scale_w), int(y2 * scale_h)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                print(f"Empty crop for bbox: {(x1, y1, x2, y2)}")
                continue
            person_crops.append(person_crop)
            bboxes.append([x1, y1, x2, y2])
            
        if len(person_crops) == 0:
            return self.tracker.update(detections)
        else:
            input_tensor = self.preprocess_for_pose(person_crops)

        if self.pose_fw=='torch':    
            input_tensor = torch.tensor(input_tensor).float()
            with torch.no_grad():
                pose_results = self.pose_model.forward(inputs = input_tensor, data_samples =  None, mode='tensor')

        elif self.pose_fw=='onnx':
            pose_results = self.pose_session.run(None, {self.pose_input_name: input_tensor})
            for i in range(len(pose_results)):
                pose_results[i] = torch.tensor(pose_results[i])

        elif self.pose_fw=='trt':
            pose_bsize = 4
            pose_results = []
            self._pycuda_context.push()
            for _itr in range(0, input_tensor.shape[0], pose_bsize):
                batch_data = input_tensor[_itr:_itr+pose_bsize]
                actual_batch = batch_data.shape[0]
                
                # Set input shape for dynamic batch engines
                input_shape = (actual_batch, 3, 256, 192)
                self.pose_context.set_input_shape(self.pose_inputs[0]['name'], input_shape)
                
                # Set tensor addresses AFTER setting input shape (required for dynamic shapes)
                for inp in self.pose_inputs:
                    self.pose_context.set_tensor_address(inp['name'], int(inp['device']))
                for out in self.pose_outputs:
                    self.pose_context.set_tensor_address(out['name'], int(out['device']))
                
                # Copy input data to device (only copy actual batch size)
                data_size = actual_batch * 3 * 256 * 192
                np.copyto(self.pose_inputs[0]['host'][:data_size], batch_data.ravel())
                cuda.memcpy_htod_async(self.pose_inputs[0]['device'], self.pose_inputs[0]['host'], self.pose_stream)
                
                # Run inference
                self.pose_context.execute_async_v3(stream_handle=self.pose_stream.handle)
                
                # Get inferred output shapes after execution
                inferred_output_shapes = []
                for output in self.pose_outputs:
                    inferred_shape = self.pose_context.get_tensor_shape(output['name'])
                    inferred_output_shapes.append(inferred_shape)
                
                # Copy output data back to host using inferred shapes
                for output, inferred_shape in zip(self.pose_outputs, inferred_output_shapes):
                    output_size = np.prod(inferred_shape)
                    cuda.memcpy_dtoh_async(output['host'][:int(output_size)], output['device'], self.pose_stream)
                
                # Synchronize the stream
                self.pose_stream.synchronize()
                
                # Retrieve outputs using inferred shapes
                batch_results = []
                for output, inferred_shape in zip(self.pose_outputs, inferred_output_shapes):
                    output_array = output['host'][:np.prod(inferred_shape)].reshape(inferred_shape)
                    batch_results.append(torch.from_numpy(output_array))
                pose_results.append(batch_results)
            
            # Flatten batched results for TensorRT
            if len(pose_results) > 0:
                # Concatenate all batches
                pose_results = [
                    torch.cat([batch[0] for batch in pose_results], dim=0),
                    torch.cat([batch[1] for batch in pose_results], dim=0)
                ]
            self._pycuda_context.pop()
        else:
            raise ValueError(f"Unsupported pose framework: {self.pose_fw}")
        
        for i, [pose_res_0, pose_res_1, bbox] in enumerate(zip(pose_results[0], pose_results[1], bboxes)):
            x1, y1, x2, y2 = bbox

            pose_res = to_numpy([pose_res_0[None], pose_res_1[None]], unzip=True)
            keypoints, keypoint_scores = self.decoder.decode(*pose_res[0])
            keypoints = keypoints[0]
            keypoint_scores = keypoint_scores[0]
            
            # Extract keypoints
            if len(pose_res) > 0:
                # Scale keypoints back to original image coordinates
                scale_x = (x2 - x1) / 192
                scale_y = (y2 - y1) / 256
                keypoints[:, 0] = keypoints[:, 0] * scale_x + x1
                keypoints[:, 1] = keypoints[:, 1] * scale_y + y1
                
                # Calculate centroid from valid keypoints
                valid_mask = keypoint_scores > 0.01
                if np.any(valid_mask):
                    #centroid = np.mean(keypoints[valid_mask], axis=0)
                    detections.append({
                        'centroid': [(x1+x2)/2,(y1+y2)/2],#centroid,
                        'keypoints': keypoints,
                        'keypoint_scores': keypoint_scores,
                        'bbox': [[x1, y1, x2, y2]]
                    })
        tracked_objects = self.tracker.update(detections)
        return tracked_objects
    
    def draw_pose(self, frame, keypoints, keypoint_scores, track_id=None, color=(0, 255, 0)):
        """Draw pose on frame"""
        h, w = frame.shape[:2]
        
        # Draw keypoints
        for i, ((x, y), score) in enumerate(zip(keypoints, keypoint_scores)):
            if score > 0.3:  # Only draw visible keypoints
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 3, color, -1)

        keypoints = np.array(keypoints)
        # Draw skeleton
        for (start_idx, end_idx) in COCO_SKELETON:
            if (start_idx < len(keypoint_scores) and end_idx < len(keypoint_scores) and
                keypoint_scores[start_idx] > 0.3 and keypoint_scores[end_idx] > 0.3):
                start_point = keypoints[start_idx].astype(int)
                end_point = keypoints[end_idx].astype(int)
                
                # Check if points are within frame bounds
                if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                    0 <= end_point[0] < w and 0 <= end_point[1] < h):
                    cv2.line(frame, tuple(start_point), tuple(end_point), color, 2)
        
        # Draw track ID
        if track_id is not None and len(keypoints) > 0:
            # Use nose position or first valid keypoint
            text_pos = None
            for i, (kp, score) in enumerate(zip(keypoints, keypoint_scores)):
                if score > 0.3:
                    text_pos = (int(kp[0]), int(kp[1]) - 10)
                    break
            
            if text_pos and 0 <= text_pos[0] < w and 0 <= text_pos[1] < h:
                cv2.putText(frame, f"ID:{track_id}", text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RTMPose Real-time Pose Tracking')
    parser.add_argument('input', nargs='?', default='webcam', 
                       help='Input video path or "webcam" for camera')
    parser.add_argument('--model', default='human', 
                       help='Model alias (human, body26, wholebody, etc.)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--conf', type=float, default=0.01,
                       help='Confidence threshold')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--save-dir', help='Directory to save frames')
    parser.add_argument('--use-onnx', action='store_true',
                       help='Use ONNX models instead of PyTorch')
    parser.add_argument('--det-onnx', help='Path to detection ONNX model')
    parser.add_argument('--pose-onnx', help='Path to pose estimation ONNX model')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = RTMPoseTracker(
        model_alias=args.model,
        device=args.device,
        conf_threshold=args.conf,
        use_onnx=args.use_onnx,
        det_onnx_path=args.det_onnx,
        pose_onnx_path=args.pose_onnx
    )
    
    # Setup input
    if args.input == 'webcam':
        cap = cv2.VideoCapture(0)
        input_name = "webcam"
    else:
        cap = cv2.VideoCapture(args.input)
        input_name = Path(args.input).name
    
    if not cap.isOpened():
        logging.error(f"Failed to open {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"Processing {input_name} - {width}x{height} @ {fps:.1f} FPS")
    
    # Setup output video writer if requested
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Setup save directory
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    paused = False
    
    # Generate colors for different track IDs
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
    ]
    
    print("Controls:")
    print("  q - Quit")
    print("  Space - Pause/Resume")
    print("  s - Save current frame")
    print("  r - Reset tracker")
    print()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                tracked_objects = tracker.process_frame(frame)
                
                # Draw results
                for track_id, detection in tracked_objects.items():
                    color = colors[track_id % len(colors)]
                    
                    # Draw bounding box if available
                    if detection.get('bbox') is not None:
                        bboxes = detection['bbox']
                        for bbox in bboxes:
                            pt1 = (int(bbox[0]), int(bbox[1]))
                            pt2 = (int(bbox[2]), int(bbox[3]))
                            cv2.rectangle(frame, pt1, pt2, color, 2)
                            
                    # Draw pose
                    tracker.draw_pose(
                        frame, 
                        detection['keypoints'],
                        detection['keypoint_scores'],
                        track_id,
                        color
                    )
                
                # Add info text
                info_text = f"Frame: {frame_count} | Objects: {len(tracked_objects)}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                frame_count += 1
            
            # Display frame
            plt.imshow(frame)
            plt.show()
            '''
            # Write to output video if requested
            if writer:
                writer.write(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and save_dir:
                save_path = save_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"Saved frame to {save_path}")
            elif key == ord('r'):
                tracker.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
                print("Tracker reset")
            '''
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        '''
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        '''
        print(f"Processed {frame_count} frames")

if __name__ == '__main__':
    main()