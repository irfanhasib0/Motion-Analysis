#!/usr/bin/env python3
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
       
from movenet_predictor import MoveNetPredictor
from blazepose_predictor import BlazePosePredictor

from openmmlab_models import *
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.utils.tensor_utils import to_numpy
logging.basicConfig(level=logging.INFO)

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

class BasePoseTracker:
    """Base class for pose trackers"""
    def __init__(self):
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
    
    def process_frame(self, frame):
        """Process a single frame and return tracked poses"""
        raise NotImplementedError("Must be implemented in subclass")
    
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
                
class MovenetPoseTracker(BasePoseTracker):
    """Pose tracker using MoveNet model"""
    
    def __init__(self, model_type='multipose'):
        """
        Initialize MoveNet pose tracker
        
        Args:
            model_type: 'lightning', 'thunder', or 'multipose'
        """
        super().__init__()
        self.movenet_predictor = MoveNetPredictor(model_type=model_type)

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get predictions with pixel coordinates
        if self.movenet_predictor.is_multipose:
            keypoints, bboxes = self.movenet_predictor.predict_with_pixels(rgb_frame)
        else:
            keypoints_single, bbox_single = self.movenet_predictor.predict_with_pixels(rgb_frame)
            # Convert to multi-pose format
            keypoints = np.array([keypoints_single]) if len(keypoints_single) > 0 else np.zeros((0, 17, 3))
            bboxes = np.array([bbox_single]) if len(bbox_single) == 4 else np.zeros((0, 4))
        
        detections = []
        for kp, bbox in zip(keypoints, bboxes):
            # Extract keypoint scores (confidence values)
            keypoint_scores = kp[:, 2]
            
            # Calculate centroid from visible keypoints
            valid_kp = kp[keypoint_scores > 0.3][:, :2]  # Use only visible keypoints
            if len(valid_kp) == 0:
                continue
            
            centroid = valid_kp.mean(axis=0).tolist()
            
            # Convert bbox from [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
            bbox_xyxy = [bbox[1], bbox[0], bbox[3], bbox[2]]
            
            detections.append({
                'keypoints': kp[:, :2][:, [1, 0]],  # Swap to y, x coordinates
                'keypoint_scores': keypoint_scores,
                'centroid': centroid[::-1],  # Swap centroid to y, x
                'bbox': [bbox_xyxy]
            })
        
        tracked_objects = self.tracker.update(detections)
        return tracked_objects

class BlazePoseTracker(BasePoseTracker):
    """Pose tracker using BlazePose model"""
    
    def __init__(self, model_complexity=2):
        """
        Initialize BlazePose pose tracker
        
        Args:
            model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
        """
        super().__init__()
        self.blazepose_predictor = BlazePosePredictor()

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get predictions with pixel coordinates
        keypoints, bboxes = self.blazepose_predictor.predict_with_pixels(rgb_frame)
        detections = []
        for kp, bbox in zip(keypoints, bboxes):
            # Extract keypoint scores (confidence values)
            keypoint_scores = kp[:, 2]
            
            # Calculate centroid from visible keypoints
            valid_kp = kp[keypoint_scores > 0.3][:, :2]  # Use only visible keypoints
            if len(valid_kp) == 0:
                continue
            
            centroid = valid_kp.mean(axis=0).tolist()
            
            # Convert bbox from [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
            bbox_xyxy = [bbox[1], bbox[0], bbox[3], bbox[2]]
            
            detections.append({
                'keypoints': kp[:, :2][:, [1, 0]],  # Swap to y, x coordinates
                'keypoint_scores': keypoint_scores,
                'centroid': centroid[::-1],  # Swap centroid to y, x
                'bbox': [bbox_xyxy]
            })
        
        tracked_objects = self.tracker.update(detections)
        return tracked_objects

class RTMPoseTracker(BasePoseTracker):
    def __init__(self, device='cpu', conf_threshold=0.3, det_fw='torch', pose_fw='torch'):
        super().__init__()
        self.detector = RTMDet(device=device, det_fw=det_fw, conf_threshold=conf_threshold)
        self.pose_det = RTMPose(device=device, pose_fw=pose_fw, conf_threshold=conf_threshold)
    
    def process_frame(self, frame):
        person_boxes, person_scores =self.detector.detection_inference(frame)
        detections = self.pose_det.pose_inference(frame, person_boxes, person_scores)
        tracked_objects = self.tracker.update(detections)
        return tracked_objects
        
class RTMPose:
    def __init__(self, device ='cpu', pose_fw='torch', conf_threshold=0.3):
        self.pose_fw = pose_fw
        self.conf_threshold = conf_threshold
        
        if self.pose_fw == 'onnx':
            self.pose_model = OnnxModelPose(pose_weights.replace('.pth', '.onnx'), device=device)
    
        elif self.pose_fw == 'trt':
            self.pose_model = TRTModelPose(pose_weights.replace('.pth', '.engine'), device=device)

        elif self.pose_fw == 'torch':
            self.pose_model = TorchModelPose(pose_model=pose_model, pose_weights=pose_weights, device=device)
        else:
            raise ValueError(f"Unsupported pose framework: {self.pose_fw}")
        
        self.pose_input_shape, self.pose_mean, self.pose_std = self.pose_model.get_params()
        

        self.decoder = SimCCLabel(
                    input_size=(256, 192),
                    sigma=(4.9, 5.66),
                    simcc_split_ratio=2,
                    normalize=False,
                    use_dark=True
                )
    
    def preprocess_for_pose(self, frame, bboxes, scores):
        """Preprocess cropped image for ONNX pose model"""
        img_batch = []
        proc_bboxes = []
        for bbox, score in zip(bboxes, scores):
            if score < self.conf_threshold:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            img_crop = frame[y1:y2, x1:x2]
            if img_crop.size == 0:
                print(f"Empty crop for bbox: {(x1, y1, x2, y2)}")
                continue
            target_h, target_w = self.pose_input_shape[0], self.pose_input_shape[1]
            img_resized = cv2.resize(img_crop, (target_w, target_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = (img_rgb - self.pose_mean) / self.pose_std
            img_transposed = img_normalized.transpose(2, 0, 1)
            img_batch.append(np.expand_dims(img_transposed, axis=0).astype(np.float32))
            proc_bboxes.append(bbox)
        img_batch = np.vstack(img_batch)
        return img_batch, proc_bboxes
    
    def postprocess_pose_results(self, pose_results, bboxes):
        detections = []
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
        return detections
    
    def pose_inference(self, frame, person_bboxes, person_scores):
        input_tensor, proc_bboxes = self.preprocess_for_pose(frame, person_bboxes, person_scores)
        if len(proc_bboxes) == 0:
            return []
        pose_results = self.pose_model._forward(input_tensor)
        detections = self.postprocess_pose_results(pose_results, proc_bboxes)
        return detections
    
class RTMDet:
    def __init__(self, device='cpu', conf_threshold=0.3, det_fw='torch'):
        """
        Initialize RTMDet detector
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.det_fw = det_fw
        
        if self.det_fw == 'onnx':
            self.det_model = OnnxModelDet(det_weights.replace('.pth', '.onnx'), device=device)

        elif self.det_fw == 'trt':
            self.det_model = TRTModelDet(det_weights.replace('.pth', '.engine'), device=device)

        elif self.det_fw == 'torch':
            self.det_model = TorchModelDet(det_model=det_model, det_weights=det_weights, device=device)
        else:
            raise ValueError(f"Unsupported detection framework: {self.det_fw}")
        
        self.det_input_shape, self.det_mean, self.det_std = self.det_model.get_params()

    def preprocess_for_detection(self, frame):
        """Preprocess frame for ONNX detection model"""
        target_h, target_w = self.det_input_shape[0], self.det_input_shape[1]
        img_resized = cv2.resize(frame, (target_w, target_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb - self.det_mean) / self.det_std
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        return img_batch
    
    def postprocess_for_detection(self, cls_scores, bbox_preds, img_shape, ori_shape, 
                          num_classes=80, score_thr=0.3, nms_iou_thr=0.65, max_per_img=100):
        """Decode RTMDet outputs to bounding boxes.
        
        Args:
            cls_scores: List of [1, 80, H, W] classification tensors per FPN level
            bbox_preds: List of [1, 4, H, W] bbox tensors (l,t,r,b format)
            img_shape: Preprocessed image shape (H, W)
            ori_shape: Original image shape (H, W)
            
        Returns:
            [N, 6] array: [x1, y1, x2, y2, score, class_id] in original image coords
        """
        
        strides = [8, 16, 32]
        
        if isinstance(cls_scores[0], np.ndarray):
            cls_scores = [torch.from_numpy(x) for x in cls_scores]
        if isinstance(bbox_preds[0], np.ndarray):
            bbox_preds = [torch.from_numpy(x) for x in bbox_preds]
        
        all_bboxes = []
        all_scores = []
        all_labels = []
        
        # Process each FPN level
        for level_idx, (cls_score, bbox_pred, stride) in enumerate(zip(cls_scores, bbox_preds, strides)):
            # cls_score: [1, 80, H, W] -> [H, W, 80] -> [H*W, 80]
            cls_score = cls_score[0].permute(1, 2, 0).reshape(-1, num_classes)
            # bbox_pred: [1, 4, H, W] -> [H, W, 4] -> [H*W, 4]
            bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
            
            # Apply sigmoid to get class scores (RTMDet uses sigmoid, not softmax)
            scores = torch.sigmoid(cls_score)
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
    
    def detection_inference(self, frame):
        """Process frame using PyTorch/MMPose models with bare minimum inference"""
        # Step 1: Direct detection inference
        # Preprocess frame for detection
        ori_shape = torch.tensor(frame.shape[:2])
        frame_rsz = cv2.resize(frame, (self.det_input_shape[1], self.det_input_shape[0]))
        img_shape = torch.tensor(frame_rsz.shape[:2])

        frame_rsz = self.preprocess_for_detection(frame_rsz)
        bbox_preds, class_scores = self.det_model._forward(frame_rsz)        
        det_outputs = self.postprocess_for_detection(class_scores, bbox_preds, img_shape.numpy(), ori_shape.numpy(), score_thr=self.conf_threshold)
        
        person_mask   = det_outputs[:,5] == 0
        person_bboxes = det_outputs[:,:4][person_mask]
        person_scores = det_outputs[:,4][person_mask]
        
        return person_bboxes, person_scores
        
    
    

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