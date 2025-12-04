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

#pose2d = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
#pose_weights = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
#det_model = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
#det_weights = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose2d='../configs/openmmlab/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
pose2d_weights='../models/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth'
#pose2d_weights='./model.onnx'
det_model='../configs/openmmlab/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
det_weights='../models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

for path in sys.path[-4:]+[pose2d, pose2d_weights, det_model, det_weights]:
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

class ONNXInferenceWrapper:
    """
    Wrapper for ONNX-based pose estimation.
    Provides the same interface as MMPoseInferencer but uses ONNX Runtime.
    """
    
    def __init__(self, det_onnx_path, pose_onnx_path, device='cpu'):
        """
        Initialize ONNX wrapper
        
        Args:
            det_onnx_path: Path to detection ONNX model
            pose_onnx_path: Path to pose estimation ONNX model
            device: 'cpu' or 'cuda'
        """
        import onnxruntime as ort
        
        # Setup ONNX Runtime providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Load ONNX models
        self.det_session = ort.InferenceSession(det_onnx_path, providers=providers)
        self.pose_session = ort.InferenceSession(pose_onnx_path, providers=providers)
        
        # Get model metadata
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_input_shape = self.det_session.get_inputs()[0].shape
        self.pose_input_name = self.pose_session.get_inputs()[0].name
        self.pose_input_shape = self.pose_session.get_inputs()[0].shape
        
        # Preprocessing parameters (ImageNet statistics)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        logging.info(f"ONNX Detection model loaded: {det_onnx_path}")
        logging.info(f"  Input: {self.det_input_name}, Shape: {self.det_input_shape}")
        logging.info(f"ONNX Pose model loaded: {pose_onnx_path}")
        logging.info(f"  Input: {self.pose_input_name}, Shape: {self.pose_input_shape}")
    
    def preprocess_image(self, image, target_size):
        """Preprocess image for ONNX model input"""
        # Resize
        img = cv2.resize(image, target_size)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = (img - self.mean) / self.std
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension
        img = np.expand_dims(img, 0).astype(np.float32)
        return img
    
    def run_detection(self, frame, conf_threshold=0.3):
        """Run detection on frame and return bboxes"""
        h, w = frame.shape[:2]
        target_h, target_w = self.det_input_shape[2], self.det_input_shape[3]
        import pdb; pdb.set_trace()
        # Preprocess
        input_tensor = self.preprocess_image(frame, (target_w, target_h))
        
        # Run inference
        outputs = self.det_session.run(None, {self.det_input_name: input_tensor})
        
        # RTMDet outputs: [cls_scores (3 scales), bbox_preds (3 scales)]
        # Typically outputs[0-2] are classification scores, outputs[3-5] are bbox predictions
        bboxes = []
        scale_x = w / target_w
        scale_y = h / target_h
        
        # Process each scale
        num_scales = len(outputs) // 2
        for scale_idx in range(num_scales):
            cls_scores = outputs[scale_idx]  # Shape: [batch, num_anchors, num_classes]
            bbox_preds = outputs[scale_idx + num_scales]  # Shape: [batch, num_anchors, 4]
            
            # Remove batch dimension
            cls_scores = cls_scores[0]
            bbox_preds = bbox_preds[0]
            
            # Get person class scores (class 0 in COCO)
            person_scores = cls_scores[0]
            
            # Filter by confidence threshold
            valid_indices = person_scores > conf_threshold
            import pdb; pdb.set_trace()
            if not np.any(valid_indices):
                continue
            
            valid_bboxes = bbox_preds[valid_indices]
            valid_scores = person_scores[valid_indices]
            
            # Convert bbox format (usually center_x, center_y, w, h or x1, y1, x2, y2)
            for bbox, score in zip(valid_bboxes, valid_scores):
                # Assuming bbox format is [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                
                # Scale back to original image size
                x1 = np.clip(x1 * scale_x, 0, w)
                y1 = np.clip(y1 * scale_y, 0, h)
                x2 = np.clip(x2 * scale_x, 0, w)
                y2 = np.clip(y2 * scale_y, 0, h)
                
                bboxes.append([x1, y1, x2, y2, float(score)])
        
        # Apply NMS to remove duplicate detections
        if len(bboxes) > 0:
            bboxes = self._apply_nms(bboxes, iou_threshold=0.5)
        
        return bboxes
    
    def run_pose(self, frame, bbox):
        """Run pose estimation on cropped bbox region"""
        # Crop bbox region with margin
        x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        
        crop_h, crop_w = crop.shape[:2]
        target_h, target_w = self.pose_input_shape[2], self.pose_input_shape[3]
        
        # Preprocess
        input_tensor = self.preprocess_image(crop, (target_w, target_h))
        
        # Run inference
        outputs = self.pose_session.run(None, {self.pose_input_name: input_tensor})
        
        # Postprocess - adjust based on your pose model output format
        pose_output = outputs[0][0]  # Remove batch dimension
        
        keypoints = []
        keypoint_scores = []
        
        # Handle different output formats
        if len(pose_output.shape) == 2 and pose_output.shape[1] == 2:
            # Format: [num_keypoints, 2] - xy coordinates
            scale_x = crop_w / target_w
            scale_y = crop_h / target_h
            
            for kpt in pose_output:
                x_abs = x1 + (kpt[0] * scale_x)
                y_abs = y1 + (kpt[1] * scale_y)
                keypoints.append([x_abs, y_abs])
                keypoint_scores.append(1.0)  # No confidence in this format
        
        elif len(pose_output.shape) == 2 and pose_output.shape[1] == 3:
            # Format: [num_keypoints, 3] - xy + confidence
            scale_x = crop_w / target_w
            scale_y = crop_h / target_h
            
            for kpt in pose_output:
                x_abs = x1 + (kpt[0] * scale_x)
                y_abs = y1 + (kpt[1] * scale_y)
                keypoints.append([x_abs, y_abs])
                keypoint_scores.append(float(kpt[2]))
        
        elif len(pose_output.shape) == 3:
            # Format: [num_keypoints, h, w] - heatmaps
            for heatmap in pose_output:
                # Find max location
                max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        return np.array(keypoints), np.array(keypoint_scores)
    
    def _apply_nms(self, bboxes, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to bboxes"""
        if len(bboxes) == 0:
            return []
        
        bboxes = np.array(bboxes)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        scores = bboxes[:, 4]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return bboxes[keep].tolist()
    
    def __call__(self, frame, return_vis=False):
        """
        Main inference method - mimics MMPoseInferencer interface
        
        Args:
            frame: Input frame (numpy array or path)
            return_vis: Whether to return visualization (not implemented for ONNX)
        
        Yields:
            dict: Results in the same format as MMPoseInferencer
        """
        # Load image if path is provided
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        
        # Run detection
        bboxes = self.run_detection(frame, conf_threshold=0.3)
        # Run pose for each detected person
        predictions = []
        for bbox in bboxes:
            keypoints, keypoint_scores = self.run_pose(frame, bbox)
            
            if keypoints is not None:
                predictions.append({
                    'keypoints': keypoints,
                    'keypoint_scores': keypoint_scores,
                    'bbox': [bbox[:4]]  # Wrap in list to match MMPose format
                })
        
        # Return in MMPoseInferencer format
        result = {
            'predictions': [predictions],
            'visualization': None
        }
        
        yield result


class RTMPoseTracker:
    """Main class for RTMPose tracking"""
    
    def __init__(self, model_alias='human', device='cpu', conf_threshold=0.3, 
                 use_onnx=False, det_onnx_path=None, pose_onnx_path=None):
        """
        Initialize RTMPose tracker
        
        Args:
            model_alias: Model alias ('human', 'body26', 'wholebody', etc.)
            device: Device to run on ('cpu', 'cuda')
            conf_threshold: Detection confidence threshold
            use_onnx: If True, use ONNX wrapper instead of MMPose
            det_onnx_path: Path to detection ONNX model (required if use_onnx=True)
            pose_onnx_path: Path to pose estimation ONNX model (required if use_onnx=True)
        """
        self.conf_threshold = conf_threshold
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
        self.use_onnx = use_onnx
        
        if use_onnx:
            # Use ONNX wrapper
            if not det_onnx_path or not pose_onnx_path:
                raise ValueError("Both det_onnx_path and pose_onnx_path must be provided when use_onnx=True")
            
            self.inferencer = ONNXInferenceWrapper(
                det_onnx_path=det_onnx_path,
                pose_onnx_path=pose_onnx_path,
                device=device
            )
            logging.info(f"Initialized with ONNX models")
        else:
            # Use original MMPose inferencer
            from mmpose.apis import MMPoseInferencer
            
            self.inferencer = MMPoseInferencer(
                pose2d=pose2d,
                pose2d_weights=pose2d_weights,
                det_model=det_model,
                det_weights=det_weights,
                det_cat_ids=[0],  # Person class
                device=device
            )
            logging.info(f"RTMPose inferencer initialized with model: {model_alias}")

    def process_frame(self, frame):
        """Process a single frame and return results"""
        # Run inference (works with both MMPose and ONNX wrapper due to same interface)
        result_generator = self.inferencer(frame, return_vis=False)
        result = next(result_generator)
        
        # Extract predictions
        predictions = result['predictions'][0] if result['predictions'] else []
        
        # Process detections for tracking
        detections = []
        for pred in predictions:
            if 'keypoint_scores' in pred:
                # Check if detection confidence is above threshold
                avg_score = np.mean(pred['keypoint_scores'])
                if avg_score >= self.conf_threshold:
                    # Calculate centroid from keypoints
                    keypoints = pred['keypoints']
                    valid_kpts = keypoints
                    if len(valid_kpts) > 0:
                        centroid = np.mean(valid_kpts, axis=0)
                        detections.append({
                            'centroid': centroid,
                            'keypoints': keypoints,
                            'keypoint_scores': pred['keypoint_scores'],
                            'bbox': pred.get('bbox', None)
                        })
        
        # Update tracker
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