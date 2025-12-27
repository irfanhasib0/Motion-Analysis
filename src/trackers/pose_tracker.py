#!/usr/bin/env python3
#wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
#wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
import cv2
import time
import torch
import numpy as np

import sys
import logging
from pathlib import Path
import json
import argparse
from matplotlib import pyplot as plt
       
from models.movenet_predictor import MoveNetPredictor
from models.blazepose_predictor import BlazePosePredictor
from models.openmmlab_models import RTMDet, RTMPose
from models.deimv2_models import DEIMDet
from trackers.trackers import SimpleTracker, ByteTracker

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


class BasePoseTracker:
    """Base class for pose trackers"""
    def __init__(self):
        self.colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
        self.frame_count = 0

    def process_frame(self, frame):
        """Process a single frame and return tracked poses"""
        raise NotImplementedError("Must be implemented in subclass")
    
    def _draw_pose(self, frame, keypoints, keypoint_scores, track_id=None, color=(0, 255, 0)):
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
                
    def compute_bbox_pose(self, frame):
        # Process frame
        tracked_objects = self.process_frame(frame)
        
        # Draw results
        for track_id, detection in tracked_objects.items():
            color = self.colors[track_id % len(self.colors)]
            
            # Draw bounding box if available
            if detection.get('bbox') is not None:
                bboxes = detection['bbox']
                for bbox in bboxes:
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(frame, pt1, pt2, color, 2)
                    
            # Draw pose
            self._draw_pose(
                frame, 
                detection['keypoints'],
                detection['keypoint_scores'],
                track_id,
                color
            )
        
        # Add info texts
        info_text = f"Frame: {self.frame_count} | Objects: {len(tracked_objects)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.frame_count += 1
        return frame
                
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
    
    def __init__(self, conf_threshold=0.3):
        """
        Initialize BlazePose pose tracker
        
        Args:
            model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
        """
        super().__init__()
        self.conf_threshold = conf_threshold
        self.detector = RTMDet(device='cuda', det_fw='torch', conf_threshold=conf_threshold)
        self.blazepose_predictor = BlazePosePredictor()

    def process_frame(self, frame):
        person_boxes, _ = self.detector.detection_inference(frame)
        detections = []
        for box in person_boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # Add padding for better pose estimation
            h, w = frame.shape[:2]
            pad = 20
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(w, x2 + pad)
            y2_pad = min(h, y2 + pad)
            
            person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if person_crop.size == 0:
                continue
                
            # Convert BGR to RGB for BlazePose
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            #person_crop_rgb = cv2.resize(person_crop_rgb, (192, 256))
            # Get predictions with pixel coordinates
            keypoints, bboxes = self.blazepose_predictor.predict_with_pixels(person_crop_rgb)
            
            if len(keypoints) == 0:
                continue
                
            kp_transformed = []
            keypoint_scores = []
            bbox_xyxy = box
            
            for kp, bbox in zip(keypoints, bboxes):
                keypoint_scores = kp[:, 2]
                
                # Transform keypoints from crop coordinates to original frame coordinates
                curr_kp_transformed = kp[:, :2].copy()
                curr_kp_transformed[:, 0] += x1_pad  # Add x offset with padding
                curr_kp_transformed[:, 1] += y1_pad  # Add y offset with padding
                if len(curr_kp_transformed) > len(kp_transformed):
                    kp_transformed = curr_kp_transformed
                    bbox_xyxy = [bbox[0] + x1_pad, bbox[1] + y1_pad, bbox[2] + x1_pad, bbox[3] + y1_pad]
            
            detections.append({
                'keypoints': kp_transformed,
                'keypoint_scores': keypoint_scores,
                'centroid': ((x1 + x2)/2, (y1 + y2)/2),
                'bbox': [bbox_xyxy]
            })
        
        tracked_objects = self.tracker.update(detections)
        return tracked_objects

class RTMPoseTracker(BasePoseTracker):
    def __init__(self, device='cuda', conf_threshold=0.3, det_fw='torch', pose_fw='torch'):
        super().__init__()
        #self.detector = RTMDet(device=device, det_fw=det_fw, conf_threshold=conf_threshold)
        self.detector = DEIMDet(device=device, det_fw=det_fw, conf_threshold=conf_threshold)
        self.pose_det = RTMPose(device=device, pose_fw=pose_fw, conf_threshold=conf_threshold/3)
    
    def process_frame(self, frame):
        person_boxes, person_scores =self.detector.detection_inference(frame)
        detections = self.pose_det.pose_inference(frame, person_boxes, person_scores)
        tracked_objects = self.tracker.update(detections)
        return tracked_objects