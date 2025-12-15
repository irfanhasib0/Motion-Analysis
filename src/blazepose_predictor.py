#!/usr/bin/env python3
"""
BlazePose Pose Estimation Module (MediaPipe Tasks API)

Simple wrapper for MediaPipe BlazePose model using Tasks API.
Provides easy-to-use predict function that takes RGB numpy array and returns
pose keypoints and bounding boxes.

BlazePose detects 33 pose landmarks in 3D:
- 0-10: Face landmarks
- 11-22: Upper body (shoulders, elbows, wrists)
- 23-32: Lower body (hips, knees, ankles, feet)

Each landmark: [x, y, z, visibility]
- x, y: Normalized [0, 1] relative to image dimensions
- z: Depth relative to hips (smaller is closer)
- visibility: Likelihood of being visible [0, 1]

Usage:
    predictor = BlazePosePredictor(num_poses=2, min_detection_confidence=0.5)
    keypoints, bboxes = predictor.predict(rgb_image)
"""

import sys
import os
import numpy as np
import cv2

# Use pip-installed mediapipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
class BlazePosePredictor:
    """MediaPipe BlazePose pose estimation predictor using Solutions API"""
    
    # Landmark indices (33 total)
    LANDMARK_NAMES = [
        'nose',                      # 0
        'left_eye_inner',           # 1
        'left_eye',                 # 2
        'left_eye_outer',           # 3
        'right_eye_inner',          # 4
        'right_eye',                # 5
        'right_eye_outer',          # 6
        'left_ear',                 # 7
        'right_ear',                # 8
        'mouth_left',               # 9
        'mouth_right',              # 10
        'left_shoulder',            # 11
        'right_shoulder',           # 12
        'left_elbow',               # 13
        'right_elbow',              # 14
        'left_wrist',               # 15
        'right_wrist',              # 16
        'left_pinky',               # 17
        'right_pinky',              # 18
        'left_index',               # 19
        'right_index',              # 20
        'left_thumb',               # 21
        'right_thumb',              # 22
        'left_hip',                 # 23
        'right_hip',                # 24
        'left_knee',                # 25
        'right_knee',               # 26
        'left_ankle',               # 27
        'right_ankle',              # 28
        'left_heel',                # 29
        'right_heel',               # 30
        'left_foot_index',          # 31
        'right_foot_index'          # 32
    ]
    
    # Skeleton connections for visualization
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),     # Left eye
        (0, 4), (4, 5), (5, 6), (6, 8),     # Right eye
        (9, 10),                             # Mouth
        (11, 12),                            # Shoulders
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
        (11, 23), (12, 24), (23, 24),       # Torso
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
    ]
    
    def __init__(self, 
                 model_path='../models/pose_landmarker.task',
                 num_poses=1,
                 min_pose_detection_confidence=0.5,
                 min_pose_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize BlazePose predictor using Tasks API
        
        Args:
            model_path: Path to .task model file (downloads default if None)
            num_poses: Maximum number of poses to detect (1-10)
            min_pose_detection_confidence: Minimum confidence for detection [0, 1]
            min_pose_presence_confidence: Minimum confidence for presence [0, 1]
            min_tracking_confidence: Minimum confidence for tracking [0, 1]
        """
        print("Initializing MediaPipe BlazePose (Tasks API)...")
        print(f"MediaPipe version: {mp.__version__}")
        print(f"MediaPipe location: {mp.__file__}")
        
        # Download model if not provided
        if model_path is None:
            model_path = self._get_default_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Using model: {model_path}")
        
        # Create PoseLandmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            #running_mode=vision.RunningMode.IMAGE,
            #num_poses=num_poses,
            #min_pose_detection_confidence=min_pose_detection_confidence,
            #min_pose_presence_confidence=min_pose_presence_confidence,
            #min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )
        
        # Create detector
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.num_poses = num_poses
        
        print(f"BlazePose initialized successfully")
        print(f"Max poses: {num_poses}")
        print(f"Detection confidence: {min_pose_detection_confidence}")
    
    def _get_default_model(self):
        """Download default BlazePose model from MediaPipe"""
        model_dir = os.path.join(os.path.dirname(__file__), '../models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'pose_landmarker_heavy.task')
        
        if not os.path.exists(model_path):
            print("Downloading default BlazePose model...")
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise
        else:
            print(f"Using cached model: {model_path}")
        
        return model_path
    

    
    def predict(self, image, conf_threshold=0.1):
        """
        Predict pose keypoints from RGB image
        
        Args:
            image: RGB numpy array [H, W, 3] with values [0, 255]
            conf_threshold: Minimum visibility threshold for keypoints
        
        Returns:
            keypoints: np.array [N, 33, 4] - N people, 33 landmarks, (x, y, z, visibility)
                      Coordinates are normalized [0, 1]
            bboxes: np.array [N, 4] - (xmin, ymin, xmax, ymax) normalized [0, 1]
        """
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect poses
        results = self.detector.detect(mp_image)
        
        all_keypoints = []
        all_bboxes = []
        
        # Check if any poses were detected
        if not results.pose_landmarks:
            return np.zeros((0, 33, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
        # Process each detected pose
        for pose_landmarks in results.pose_landmarks:
            # Extract keypoints [33, 4]
            kpts = np.zeros((33, 4), dtype=np.float32)
            for i, landmark in enumerate(pose_landmarks):
                kpts[i, 0] = landmark.x
                kpts[i, 1] = landmark.y
                kpts[i, 2] = landmark.z
                kpts[i, 3] = landmark.visibility
            
            # Filter by visibility
            if np.mean(kpts[:, 3]) < conf_threshold:
                continue
                
            # Calculate bounding box from visible landmarks
            visible_kpts = kpts[kpts[:, 3] > conf_threshold]
            
            if len(visible_kpts) > 0:
                xmin = np.min(visible_kpts[:, 0])
                ymin = np.min(visible_kpts[:, 1])
                xmax = np.max(visible_kpts[:, 0])
                ymax = np.max(visible_kpts[:, 1])
                
                # Add padding (15%)
                width = xmax - xmin
                height = ymax - ymin
                padding = 0.15
                
                xmin = max(0, xmin - width * padding)
                ymin = max(0, ymin - height * padding)
                xmax = min(1, xmax + width * padding)
                ymax = min(1, ymax + height * padding)
                
                bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            else:
                bbox = np.array([0, 0, 1, 1], dtype=np.float32)
            
            all_keypoints.append(kpts)
            all_bboxes.append(bbox)
        
        if len(all_keypoints) == 0:
            return np.zeros((0, 33, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
        return np.array(all_keypoints), np.array(all_bboxes)
    
    def predict_with_pixels(self, image, conf_threshold=0.5):
        """
        Predict pose keypoints and return in pixel coordinates
        
        Args:
            image: RGB numpy array [H, W, 3]
            conf_threshold: Minimum visibility threshold
        
        Returns:
            keypoints: np.array [N, 33, 4] - (x, y, z, visibility) in pixels (z unchanged)
            bboxes: np.array [N, 4] - (xmin, ymin, xmax, ymax) in pixels
        """
        h, w = image.shape[:2]
        
        keypoints, bboxes = self.predict(image, conf_threshold)
        
        if len(keypoints) > 0:
            keypoints_px = keypoints.copy()
            keypoints_px[:, :, 0] *= w  # x coordinates
            keypoints_px[:, :, 1] *= h  # y coordinates
            # z stays as is (depth)
            
            bboxes_px = bboxes.copy()
            bboxes_px[:, [0, 2]] *= w  # x coordinates
            bboxes_px[:, [1, 3]] *= h  # y coordinates
            
            return keypoints_px, bboxes_px
        
        return keypoints, bboxes
    
    def draw_predictions(self, image, keypoints, bboxes=None, 
                        conf_threshold=0.5, draw_skeleton=True):
        """
        Draw predictions on image
        
        Args:
            image: RGB numpy array [H, W, 3]
            keypoints: Keypoints in pixel coordinates [N, 33, 4]
            bboxes: Bounding boxes in pixel coordinates [N, 4]
            conf_threshold: Minimum visibility to draw
            draw_skeleton: Whether to draw skeleton connections
        
        Returns:
            Image with drawn predictions
        """
        img = image.copy()
        
        if len(keypoints) == 0:
            return img
        
        # Draw each detected pose
        for person_idx in range(len(keypoints)):
            kpts = keypoints[person_idx]
            bbox = bboxes[person_idx] if bboxes is not None else None
            
            # Draw bounding box
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox.astype(int)
                color = self._get_color(person_idx)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw skeleton
            if draw_skeleton:
                for connection in self.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    
                    start_point = kpts[start_idx]
                    end_point = kpts[end_idx]
                    
                    if (start_point[3] > conf_threshold and 
                        end_point[3] > conf_threshold):
                        start_pos = (int(start_point[0]), int(start_point[1]))
                        end_pos = (int(end_point[0]), int(end_point[1]))
                        
                        color = self._get_color(person_idx)
                        cv2.line(img, start_pos, end_pos, color, 2)
            
            # Draw keypoints
            for i, kpt in enumerate(kpts):
                if kpt[3] > conf_threshold:
                    x, y = int(kpt[0]), int(kpt[1])
                    color = self._get_color(person_idx)
                    cv2.circle(img, (x, y), 3, color, -1)
        
        return img
    
    def _get_color(self, idx):
        """Get color for person index"""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (128, 0, 255), (0, 128, 255)
        ]
        return colors[idx % len(colors)]
    
    def close(self):
        """Close the detector"""
        if self.detector:
            self.detector.close()


def demo():
    """Demo script showing how to use BlazePosePredictor"""
    
    # Initialize predictor
    print("=== MediaPipe BlazePose Demo (Tasks API) ===")
    predictor = BlazePosePredictor(
        num_poses=2, 
        min_pose_detection_confidence=0.5
    )
    
    # Test with webcam or video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or path for video
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict pose
            keypoints, bboxes = predictor.predict_with_pixels(rgb_frame, conf_threshold=0.5)
            
            # Draw predictions
            output_frame = predictor.draw_predictions(
                rgb_frame, keypoints, bboxes, 
                conf_threshold=0.5, draw_skeleton=True
            )
            
            # Convert back to BGR for display
            output_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            # Add info text
            cv2.putText(output_bgr, f"People detected: {len(keypoints)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('BlazePose Detection', output_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        predictor.close()


if __name__ == "__main__":
    demo()
