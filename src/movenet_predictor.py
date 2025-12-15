#!/usr/bin/env python3
"""
MoveNet Pose Estimation Module

Simple wrapper for MoveNet Lightning and Thunder models from TensorFlow Hub.
Provides easy-to-use predict function that takes RGB numpy array and returns
pose keypoints and bounding boxes.

Models:
- Lightning: Fast, lower accuracy (192x192 input)
- Thunder: Slower, higher accuracy (256x256 input)

Keypoint Format:
17 keypoints in COCO format:
[nose, left_eye, right_eye, left_ear, right_ear,
 left_shoulder, right_shoulder, left_elbow, right_elbow,
 left_wrist, right_wrist, left_hip, right_hip,
 left_knee, right_knee, left_ankle, right_ankle]

Each keypoint: [y, x, confidence]
Coordinates are normalized [0, 1] relative to image dimensions

Usage:
    predictor = MoveNetPredictor(model_type='lightning')
    keypoints, bbox = predictor.predict(rgb_image)
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class MoveNetPredictor:
    """MoveNet pose estimation predictor"""
    
    # Model URLs from TensorFlow Hub
    MODELS = {
        'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
        'thunder': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
        'multipose': 'https://tfhub.dev/google/movenet/multipose/lightning/1'
    }
    
    # Input sizes for each model
    INPUT_SIZES = {
        'lightning': (192, 192),
        'thunder': (256, 256),
        'multipose': (256, 256)
    }
    
    # Keypoint names (COCO format)
    KEYPOINT_NAMES = [
        'nose',
        'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]
    
    # Skeleton connections for visualization
    SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6), (5, 7), (7, 9),   # Left arm
        (6, 8), (8, 10),                   # Right arm
        (5, 6), (5, 11), (6, 12),         # Torso
        (11, 12), (11, 13), (13, 15),     # Left leg
        (12, 14), (14, 16)                # Right leg
    ]
    
    def __init__(self, model_type='lightning'):
        """
        Initialize MoveNet predictor
        
        Args:
            model_type: 'lightning', 'thunder', or 'multipose'
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Invalid model_type. Choose from: {list(self.MODELS.keys())}")
        
        self.model_type = model_type
        self.input_size = self.INPUT_SIZES[model_type]
        self.is_multipose = (model_type == 'multipose')
        
        print(f"Loading MoveNet {model_type.capitalize()} model...")
        self.model = hub.load(self.MODELS[model_type])
        
        if self.is_multipose:
            self.movenet = self.model.signatures['serving_default']
        else:
            self.movenet = self.model.signatures['serving_default']
        
        print(f"MoveNet {model_type.capitalize()} loaded successfully")
        print(f"Input size: {self.input_size}")
        print(f"Model type: {'Multi-pose' if self.is_multipose else 'Single-pose'}")
    
    def preprocess(self, image):
        """
        Preprocess image for MoveNet
        
        Args:
            image: RGB numpy array [H, W, 3] with values [0, 255]
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Store original dimensions
        self._orig_height = image.shape[0]
        self._orig_width = image.shape[1]
        
        # Calculate scale and padding for resize_with_pad
        target_h, target_w = self.input_size
        scale = min(target_h / self._orig_height, target_w / self._orig_width)
        
        # New dimensions after scaling
        new_h = int(self._orig_height * scale)
        new_w = int(self._orig_width * scale)
        
        # Padding offsets
        self._pad_top = (target_h - new_h) // 2
        self._pad_left = (target_w - new_w) // 2
        self._scale = scale
        
        # Resize to model input size
        image_resized = tf.image.resize_with_pad(
            image, 
            self.input_size[0], 
            self.input_size[1]
        )
        
        # Convert to int32 (MoveNet expects integer input)
        image_int = tf.cast(image_resized, dtype=tf.int32)
        
        # Add batch dimension
        image_batch = tf.expand_dims(image_int, axis=0)
        
        return image_batch
    
    def predict(self, image, conf_threshold=0.3):
        """
        Predict pose keypoints from RGB image
        
        Args:
            image: RGB numpy array [H, W, 3] with values [0, 255]
            conf_threshold: Minimum confidence threshold for keypoints
        
        Returns:
            For single-pose models:
                keypoints: np.array [17, 3] - (y, x, confidence) normalized [0, 1]
                bbox: np.array [4] - (ymin, xmin, ymax, xmax) normalized [0, 1]
            
            For multi-pose models:
                keypoints: np.array [N, 17, 3] - N people detected
                bboxes: np.array [N, 4] - Bounding boxes for N people
        """
        if isinstance(image, np.ndarray):
            image = tf.convert_to_tensor(image)
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.movenet(input_tensor)
        
        if self.is_multipose:
            # Multi-pose output
            keypoints_with_scores = outputs['output_0'].numpy()
            
            # Shape: [1, num_detections, 56]
            # 56 = 17 keypoints * 3 (y, x, score) + 5 (bbox + score)
            num_detections = keypoints_with_scores.shape[1]
            
            all_keypoints = []
            all_bboxes = []
            
            for i in range(num_detections):
                detection = keypoints_with_scores[0, i]
                
                # Extract keypoints [17, 3]
                kpts = detection[:51].reshape(17, 3)
                
                # Filter by confidence
                if np.mean(kpts[:, 2]) < conf_threshold:
                    continue
                
                # Extract bbox [4] + score
                bbox_data = detection[51:]
                bbox = np.array([
                    bbox_data[0],  # ymin
                    bbox_data[1],  # xmin
                    bbox_data[2],  # ymax
                    bbox_data[3]   # xmax
                ])
                
                all_keypoints.append(kpts)
                all_bboxes.append(bbox)
            
            if len(all_keypoints) == 0:
                return np.zeros((0, 17, 3)), np.zeros((0, 4))
            
            return np.array(all_keypoints), np.array(all_bboxes)
        
        else:
            # Single-pose output
            keypoints_with_scores = outputs['output_0'].numpy()
            
            # Shape: [1, 1, 17, 3]
            keypoints = keypoints_with_scores[0, 0]  # [17, 3]
            
            # Calculate bounding box from keypoints
            # Filter valid keypoints (confidence > threshold)
            valid_kpts = keypoints[keypoints[:, 2] > conf_threshold]
            
            if len(valid_kpts) > 0:
                ymin = np.min(valid_kpts[:, 0])
                xmin = np.min(valid_kpts[:, 1])
                ymax = np.max(valid_kpts[:, 0])
                xmax = np.max(valid_kpts[:, 1])
                
                # Add padding (15% to better cover the whole person)
                padding = 0.15
                height = ymax - ymin
                width = xmax - xmin
                
                ymin = max(0, ymin - height * padding)
                xmin = max(0, xmin - width * padding)
                ymax = min(1, ymax + height * padding)
                xmax = min(1, xmax + width * padding)
                
                bbox = np.array([ymin, xmin, ymax, xmax])
            else:
                # No valid keypoints
                bbox = np.array([0, 0, 1, 1])
            
            return keypoints, bbox
    
    def _transform_to_original_coords(self, coords, coord_type='keypoint'):
        """
        Transform normalized coordinates from model space to original image pixel space
        
        Args:
            coords: Normalized coordinates [0, 1] from model
            coord_type: 'keypoint' or 'bbox'
        """
        target_h, target_w = self.input_size
        
        if coord_type == 'keypoint':
            # coords shape: [17, 3] or [N, 17, 3]
            result = coords.copy().astype(np.float32)  # Ensure float for division
            
            # Convert from [0,1] to pixel coordinates in padded image
            if result.ndim == 2:  # Single pose [17, 3]
                result[:, 0] = result[:, 0] * target_h  # y in padded image
                result[:, 1] = result[:, 1] * target_w  # x in padded image
                
                # Remove padding offset
                result[:, 0] -= self._pad_top
                result[:, 1] -= self._pad_left
                
                # Scale back to original image size
                result[:, 0] /= self._scale
                result[:, 1] /= self._scale
            else:  # Multi pose [N, 17, 3]
                result[:, :, 0] = result[:, :, 0] * target_h
                result[:, :, 1] = result[:, :, 1] * target_w
                
                result[:, :, 0] -= self._pad_top
                result[:, :, 1] -= self._pad_left
                
                result[:, :, 0] /= self._scale
                result[:, :, 1] /= self._scale
            
            return result
        
        elif coord_type == 'bbox':
            # coords shape: [4] or [N, 4] - (ymin, xmin, ymax, xmax)
            result = coords.copy().astype(np.float32)  # Ensure float for division
            
            if result.ndim == 1:  # Single bbox [4]
                result[[0, 2]] = result[[0, 2]] * target_h  # y coords
                result[[1, 3]] = result[[1, 3]] * target_w  # x coords
                
                result[[0, 2]] -= self._pad_top
                result[[1, 3]] -= self._pad_left
                
                result[[0, 2]] /= self._scale
                result[[1, 3]] /= self._scale
            else:  # Multi bbox [N, 4]
                result[:, [0, 2]] = result[:, [0, 2]] * target_h
                result[:, [1, 3]] = result[:, [1, 3]] * target_w
                
                result[:, [0, 2]] -= self._pad_top
                result[:, [1, 3]] -= self._pad_left
                
                result[:, [0, 2]] /= self._scale
                result[:, [1, 3]] /= self._scale
            
            return result
    
    def predict_with_pixels(self, image, conf_threshold=0.3):
        """
        Predict pose keypoints and return in pixel coordinates
        
        Args:
            image: RGB numpy array [H, W, 3]
            conf_threshold: Minimum confidence threshold
        
        Returns:
            For single-pose:
                keypoints: np.array [17, 3] - (y, x, confidence) in pixels
                bbox: np.array [4] - (ymin, xmin, ymax, xmax) in pixels
            
            For multi-pose:
                keypoints: np.array [N, 17, 3] - in pixels
                bboxes: np.array [N, 4] - in pixels
        """
        if self.is_multipose:
            keypoints, bboxes = self.predict(image, conf_threshold)
            
            # Transform to original image coordinates
            if len(keypoints) > 0:
                keypoints_px = self._transform_to_original_coords(keypoints, 'keypoint')
                bboxes_px = self._transform_to_original_coords(bboxes, 'bbox')
                return keypoints_px, bboxes_px
            else:
                return keypoints, bboxes
        else:
            keypoints, bbox = self.predict(image, conf_threshold)
            
            # Transform to original image coordinates
            keypoints_px = self._transform_to_original_coords(keypoints, 'keypoint')
            bbox_px = self._transform_to_original_coords(bbox, 'bbox')
            
            return keypoints_px, bbox_px
    
    def draw_predictions(self, image, keypoints, bbox=None, 
                        conf_threshold=0.3, draw_skeleton=True):
        """
        Draw predictions on image
        
        Args:
            image: RGB numpy array [H, W, 3]
            keypoints: Keypoints in pixel coordinates [17, 3] or [N, 17, 3]
            bbox: Bounding box(es) in pixel coordinates
            conf_threshold: Minimum confidence to draw
            draw_skeleton: Whether to draw skeleton connections
        
        Returns:
            Image with drawn predictions
        """
        import cv2
        
        img = image.copy()
        h, w = img.shape[:2]
        
        # Handle multi-pose
        if keypoints.ndim == 3:
            for person_idx in range(len(keypoints)):
                kpts = keypoints[person_idx]
                bb = bbox[person_idx] if bbox is not None else None
                img = self._draw_single_pose(
                    img, kpts, bb, conf_threshold, draw_skeleton
                )
        else:
            img = self._draw_single_pose(
                img, keypoints, bbox, conf_threshold, draw_skeleton
            )
        
        return img
    
    def _draw_single_pose(self, image, keypoints, bbox, conf_threshold, draw_skeleton):
        """Draw single pose on image"""
        import cv2
        
        # Draw bounding box
        if bbox is not None:
            ymin, xmin, ymax, xmax = bbox.astype(int)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw skeleton
        if draw_skeleton:
            for edge in self.SKELETON_EDGES:
                y1, x1, conf1 = keypoints[edge[0]]
                y2, x2, conf2 = keypoints[edge[1]]
                
                if conf1 > conf_threshold and conf2 > conf_threshold:
                    cv2.line(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0), 2
                    )
        
        # Draw keypoints
        for i, (y, x, conf) in enumerate(keypoints):
            if conf > conf_threshold:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
                # Optional: Draw keypoint name
                # cv2.putText(image, self.KEYPOINT_NAMES[i], 
                #            (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                #            0.3, (255, 255, 255), 1)
        
        return image


def demo():
    """Demo script showing how to use MoveNetPredictor"""
    import cv2
    
    # Initialize predictor
    print("=== MoveNet Lightning Demo ===")
    predictor = MoveNetPredictor(model_type='lightning')
    
    # Test with webcam or video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or path for video
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict pose
        keypoints, bbox = predictor.predict_with_pixels(rgb_frame, conf_threshold=0.3)
        
        # Draw predictions
        output_frame = predictor.draw_predictions(
            rgb_frame, keypoints, bbox, 
            conf_threshold=0.3, draw_skeleton=True
        )
        
        # Convert back to BGR for display
        output_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        # Add info text
        cv2.putText(output_bgr, f"Model: {predictor.model_type}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('MoveNet Pose Estimation', output_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
