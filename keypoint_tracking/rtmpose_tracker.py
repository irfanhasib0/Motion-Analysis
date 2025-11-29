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
"""
#xtcocotools==1.14.3
#munkres==1.1.4
#wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
#wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
import cv2
import numpy as np
import sys
import logging
from pathlib import Path
import json
import argparse
from matplotlib import pyplot as plt
sys.path.append(str(Path('/projects/ext/libs') / 'mmengine'))
sys.path.append(str(Path('/projects/ext/libs') / 'mmcv'))
sys.path.append(str(Path('/projects/ext/libs') / 'mmdetection'))
sys.path.append(str(Path('/projects/ext/libs') / 'mmpose'))
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
    
    def __init__(self, model_alias='human', device='cpu', conf_threshold=0.3):
        """
        Initialize RTMPose tracker
        
        Args:
            model_alias: Model alias ('human', 'body26', 'wholebody', etc.)
            device: Device to run on ('cpu', 'cuda')
            conf_threshold: Detection confidence threshold
        """
        try:
            from mmpose.apis import MMPoseInferencer
            #pose_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
            #pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
            #det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
            #det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

            # Initialize the inferencer
            self.inferencer = MMPoseInferencer(
                pose2d='/projects/ext/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py',
                pose2d_weights='/projects/ext/models/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth',
                det_model='/projects/ext/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
                det_weights='/projects/ext/models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
                det_cat_ids=[0],  # Person class
                device=device
            )
            
            self.conf_threshold = conf_threshold
            self.tracker = SimpleTracker(max_disappeared=30, max_distance=100)
            
            logging.info(f"RTMPose inferencer initialized with model: {model_alias}")
            
        except ImportError as e:
            logging.error(f"Failed to import MMPose: {e}")
            logging.error("Please install mmpose: pip install mmpose")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to initialize inferencer: {e}")
            sys.exit(1)
    
    def process_frame(self, frame):
        """Process a single frame and return results"""
        
        # Run inference
        result_generator = self.inferencer(frame, return_vis=False)
        result = next(result_generator)
        #import pdb; pdb.set_trace()
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
                    #import pdb; pdb.set_trace()
                    valid_kpts = keypoints#[np.array(pred['keypoint_scores']) > 0.3]
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
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = RTMPoseTracker(
        model_alias=args.model,
        device=args.device,
        conf_threshold=args.conf
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