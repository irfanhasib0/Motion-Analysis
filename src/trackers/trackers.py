import sys
import cv2
import numpy as np
    
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
    
class ByteTracker:
    """ByteTracker implementation for multi-object tracking"""
    
    sys.path.insert(0,'../libs/ByteTrack')
    try:
        from yolox.tracker.byte_tracker import BYTETracker, STrack
        from yolox.tracker.basetrack import BaseTrack, TrackState
    except ImportError:
        print("ByteTrack dependencies not found.")

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        """
        Initialize ByteTracker
        
        Args:
            track_thresh: Detection confidence threshold for track initialization
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
            frame_rate: Frame rate of the video
        """
        
        self.BYTETracker = BYTETracker
        self.STrack = STrack
        self.BaseTrack = BaseTrack
        self.TrackState = TrackState
        
        # Create args object for BYTETracker
        class Args:
            def __init__(self):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh
                self.mot20 = False  # Not using MOT20 dataset format
        
        self.args = Args()
        self.tracker = self.BYTETracker(self.args, frame_rate=frame_rate)
        self.frame_id = 0
    # nbyte_tracer.py: np.float -> np.float32
    # matching.py : cython_bbox -> numpy implementation
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'keypoints': Nx2 array of keypoint coordinates
                - 'keypoint_scores': N array of keypoint scores
                - 'centroid': (x, y) tuple
                - Optional 'score': detection confidence (will use mean keypoint score if not provided)
        
        Returns:
            Dict mapping track_id to detection dict
        """
        self.frame_id += 1
        
        if len(detections) == 0:
            online_targets = self.tracker.update(
                np.empty((0, 5)),
                [480, 640],
                [480, 640]
            )
            return {}
        
        # Convert detections to format expected by ByteTrack: [x1, y1, x2, y2, score]
        detection_array = []
        for det in detections:
            bbox = det['bbox']#[0] if isinstance(det['bbox'], list) else det['bbox']
            
            # Calculate detection score from keypoint scores if not provided
            if 'score' in det:
                score = det['score']
            else:
                score = np.mean(det['keypoint_scores'][det['keypoint_scores'] > 0.1])
                score = max(score, 0.3)  # Ensure minimum score
            
            detection_array.append([bbox[0], bbox[1], bbox[2], bbox[3], score])
        
        detection_array = np.array(detection_array)
        
        # Get image info (use bbox bounds as approximate image size)
        img_info = [
            np.max(detection_array[:, 3]),  # max y
            np.max(detection_array[:, 2])   # max x
        ]
        img_size = img_info
        
        # Update tracker
        online_targets = self.tracker.update(
            detection_array,
            img_info,
            img_size
        )
        
        # Map tracks back to detections
        result = {}
        for track in online_targets:
            track_id = track.track_id
            tlbr = track.tlbr  # [x1, y1, x2, y2]
            
            # Find matching detection using IOU or centroid distance
            best_match_idx = None
            best_iou = 0.0
            
            for idx, det in enumerate(detections):
                det_bbox = det['bbox']#[0] if isinstance(det['bbox'], list) else det['bbox']
                
                # Calculate IOU
                x1 = max(tlbr[0], det_bbox[0])
                y1 = max(tlbr[1], det_bbox[1])
                x2 = min(tlbr[2], det_bbox[2])
                y2 = min(tlbr[3], det_bbox[3])
                
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1])
                area2 = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            # Add matched detection to results
            if best_match_idx is not None and best_iou > 0.3:
                matched_det = detections[best_match_idx].copy()
                matched_det['track_id'] = track_id
                matched_det['track_score'] = track.score
                result[track_id] = matched_det
        
        return result
    
    def reset(self):
        """Reset tracker state"""
        self.tracker = self.BYTETracker(self.args, frame_rate=30)
        self.frame_id = 0
