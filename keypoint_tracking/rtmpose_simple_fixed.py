#!/usr/bin/env python3
"""
RTMPose Tracker with Fixed Configuration

This version handles missing configuration files and provides proper fallbacks.
Works with local MMPose installations and handles registry issues.

Usage:
    python rtmpose_simple_fixed.py                    # Use webcam
    python rtmpose_simple_fixed.py video.mp4          # Use video file
"""

import os
import sys
import warnings
from pathlib import Path

# Add local libraries to path FIRST
current_dir = Path(__file__).parent
lib_paths = [
    current_dir / 'libs' / 'mmengine',
    current_dir / 'libs' / 'mmcv',
    current_dir / 'libs' / 'mmdetection', 
    current_dir / 'libs' / 'mmpose'
]

for path in lib_paths:
    if path.exists():
        sys.path.insert(0, str(path))
        print(f"Added to path: {path}")

# Set environment variables
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO)

def create_simple_pose_estimator():
    """Create a simple pose estimator without complex dependencies"""
    
    try:
        # Try the modern MMPose inferencer first
        from mmpose.apis import MMPoseInferencer
        
        # Use a simpler configuration approach
        print("Trying to create MMPose inferencer...")
        
        # Try different model specifications
        model_configs = [
            'human',  # Simple alias
            'rtmpose-m',  # Direct model name
            {'pose2d': 'human', 'det_model': 'human'}  # Explicit config
        ]
        
        for config in model_configs:
            try:
                print(f"Trying config: {config}")
                if isinstance(config, dict):
                    inferencer = MMPoseInferencer(**config, device='cpu')
                else:
                    inferencer = MMPoseInferencer(config, device='cpu')
                print("✅ MMPose inferencer created successfully!")
                return inferencer, 'mmpose'
            except Exception as e:
                print(f"Config {config} failed: {e}")
                continue
        
        raise Exception("All MMPose configs failed")
        
    except Exception as e:
        print(f"MMPose inferencer failed: {e}")
        print("Falling back to basic OpenCV pose estimation...")
        return create_opencv_fallback(), 'opencv'

def create_opencv_fallback():
    """Fallback to OpenCV DNN-based pose estimation"""
    
    class OpenCVPoseEstimator:
        def __init__(self):
            self.net = None
            self.setup_network()
        
        def setup_network(self):
            """Setup OpenCV DNN network for pose estimation"""
            try:
                # This is a placeholder - you would need to download pose models
                print("OpenCV pose estimation would require additional model files")
                print("For now, this will just detect people without pose estimation")
                
                # Initialize face cascade as a simple detector fallback
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
            except Exception as e:
                print(f"OpenCV setup failed: {e}")
                self.face_cascade = None
        
        def __call__(self, image, **kwargs):
            """Simple detection fallback"""
            results = {
                'predictions': [[]],
                'visualization': [image.copy()]
            }
            
            if self.face_cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                vis_image = image.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(vis_image, "Person Detected", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                results['visualization'] = [vis_image]
                
                # Create fake keypoints for detected faces
                for (x, y, w, h) in faces:
                    fake_keypoints = np.array([
                        [x + w//2, y + h//4, 0.9],  # nose
                        [x + w//3, y + h//4, 0.8],  # left eye  
                        [x + 2*w//3, y + h//4, 0.8], # right eye
                        [x + w//4, y + h//2, 0.7],   # left ear
                        [x + 3*w//4, y + h//2, 0.7], # right ear
                    ])
                    
                    results['predictions'][0].append({
                        'keypoints': fake_keypoints[:, :2],
                        'keypoint_scores': fake_keypoints[:, 2]
                    })
            
            return [results]  # Return as generator-like
    
    return OpenCVPoseEstimator()

def main():
    parser = argparse.ArgumentParser(description='Fixed RTMPose Demo')
    parser.add_argument('input', nargs='?', default=0, help='Video file or camera index')
    parser.add_argument('--save', help='Save output video')
    
    args = parser.parse_args()
    
    print("🔧 Setting up pose estimator...")
    
    # Create pose estimator
    pose_estimator, estimator_type = create_simple_pose_estimator()
    
    # Setup input
    try:
        input_source = int(args.input)
        source_name = f"Camera {input_source}"
    except ValueError:
        input_source = args.input
        source_name = Path(args.input).name
        if not Path(args.input).exists():
            print(f"Error: File '{args.input}' not found")
            return
    
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Cannot open {source_name}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing: {source_name} ({width}x{height} @ {fps:.1f} FPS)")
    print(f"Using estimator: {estimator_type}")
    print("\nControls: Space=Pause, Q=Quit, S=Save frame")
    
    # Setup video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving to: {args.save}")
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run pose estimation
                try:
                    if estimator_type == 'mmpose':
                        result_generator = pose_estimator(
                            frame, show=False, return_vis=True, radius=3, thickness=2
                        )
                        result = next(result_generator)
                        
                        if result['visualization']:
                            vis_frame = result['visualization'][0]
                        else:
                            vis_frame = frame.copy()
                        
                        num_people = len(result['predictions'][0]) if result['predictions'] else 0
                        
                    else:  # opencv fallback
                        result_list = pose_estimator(frame)
                        result = result_list[0] if result_list else {'visualization': [frame], 'predictions': [[]]}
                        vis_frame = result['visualization'][0]
                        num_people = len(result['predictions'][0])
                    
                    # Add info overlay
                    cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"People: {num_people}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Engine: {estimator_type}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Estimation error: {e}")
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, f"Error: {str(e)[:50]}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                frame_count += 1
                
                if writer:
                    writer.write(vis_frame)
            
            # Display frame
            cv2.imshow(f'Pose Estimation - {estimator_type}', vis_frame)
            
            # Handle input
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'PAUSED' if paused else 'PLAYING'}")
            elif key == ord('s'):
                save_path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, vis_frame)
                print(f"Saved: {save_path}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

if __name__ == '__main__':
    main()