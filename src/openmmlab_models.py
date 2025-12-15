import sys
import logging

import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np
from pathlib import Path

sys.path.append(str(Path('../libs') / 'mmengine'))
sys.path.append(str(Path('../libs') / 'mmcv'))
sys.path.append(str(Path('../libs') / 'mmdetection'))
sys.path.append(str(Path('../libs') / 'mmpose'))
from mmpose.apis import init_model
from mmdet.apis import init_detector

#pose2d = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
#pose_weights = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
#det_model = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
#det_weights = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

pose_model='../configs/openmmlab/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
pose_weights='../models/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth'
det_model='../configs/openmmlab/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
det_weights='../models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

for path in sys.path[-4:]+[pose_model, pose_weights, det_model, det_weights]:
    print(f"Resolved {Path(path).exists()}, {path}")
    

class TorchModelPose:
    def __init__(self, pose_model=pose_model, pose_weights=pose_weights, device='cpu'):
        # Build and load pose model directly from config and weights
        self.pose_model = init_model(
            config=pose_model,
            checkpoint=pose_weights,
            device=device
        )
        logging.info(f"RTMPose inferencer initialized")
        self.pose_input_shape = self.pose_model.head.input_size[::-1]
        self.pose_mean = np.array(self.pose_model.data_preprocessor.mean, dtype=np.float32).reshape(3)
        self.pose_std  = np.array(self.pose_model.data_preprocessor.std, dtype=np.float32).reshape(3)
    
    def get_params(self):
        return self.pose_input_shape, self.pose_mean, self.pose_std
    
    def _forward(self, input_tensor):
        input_tensor = torch.tensor(input_tensor).float()
        with torch.no_grad():
            pose_results = self.pose_model.forward(inputs = input_tensor, data_samples =  None, mode='tensor')
        return pose_results
        
class OnnxModelPose:
    def __init__(self, model_path, device='cpu'):
        # Setup ONNX Runtime providers
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]

        # Load pose estimation ONNX model
        self.pose_session = ort.InferenceSession(pose_weights.replace('.pth', '.onnx'), providers=providers)
        logging.info(f"Loaded pose estimation ONNX model from {pose_weights.replace('.pth', '.onnx')}")
        self.pose_input_name = self.pose_session.get_inputs()[0].name
        self.pose_input_shape = self.pose_session.get_inputs()[0].shape[2:]  # Get H, W
        self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        logging.info(f"Pose input: {self.pose_input_name}, shape: {self.pose_input_shape}")
    
    def get_params(self):
        return self.pose_input_shape, self.pose_mean, self.pose_std
    
    def _forward(self, input_tensor):
        pose_results = self.pose_session.run(None, {self.pose_input_name: input_tensor})
        for i in range(len(pose_results)):
            pose_results[i] = torch.tensor(pose_results[i])
        return pose_results

        
class TRTModelPose:
    def __init__(self, model_path, device='cpu'):
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
        self._pycuda_context = pycuda.autoinit.context  # Keep reference to avoid premature cleanup

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
            self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

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
        
        logging.info(f"Loaded pose estimation TensorRT model from {pose_weights.replace('.pth', '.engine')}")
        logging.info(f"Pose input shape: {self.pose_input_shape}")
    
    def __del__(self):
        """Cleanup TensorRT resources"""
        try:
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
    
    def get_params(self):
        return self.pose_input_shape, self.pose_mean, self.pose_std
    
    def _forward(self, input_tensor):
        pose_bsize = 4
        pose_results = []
        self._pycuda_context.push()
        for _itr in range(0, input_tensor.shape[0], pose_bsize):
            batch_data = input_tensor[_itr:_itr+pose_bsize]
            actual_batch = batch_data.shape[0]
            
            input_shape = (actual_batch, 3, 256, 192)
            self.pose_context.set_input_shape(self.pose_inputs[0]['name'], input_shape)
            
            for inp in self.pose_inputs:
                self.pose_context.set_tensor_address(inp['name'], int(inp['device']))
            for out in self.pose_outputs:
                self.pose_context.set_tensor_address(out['name'], int(out['device']))
            
            data_size = actual_batch * 3 * 256 * 192
            np.copyto(self.pose_inputs[0]['host'][:data_size], batch_data.ravel())
            cuda.memcpy_htod_async(self.pose_inputs[0]['device'], self.pose_inputs[0]['host'], self.pose_stream)
            
            self.pose_context.execute_async_v3(stream_handle=self.pose_stream.handle)
            
            inferred_output_shapes = []
            for output in self.pose_outputs:
                inferred_shape = self.pose_context.get_tensor_shape(output['name'])
                inferred_output_shapes.append(inferred_shape)
            
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
        return pose_results
    
class TorchModelDet:
    def __init__(self, det_model=det_model, det_weights=det_weights, device='cpu'):
        # Build and load detection model directly from config and weights
        self.det_model = init_detector(
                config=det_model,
                checkpoint=det_weights,
                device=device
            )
        self.det_input_shape = [416, 416]
        self.det_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.det_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.det_session = None
    
    def get_params(self):
        return self.det_input_shape, self.det_mean, self.det_std
    
    def _forward(self, frame_rsz):
        frame_rsz = torch.tensor(frame_rsz).float()
        with torch.no_grad():
            det_result = self.det_model(inputs= frame_rsz, data_samples= None, mode='tensor')
            class_scores = det_result[0]
            bbox_preds   = det_result[1]
        return bbox_preds, class_scores

class OnnxModelDet:
    def __init__(self, model_path, device='cpu'):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
            
        self.det_session = ort.InferenceSession(det_weights.replace('.pth', '.onnx'), providers=providers)
        logging.info(f"Loaded detection ONNX model from {det_weights.replace('.pth', '.onnx')}")
        
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_input_shape = self.det_session.get_inputs()[0].shape[2:]
        logging.info(f"Detection input: {self.det_input_name}, shape: {self.det_input_shape}")
        
        self.det_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.det_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    def get_params(self):
        return self.det_input_shape, self.det_mean, self.det_std
    
    def _forward(self, frame_rsz):
        det_result = self.det_session.run(None, {self.det_input_name: frame_rsz})
        class_scores = det_result[:3]
        bbox_preds  = det_result[3:]
        return bbox_preds, class_scores

class TRTModelDet:
    def __init__(self, model_path, device='cpu'):
        self._pycuda_context = pycuda.autoinit.context  # Keep reference to avoid premature cleanup
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
            
            logging.info("TensorRT resources cleaned up")
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")
    
    def get_params(self):
        return self.det_input_shape, self.det_mean, self.det_std
    
    def _forward(self, frame_rsz):
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
        return bbox_preds, class_scores