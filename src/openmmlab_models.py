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
#from mmdet.apis import init_detector

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
    
class TorchModel:
    def __init__(self, model=None, weights=None, input_shape = None, device='cpu'):
        assert model is not None, "Model config path must be provided"
        assert weights is not None, "Model weights path must be provided"
        assert input_shape is not None, "Model input shape must be provided"
        # Build and load detection model directly from config and weights
        self.model = init_model(
                config= model,
                checkpoint=weights,
                device=device
            )
        self.model_input_shape = input_shape
        #self.det_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        #self.det_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.mean = np.array(self.model.data_preprocessor.mean, dtype=np.float32).reshape(3)
        self.std  = np.array(self.model.data_preprocessor.std, dtype=np.float32).reshape(3)
        
    def get_params(self):
        return self.model_input_shape, self.mean, self.std
    
    def _forward(self, frame_rsz):
        frame_rsz = torch.tensor(frame_rsz).float()
        with torch.no_grad():
            results = self.model(inputs= frame_rsz, data_samples= None, mode='tensor')
            # det  -> [class_scores, bbox_preds]
            # pose -> [heatmaps, tags]
        return results


    
class OnnxModelBase:
    def __init__(self, weights = None,   device='cpu'):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
            
        self.session = ort.InferenceSession(weights.replace('.pth', '.onnx'), providers=providers)
        logging.info(f"Loaded detection ONNX model from {det_weights.replace('.pth', '.onnx')}")
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        logging.info(f"Detection input: {self.input_name}, shape: {self.input_shape}")
        
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    def get_params(self):
        return self.input_shape, self.mean, self.std
    
    def _forward(self, frame_rsz):
        results = self.session.run(None, {self.input_name: frame_rsz}) 
        return results

class OnnxModelDet(OnnxModelBase):
    def __init__(self, weights = None,   device='cpu'):
        super().__init__(weights, device=device)
    
    def _forward(self, input_tensor):
        det_results = super()._forward(input_tensor)
        class_scores = det_results[:3]
        bbox_preds  = det_results[3:]
        return [class_scores, bbox_preds]

class OnnxModelPose(OnnxModelBase):
    def __init__(self, weights = None, device='cpu'):
        super().__init__(weights, device=device)

    def _forward(self, input_tensor):
        pose_results = super()._forward(input_tensor)
        for i in range(len(pose_results)):
            pose_results[i] = torch.tensor(pose_results[i])
        return pose_results

class TRTModelBase:
    def __init__(self, model_weights = None, batch_size=1):
        assert model_weights is not None, "Model weights path must be provided"

        self.batch_size = batch_size

        self._pycuda_context = pycuda.autoinit.context  # Keep reference to avoid premature cleanup
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)    
        with open(model_weights.replace('.pth', '.engine'), 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.model_engine = runtime.deserialize_cuda_engine(engine_data)
        self.model_context = self.model_engine.create_execution_context()
        
        # Allocate buffers
        self.model_inputs = []
        self.model_outputs = []
        self.model_stream = cuda.Stream()
        
        for binding in self.model_engine:
            shape = self.model_engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.model_engine.get_tensor_dtype(binding))
            
            # Replace dynamic dimensions (-1) with batch size 1 for allocation
            alloc_shape = tuple(self.batch_size if dim == -1 else dim for dim in shape)
            size = trt.volume(alloc_shape)
            logging.info(f"Allocating {binding}: shape={shape}, alloc_shape={alloc_shape}, size={size}, dtype={dtype}")
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Check if input or output using tensor mode (modern TensorRT API)
            tensor_mode = self.model_engine.get_tensor_mode(binding)
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.model_inputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': alloc_shape})
                self.model_input_shape = alloc_shape[2:]
            else:
                self.model_outputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': alloc_shape})
            
            # Set input shape for dynamic batch engines
            self.model_context.set_input_shape(self.model_inputs[0]['name'], alloc_shape)
            
            # Set tensor addresses AFTER setting input shape (required for dynamic shapes)
            for inp in self.model_inputs:
                self.model_context.set_tensor_address(inp['name'], int(inp['device']))
            for out in self.model_outputs:
                self.model_context.set_tensor_address(out['name'], int(out['device']))
         
        self.model_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.model_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        logging.info(f"Loaded detection TensorRT model from {model_weights.replace('.pth', '.engine')}")
        logging.info(f"Detection input shape: {self.model_input_shape}")
        
    def __del__(self):
        """Cleanup TensorRT resources"""
        # Free detection buffers
        if hasattr(self, 'model_inputs'):
            for inp in self.model_inputs:
                if 'device' in inp and inp['device']:
                    inp['device'].free()
        if hasattr(self, 'model_outputs'):
            for out in self.model_outputs:
                if 'device' in out and out['device']:
                    out['device'].free()
        cuda.Context.pop()
        logging.info("TensorRT resources cleaned up")
        
    def get_params(self):
        return self.model_input_shape, self.model_mean, self.model_std
    
    def _forward(self, input_tensor):
        class_scores = []
        bbox_preds = []
        self._pycuda_context.push()
        # Copy input data to device
        np.copyto(self.model_inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.model_inputs[0]['device'], self.model_inputs[0]['host'], self.model_stream)
        
        # Run inference
        self.model_context.execute_async_v3(stream_handle=self.model_stream.handle)
        
        # Get inferred output shapes
        inferred_output_shapes = []
        for output in self.model_outputs:
            inferred_shape = self.model_context.get_tensor_shape(output['name'])
            inferred_output_shapes.append(inferred_shape)
        
        # Copy output data back to host using inferred shapes
        for output, inferred_shape in zip(self.model_outputs, inferred_output_shapes):
            output_size = np.prod(inferred_shape)
            cuda.memcpy_dtoh_async(output['host'][:int(output_size)], output['device'], self.model_stream)
        
        # Synchronize the stream
        self.model_stream.synchronize()

        return self.model_outputs, inferred_output_shapes
    
class TRTModelDet(TRTModelBase):
    def __init__(self, det_weights = det_weights, device='cpu'):
        super().__init__(model_weights=det_weights, batch_size=1)
        
    def get_params(self):
        return self.model_input_shape, self.model_mean, self.model_std
    
    def _forward(self, frame_rsz):
        det_outputs, inferred_output_shapes = super()._forward(frame_rsz)
        class_scores = []
        bbox_preds = []
        # Retrieve outputs using inferred shapes
        for output, inferred_shape in zip(det_outputs, inferred_output_shapes):
            output_array = output['host'][:np.prod(inferred_shape)].reshape(inferred_shape)
            if output_array.shape[1] == 4:
                bbox_preds.append(torch.from_numpy(output_array))
            else:
                class_scores.append(torch.from_numpy(output_array))
        return [class_scores, bbox_preds]
    
class TRTModelPose(TRTModelBase):
    def __init__(self, model_path, device='cpu'):
        super().__init__(model_weights=pose_weights, batch_size= 4)
    
    def _forward(self, input_tensor):
        pose_results = []
        self._pycuda_context.push()
        input_tensor_padded =np.zeros([self.batch_size] + list(input_tensor.shape[1:]), dtype=np.float32)
        for _itr in range(0, input_tensor.shape[0], self.batch_size):
            input_tensor_padded[:min(self.batch_size, input_tensor.shape[0] - _itr)] = input_tensor[_itr:_itr+self.batch_size]
            model_outputs, inferred_output_shapes = super()._forward(input_tensor_padded)
            
            # Retrieve outputs using inferred shapes
            batch_results = []
            for output, inferred_shape in zip(model_outputs, inferred_output_shapes):
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
        return pose_results