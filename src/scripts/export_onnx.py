import torch
import sys
sys.path.append('../libs/mmengine')
sys.path.append('../libs/mmcv')
sys.path.append('../libs/mmdetection')
sys.path.append('../libs/mmpose')

from mmengine.runner import Runner
from mmengine.config import Config
from mmdet.registry import MODELS as DET_MODELS
from mmpose.registry import MODELS as POSE_MODELS
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.utils import register_all_modules as register_pose_modules
import tensorrt as trt

# Register all modules in mmdet into the registries

def export_tensorrt(onnx_path, engine_path, input_shape, fp16=False, int8=False, max_batch_size=1):
        """
        Export ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            input_shape: Tuple of (C, H, W) for input shape
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
            max_batch_size: Maximum batch size
        """
        
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        # Always use explicit batch for TensorRT 8.x+
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
        
        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Get input tensor name
        input_name = network.get_input(0).name
        
        # Create optimization profile
        profile = builder.create_optimization_profile()
        
        if max_batch_size > 1:
            # Dynamic batch: min=1, opt=1, max=max_batch_size
            min_shape = (1, *input_shape)
            opt_shape = (1, *input_shape)
            max_shape = (max_batch_size, *input_shape)
        else:
            # Fixed batch: all shapes are the same
            min_shape = (1, *input_shape)
            opt_shape = (1, *input_shape)
            max_shape = (1, *input_shape)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build and serialize engine
        serialized_engine = builder.build_serialized_network(network, config)
        
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"TensorRT engine saved to {engine_path}")

#det_model_path = "../models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
#det_cfg_path   = "../configs/openmmlab/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py"

#pose_model_path = "../models/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"
#pose_cfg_path   = "../configs/openmmlab/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py"

det_model_path    = "../models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
det_cfg_path      = "../configs/openmmlab/configs_det/rtmdet/rtmdet_m_8xb32-300e_coco.py"
pose_model_path   = "../models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
pose_cfg_path     = "../configs/openmmlab/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"

# Usage python3 export_onnx.py onnx det
#       python3 export_onnx.py onnx pose
#       python3 export_onnx.py trt det
#       python3 export_onnx.py trt pose

if __name__ == "__main__":
    mode = sys.argv[1]  # 'onnx' or 'trt'
    if mode == 'onnx':
        if sys.argv[2] == 'det':
            register_det_modules()
            det_cfg = Config.fromfile(det_cfg_path)
            model = DET_MODELS.build(det_cfg.model)
            model_save_path = det_model_path.replace('.pth', '.onnx')
            det_checkpoint = torch.load(det_model_path, map_location="cpu")
            model.load_state_dict(det_checkpoint['state_dict'], strict=False)
            model.eval()
            inp = torch.randn(1, 3, 416, 416)

        if sys.argv[2] == 'pose':
            register_pose_modules()
            pose_cfg = Config.fromfile(pose_cfg_path)
            model = POSE_MODELS.build(pose_cfg.model)
            model_save_path = pose_model_path.replace('.pth', '.onnx')
            pose_checkpoint = torch.load(pose_model_path, map_location="cpu")
            model.load_state_dict(pose_checkpoint['state_dict'], strict=False)
            model.eval()
            inp = torch.randn(1, 3, 256, 192)
        
        torch.onnx.export(
            model,
            inp,
            model_save_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
        )
    
    if mode == 'trt':
        # Define input shapes based on model type
        if sys.argv[2] == 'det':
            onnx_path   = det_model_path.replace('.pth', '.onnx')
            engine_path = det_model_path.replace('.pth', '.engine')
            input_shape = (3, 416, 416)  # Detection model
            max_batch_size = 1
        elif sys.argv[2] == 'pose':
            onnx_path = pose_model_path.replace('.pth', '.onnx')
            engine_path = pose_model_path.replace('.pth', '.engine')
            input_shape = (3, 256, 192)  # Pose model
            max_batch_size = 4
        else:
            raise ValueError(f"Unknown model type: {sys.argv[1]}")
        
        export_tensorrt(onnx_path, engine_path, input_shape, fp16=True, max_batch_size=max_batch_size)