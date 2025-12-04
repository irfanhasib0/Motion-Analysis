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

# Register all modules in mmdet into the registries

if __name__ == "__main__":

    if sys.argv[1] == 'det':
        register_det_modules()
        det_cfg = Config.fromfile("../configs/openmmlab/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py")
        model = DET_MODELS.build(det_cfg.model)
        det_checkpoint = torch.load("../models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth", map_location="cpu")
        model.load_state_dict(det_checkpoint['state_dict'], strict=False)
        model.eval()

    if sys.argv[1] == 'pose':
        register_pose_modules()
        pose_cfg = Config.fromfile("../configs/openmmlab/configs_pose/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py")
        model = POSE_MODELS.build(pose_cfg.model)
        pose_checkpoint = torch.load("../models/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth", map_location="cpu")
        model.load_state_dict(pose_checkpoint['state_dict'], strict=False)
        model.eval()

    # from mmengine.runner import load_checkpoint
    # load_checkpoint(model, "../models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth", map_location="cpu")

    dummy = torch.randn(1, 3, 256, 192)
    torch.onnx.export(
        model,
        dummy,
        f"model_{sys.argv[1]}.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
    )
