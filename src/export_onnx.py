import torch
import sys
sys.path.append('/projects/ext/libs/mmengine')
from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile("/projects/ext/configs_det/rtmdet/rtmdet_tiny_8xb32-300e_coco.py")
runner = Runner.from_cfg(cfg)
model = runner.build_model(cfg.model)
checkpoint = torch.load("/projects/ext/models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth", map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

dummy = torch.randn(1, 3, 256, 192)
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
    dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
)
