"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import onnx
import onnxsim

sys.path.insert(0, '../libs/DEIMv2')

import torch
import torch.nn as nn
from engine.core import YAMLConfig
#onnxsim
#faster_coco_eval
#calflops
config='../libs/DEIMv2/configs/deimv2/deimv2_dinov3_s_coco.yml'
resume='../models/deimv2_dinov3_s_coco.pth'
opset=17
check=False
simplify=False

cfg = YAMLConfig(config, resume=resume)

if 'HGNetv2' in cfg.yaml_cfg:
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False

if resume:
    checkpoint = torch.load(resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

class Model(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

model = Model()

img_size = cfg.yaml_cfg["eval_spatial_size"]
print(img_size)
data = torch.rand(2, 3, *img_size)
size = torch.tensor([img_size])
_ = model(data, size)

dynamic_axes = {
    'images': {0: 'N', },
    'orig_target_sizes': {0: 'N'}
}

output_file = resume.replace('.pth', '.onnx') if resume else 'model.onnx'

torch.onnx.export(
    model,
    (data, size),
    output_file,
    input_names=['images', 'orig_target_sizes'],
    output_names=['labels', 'boxes', 'scores'],
    dynamic_axes=dynamic_axes,
    opset_version=opset,
    verbose=False,
    do_constant_folding=True,
)

if check:
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    print('Check export onnx model done...')

if simplify:
    dynamic = True
    # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
    input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
    onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
    onnx.save(onnx_model_simplify, output_file)
    print(f'Simplify onnx model {check}...')
    