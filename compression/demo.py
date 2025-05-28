import argparse
import os
import sys

import torch
from nni.compression.pytorch.utils import count_flops_params
from nni.compression.pytorch.pruning import FPGMPruner
from nni.compression.pytorch.speedup.v2 import ModelSpeedup

from models.srfnet import SRFNet, SRFBasicBlock

sys.path.append('../')


def prune_and_export_model(checkpoints_path, output_onnx_path, pruning_ratio):
    config_list = [{'sparsity': pruning_ratio,
                    'op_types': ['Conv2d'],
                    'op_names': ['layer1.0', 'layer2.0.conv1', 'layer2.0.conv2', 'layer3.0.conv1', 'layer3.0.conv2',
                                 'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2',
                                 'layer4.2.conv1', 'layer4.2.conv2', 'layer5.0.conv1', 'layer5.0.conv2',
                                 'shallow_to_deep_conv']
                    }]

    model = SRFNet(SRFBasicBlock, [1, 1, 3, 1]).cpu()

    state_dict = torch.load(checkpoints_path)
    model.load_state_dict(state_dict)

    dummy_input = torch.rand(1, 1, 224, 224).cpu()

    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    print("Original Model:")
    print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    pruner = FPGMPruner(model, config_list)

    _, masks = pruner.compress()

    pruner._unwrap_model()

    ModelSpeedup(model, dummy_input, masks).speedup_model()

    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    print("Pruned Model:")
    print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    torch.onnx.export(model, dummy_input, output_onnx_path, input_names=['input'],
                      output_names=['output'], verbose=True, opset_version=11)

    print("Successfully exported ONNX model!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=False,
                        default='',
                        help='path of the checkpoint that will be pruned and exported')
    parser.add_argument('--output-path', type=str, required=False, default='',
                        help='path for saving the ONNX model')
    parser.add_argument('--pruning-ratio', type=int, required=False, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit()

    prune_and_export_model(
        checkpoints_path=args.ckpt_path,
        output_onnx_path=args.output_path,
        pruning_ratio=args.pruning_ratio
    )
