import os
import torch
from torch import nn
from monai.bundle.config_parser import ConfigParser
import numpy as np
from dynamic_network_architectures.architectures.unet import (
    ResidualEncoderUNet,
    ResidualUNet,
)


def get_kernels_strides(sizes, spacings):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    # sizes, spacings = patch_size[task_id], spacing[task_id]
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


class myUNet(nn.Module):
    def __init__(self, parser, config, is_resunet=False):
        super(myUNet, self).__init__()
        spatial_dims = 3
        default_blocks_per_stage_encoder = (1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4)

        in_channels = config["input_channels"]
        out_channels = config["output_classes"]
        patch_size = config["roi_size"]
        spacing = config["resample_resolution"]
        norm_name = config.get("norm", "instance")
        deep_supr_num = config.get("dsdepth", 3)
        self.deep_supr_num = deep_supr_num

        kernels, strides = get_kernels_strides(patch_size, spacing)
        print(f"Kernels: {kernels}, Strides: {strides}")
        num_stages = [
            not np.all(np.atleast_1d(stride) == 1) for stride in strides
        ].count(True) + 1

        use_nnunet_filters = config.get("use_nnunet_filters", True)
        if use_nnunet_filters:
            filters = [
                min(2 ** (5 + i), 320 if spatial_dims == 3 else 512)
                for i in range(len(strides))
            ]
        else:
            filters = [64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(strides)]

        print(f"Filters: {filters}")

        if is_resunet:
            self.net = ResidualUNet(
                input_channels=in_channels,
                num_classes=out_channels,
                n_stages=num_stages,
                features_per_stage=filters,
                conv_op=nn.Conv3d,
                kernel_sizes=kernels,
                strides=strides,
                n_blocks_per_stage=default_blocks_per_stage_encoder[:num_stages],
                n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={},
                dropout_op=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={"inplace": True},
                deep_supervision=True,
            )
        else:
            self.net = ResidualEncoderUNet(
                input_channels=in_channels,
                num_classes=out_channels,
                n_stages=num_stages,
                features_per_stage=filters,
                conv_op=nn.Conv3d,
                kernel_sizes=kernels,
                strides=strides,
                n_blocks_per_stage=default_blocks_per_stage_encoder[:num_stages],
                n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={},
                dropout_op=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={"inplace": True},
                deep_supervision=True,
            )

    def forward(self, x):
        outputs = self.net(x)

        if not self.training or len(outputs) == 1:
            return outputs[0]

        return outputs[: self.deep_supr_num + 1]


def get_network(parser, config):
    return myUNet(parser, config, is_resunet=True)


if __name__ == "__main__":
    config = ConfigParser.load_config_files("configs/hyper_parameters.yaml")
    model = get_network(config, config).to("cuda")
    # print all the layers
    # print(model)

    input_data = torch.randn(2, 5, 192, 192, 96).to("cuda")
    output_data = model(input_data)
    for i in output_data:
        print(i.size())
    # print the number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
