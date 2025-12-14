import torch
from monai.bundle.config_parser import ConfigParser

try:
    from .segresnet_ds import SegResNetDS
except:
    from segresnet_ds import SegResNetDS


def get_network(parser, config):
    in_channels = config["input_channels"]
    n_class = config["output_classes"]
    dsdepth = config["network"]["dsdepth"]
    init_filters = config["network"]["init_filters"]
    blocks_down = config["network"]["blocks_down"]
    norm = config["network"]["norm"]

    model = define_network(
        in_channels, n_class, init_filters, blocks_down, norm, dsdepth
    )

    return model


def define_network(
    in_channels,
    n_class,
    init_filters=32,
    blocks_down=[1, 2, 2, 4, 4, 4],
    norm="INSTANCE",
    dsdepth=3,
):
    model = SegResNetDS(
        init_filters=init_filters,
        blocks_down=blocks_down,
        norm=norm,
        in_channels=in_channels,
        out_channels=n_class,
        dsdepth=dsdepth,
    )
    # channels: midRT, preRT mask class1, preRT mask class2
    return model


if __name__ == "__main__":
    # model = define_network(in_channels=3, n_class=2).to("cuda")
    config = ConfigParser.load_config_file("../configs/hyper_parameters.yaml")
    model = get_network(config, config).to("cuda")
    # create a random input tensor

    x = torch.rand((1, 3, 192, 192, 96)).to("cuda")
    # 192, 96, 48, 24, 12, 6
    out = model(x)

    for i in out:
        print(i.shape)

    # print the number of trainable parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
