import torch
import numpy as np
import monai
from monai.bundle.config_parser import ConfigParser

try:
    from swinunetr import SwinUNETR
except:
    from .swinunetr import SwinUNETR
# from modules import SwinTransformer


def define_model(img_size, in_channels, n_class, deep_supr_num, use_checkpoint=True):
    # get the multiplication of img_size
    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=n_class,
        feature_size=[32, 48],
        embed_dim=64,
        num_heads=[4, 8, 16, 32],  
        spatial_dims=3,
        deep_supr_num=deep_supr_num,
        use_checkpoint=use_checkpoint,
    )
    return model


def get_network(parser, config):
    spatial_dims = 3
    in_channels = config["input_channels"]
    out_channels = config["output_classes"]
    patch_size = config["roi_size"]
    try:
        deep_supr_num = config["network"]["dsdepth"]
    except:
        deep_supr_num = 3
        print("deep_supr_num is set to 3")

    try:
        use_checkpoint = config["network"]["use_checkpoint"]
    except:
        use_checkpoint = True
        print("use_checkpoint is set to True")

    net = define_model(
        patch_size, in_channels, out_channels, deep_supr_num, use_checkpoint
    )

    return net


if __name__ == "__main__":
    # get the multiplication of img_size
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = [192, 192, 96]
    in_channels = 3
    n_class = 2
    # model = get_network(img_size, in_channels, n_class).to("cuda")

    config = ConfigParser.load_config_files("../configs/hyper_parameters.yaml")

    model = get_network(config, config).to("cuda")
    # print number of parameters
    print(
        f"Number of params in the model is {sum([np.prod(p.size()) for p in model.parameters()])}"
    )

    # create a dummy input
    input = torch.randn(1, 3, *img_size).to("cuda")
    # run the model
    out = model(input)
    for i in out:
        print(i.size())

    # save the model as .pt
    torch.save(model.state_dict(), "model.pt")
    # # save the model as .pth
    # torch.save(model, "model.pth")
