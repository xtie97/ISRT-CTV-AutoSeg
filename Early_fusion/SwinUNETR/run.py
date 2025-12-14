import os
from monai.utils import optional_import
from src.segmenter import main
import argparse
import shutil
import yaml
from collections import OrderedDict


class OrderedLoader(yaml.SafeLoader):
    pass


def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
)


class OrderedDumper(yaml.SafeDumper):
    pass


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


OrderedDumper.add_representer(OrderedDict, dict_representer)


def modify_yaml():
    # CLI argument setup
    # modify yaml file
    input_yaml = "configs/hyper_parameters.yaml"
    with open(input_yaml, "r") as file:
        # config = yaml.safe_load(file)
        config = yaml.load(file, Loader=OrderedLoader)
    is_infer = config["infer"]["enabled"]
    if is_infer:
        print("Inference mode is enabled.")
    else:
        return False

    out_yaml = "configs/hyper_parameters_copy.yaml"
    if not os.path.exists(out_yaml):
        shutil.copy(input_yaml, out_yaml)

    parser = argparse.ArgumentParser(
        description="Run model training with configurable parameters."
    )
    parser.add_argument(
        "--fold",
        type=int,
        nargs="+",
        default=[3],
        help="Fold numbers (e.g., --fold 0 1 2 3 4)",
    )

    args = parser.parse_args()
    fold_nums = args.fold
    print("=====================================")

    for fold_num in fold_nums:
        current_fold = config["ckpt_path"].split("_")[-1][:-1]
        # print("Current fold:", current_fold)
        config["ckpt_path"] = config["ckpt_path"].replace(
            "_" + current_fold, f"_f{fold_num}"
        )
        current_fold = config["infer"]["output_path"].split("_")[-1][:-1]
        config["infer"]["output_path"] = config["infer"]["output_path"].replace(
            "_" + current_fold, f"_f{fold_num}"
        )

        if os.path.exists(input_yaml):
            os.remove(input_yaml)
        with open(input_yaml, "w") as file:
            yaml.dump(config, file, Dumper=OrderedDumper, default_flow_style=False)

        run_model()
    return True


def run_model():
    fire, fire_is_imported = optional_import("fire")
    if fire_is_imported:
        fire.Fire(main)


if __name__ == "__main__":

    run_inf = modify_yaml()
    if not run_inf:
        run_model()
