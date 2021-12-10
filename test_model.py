import logging, logging.config
from lib.utils import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

import sys, pickle, os, yaml
from argparse import ArgumentParser

sys.path.append("..")

from lib import utils, testing
from lib.models import *


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckp_path", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="dbam_efficientnet-b0")
    parser.add_argument("--input_height", type=int, default=512)
    parser.add_argument("--input_width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--data_dir", type=str, default="../data/annotated/rsna_bone_age"
    )
    parser.add_argument("--mask_dirs", nargs="+", default=None)
    parser.add_argument(
        "--cache_data",
        action="store_true",
        help="cache images in RAM (Note: takes more than 10GB of RAM)",
    )
    parser.add_argument(
        "--img_norm_method",
        type=str,
        default=None,
        help="if None model method will be used, use only to overwrite the model's settings",
    )
    parser = testing.add_eval_args(parser)
    args = parser.parse_args()

    utils.log_system_info(logger)
    testing.evaluate_bone_age_model(args.ckp_path, args, args.output_dir)


if __name__ == "__main__":
    main()
