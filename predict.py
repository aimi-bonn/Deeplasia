import os

import logging.config
import pandas as pd
import torch

from lib.utils.log import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lib.datasets import InferenceDataset
from lib.models import BoneAgeModel
from lib import testing
import re


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckp_path", type=str)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="CNN backbone for the model. If not provided attempted to be inferred from the ckp path",
    )

    # inference options
    parser.add_argument(
        "--no_test_tta_rot",
        action="store_true",
        help="disable test time augmentation (rotations) for test set",
    )
    parser.add_argument(
        "--train_tta_rot",
        action="store_true",
        help="enable test time augmentation (rotations) for training set",
    )
    parser.add_argument(
        "--no_test_tta_flip",
        action="store_true",
        help="disable test time augmentation (flips) for test set",
    )
    parser.add_argument(
        "--train_tta_flip",
        action="store_true",
        help="enable test time augmentation (flips) for training set",
    )
    parser.add_argument(
        "--no_regression",
        action="store_true",
        help="disable regression correction of bone age predictions",
    )

    # data set stuff (only relevant option)
    parser.add_argument("--annotation_csv", type=str, default="data/annotation.csv")
    parser.add_argument(
        "--split_csv", type=str, default="data/splits/rsna_original.csv"
    )
    parser.add_argument("--split_column", type=str, default="")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--img_dir", type=str, default="../data/annotated/")
    parser.add_argument("--mask_dirs", nargs="+", default=["../data/masks/fscnn_cos"])
    parser.add_argument("--input_size", nargs="+", default=[1, 512, 512], type=int)
    parser.add_argument("--image_norm_method", type=str, default="zscore")
    parser.add_argument("--mask_crop_size", type=float, default=-1)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--rotation_angle", type=float, default=0)
    parser.add_argument("--source_col", type=str, default="image_source")

    # other options
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="predictions_results.csv")
    parser.add_argument(
        "--name",
        default=None,
        type=str,
        help="name to mark the model. If None default name stored in the ckp is used.",
    )

    parser = pl.Trainer.add_argparse_args(parser)
    return parser


def main():
    logger = logging.getLogger()
    parser = create_parser()
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(
        args, checkpoint_callback=False, logger=False
    )
    output_dir = (
        args.output_dir
        if args.output_dir
        else re.match(r".*/(version|split)_\d*", args.ckp_path)[0]
    )
    if "highRes" in args.ckp_path:
        args.input_size = [1, 1024, 1024]
        logger.info(
            "changed input size to 1024 as 'highRes' was detected in the ckp path"
        )
    if not args.mask_dirs[0]:
        args.mask_dirs = []
    logger.info(f"using masks from {args.mask_dirs}")

    loader = DataLoader(
        InferenceDataset(
            annotation_df=args.annotation_csv,
            split_path=args.split_csv,
            split_column=args.split_column,
            split_name=args.split_name,
            img_dir=args.img_dir,
            mask_dirs=args.mask_dirs,
            norm_method=args.image_norm_method,
            input_size=args.input_size,
            mask_crop_size=args.mask_crop_size,
            flip=args.flip,
            rotation_angle=args.rotation_angle,
        ),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    if "effnet" in args.ckp_path:
        args.backbone = (
            "efficientnet-b4" if "effnet-b4" in args.ckp_path else "efficientnet-b0"
        )
    model = BoneAgeModel.load_from_checkpoint(args.ckp_path)
    outputs = trainer.predict(model=model, dataloaders=loader)

    y = torch.concat([o["y"] for o in outputs]) if "y" in outputs[0].keys() else None
    names = [
        val.split("/")[-1]
        for sublist in [o["image_path"] for o in outputs]
        for val in sublist
    ]
    y_hat_out = torch.concat([o["y_hat"] for o in outputs])
    sex = torch.concat([o["sex"] for o in outputs])
    sex_hat = torch.concat([o["sex_hat"] for o in outputs])

    df = pd.DataFrame(
        {
            "image_ID": names,
            "sex": sex.squeeze(),
            "y_hat": y_hat_out.squeeze(),
            "sex_hat": sex_hat.squeeze(),
        }
    )
    if y is not None:
        df["y"] = y.squeeze()
    df.to_csv(args.output_path)
    logger.info(f"saved to {args.output_path}")


if __name__ == "__main__":
    main()
