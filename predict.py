import os

import logging.config
import pandas as pd
import torch
import yaml

from lib.utils.log import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lib.datasets import InferenceDataset
from lib.models import BoneAgeModel
from lib.legacy import from_checkpoint as load_legacy_model
from lib import testing
import re

import warnings

warnings.filterwarnings("ignore")


def main():
    logger = logging.getLogger()
    parser = create_parser()
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(
        args, checkpoint_callback=False, logger=False
    )

    loader = create_loader(args, logger)

    if args.legacy_ckp:
        model = load_legacy_model(args)
    else:
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
    if not args.legacy_ckp:
        sex_hat = torch.concat([o["sex_hat"] for o in outputs])

    # Note that the outputs are z-scores
    # they need to be re-transformed (and regression-corrected) to get the real age

    df = {
        "image_ID": names,
        "sex": sex.squeeze(),
        "y_hat": y_hat_out.squeeze(),
    }
    if not args.legacy_ckp:
        df = df | {"sex_hat": sex_hat.squeeze()}
    df = pd.DataFrame(df)

    if args.legacy_ckp:
        df["pred_" "bone_age"] = rescale_prediction(df["y_hat"], args.ckp_path)

    if y is not None:
        df["y"] = y.squeeze()
    df.to_csv(args.output_path, index=False)
    logger.info(f"saved to {args.output_path}")


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckp_path", type=str)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="CNN backbone for the model. If not provided attempted to be inferred from the ckp path",
    )
    parser.add_argument(
        "--legacy_ckp",
        action="store_true",
        help="use legacy ckp format (checkpoints from before 03/22)",
    )

    # data set stuff (only relevant options)
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

    # other options
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--output_path", type=str, default="output/predictions_results.csv"
    )

    # store activations
    parser.add_argument("--store_activations", action="store_true")
    # TODO

    parser = pl.Trainer.add_argparse_args(parser)
    return parser


def create_loader(args, logger):
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
            split_df=args.split_csv,
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
    return loader


def rescale_prediction(y_hat, ckp_path, params_path="data/parameters.yml"):
    with open(params_path, "r") as stream:
        cor_params = yaml.safe_load(stream)

    def cor_prediction_bias(yhat, slope, intercept):
        """corrects model predictions (yhat) for linear bias (defined by slope and intercept)"""
        return yhat - (yhat * slope + intercept)

    age_mean, age_sd = cor_params["age_mean"], cor_params["age_sd"]
    y_hat = y_hat * age_sd + age_mean

    ckp_path = ckp_path.split("/")[-1].split(".")[0]
    slope = cor_params[ckp_path]["slope"]
    intercept = cor_params[ckp_path]["intercept"]
    y_hat = cor_prediction_bias(y_hat, slope, intercept)
    return y_hat


if __name__ == "__main__":
    main()
