"""
module for custom functions used for evaluation and validation of trained models
"""
import logging, logging.config

logger = logging.getLogger(__name__)

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy import stats
import torch
import pandas as pd
import functools
from torch.utils.data import DataLoader

from lib import datasets, models, constants


def add_eval_args(parent_parser):
    parser = parent_parser.add_argument_group("Evaluation")
    parser.add_argument("--test_tta_rot", type=bool, default=True)
    parser.add_argument("--test_tta_flip", type=bool, default=True)
    parser.add_argument("--train_tta_rot", type=bool, default=False)
    parser.add_argument("--train_tta_flip", type=bool, default=False)
    parser.add_argument("--regress", type=bool, default=True)
    return parent_parser


def evaluate_bone_age_model(ckp_path, args, output_dir) -> dict:

    logger.info("====== Testing model =====")

    tta_rotations_test = [-10, -5, 0, 5, 10] if args.test_tta_rot else [0]
    tta_rotations_train = [-10, -5, 0, 5, 10] if args.train_tta_rot else [0]
    logger.info("Starting inference")
    dfs = predict_from_checkpoint(
        ckp_path=ckp_path,
        args=args,
        tta_rotations_test=tta_rotations_test,
        tta_flip_test=args.test_tta_flip,
        tta_rotations_train=tta_rotations_train,
        tta_flip_train=args.train_tta_flip,
    )
    results = {name: df for name, df in zip(["train", "validation", "test"], dfs)}

    def mad(df, yhat_key="y_hat"):
        return la.norm(df["y"] - df[yhat_key], 1) / len(df)

    for name, df in results.items():
        logger.info(
            f"{name} mad (without TTA and regression): {mad(df, 'y_hat-rot=0-no_flip')}"
        )
    log_dict = {
        "hp/validation_mad": mad(results["validation"], "y_hat-rot=0-no_flip"),
        "hp/validation_mad_reg": -1,
        "hp/validation_mad_reg_tta": -1,
        "hp/test_mad": mad(results["test"], "y_hat-rot=0-no_flip"),
        "hp/test_mad_reg": -1,
        "hp/test_mad_reg_tta": -1,
    }
    logger.info("===== regress correct on raw images =====")
    if args.regress:
        slope, intercept, _, _, _ = calc_prediction_bias(
            results["train"]["y"], results["train"]["y_hat-rot=0-no_flip"]
        )
        for name, df in results.items():
            df["y_hat_reg"] = cor_prediction_bias(
                df["y_hat-rot=0-no_flip"], slope, intercept
            )
            logger.info(f"{name} mad (after regression): {mad(df, 'y_hat_reg')}")
        for name in ["validation", "test"]:
            log_dict[f"hp/{name}_mad_reg"] = mad(results[name], "y_hat_reg")

    if args.regress and (args.test_tta_rot or args.test_tta_flip):
        slope, intercept, _, _, _ = calc_prediction_bias(
            results["train"]["y"], results["train"]["y_hat"]
        )
        for name, df in results.items():
            df["y_hat_reg_tta"] = cor_prediction_bias(df["y_hat"], slope, intercept)
            logger.info(
                f"{name} mad (after regression and TTA): {mad(df, 'y_hat_reg_tta')}"
            )
        for name in ["validation", "test"]:
            log_dict[f"hp/{name}_mad_reg_tta"] = mad(results[name], "y_hat_reg")

    if output_dir:
        output_dir = os.path.join(output_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"saving outputs to {output_dir}")
        for name, df in results.items():
            df.to_csv(os.path.join(output_dir, f"{name}_pred.csv"))
            model_name = os.path.basename(os.path.dirname(output_dir))
            os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
            save_correlation_plot(
                df,
                mad,
                title=f"Performance of {model_name} on {name} set",
                save_path=os.path.join(output_dir, "plots", f"{name}.png"),
            )
    return log_dict


def predict_from_checkpoint(
    ckp_path,
    args,
    tta_rotations_test=[-10, -5, 0, 5, 10],
    tta_flip_test=True,
    tta_rotations_train=[0],
    tta_flip_train=False,
) -> [pd.DataFrame]:
    model = models.get_model_class(args).load_from_checkpoint(ckp_path)
    mean, sd = model.y_mean, model.y_sd

    data_dir = args.data_dir if args.data_dir else model.data_dir
    if not data_dir:
        data_dir = constants.path_to_rsna_dir

    train_df = predict_bone_age(
        model,
        args,
        tta_rotations_train,
        tta_flip_train,
        os.path.join(data_dir, "annotation_bone_age_training_data_set.csv"),
        os.path.join(data_dir, "bone_age_training_data_set"),
        args.mask_dirs,
        mean,
        sd,
    )
    val_df = predict_bone_age(
        model,
        args,
        tta_rotations_test,
        tta_flip_test,
        os.path.join(data_dir, "annotation_bone_age_validation_data_set.csv"),
        os.path.join(data_dir, "bone_age_validation_data_set"),
        args.mask_dirs,
        mean,
        sd,
    )
    test_df = predict_bone_age(
        model,
        args,
        tta_rotations_test,
        tta_flip_test,
        os.path.join(data_dir, "annotation_bone_age_test_data_set.csv"),
        os.path.join(data_dir, "bone_age_test_data_set"),
        args.mask_dirs,
        mean,
        sd,
    )

    def vote(df):
        df["y_hat"] = df[
            df.columns[df.columns.to_series().str.contains("y_hat")]
        ].apply(np.mean, axis=1)
        return df

    train_df, val_df, test_df = vote(train_df), vote(val_df), vote(test_df)
    return train_df, val_df, test_df


def predict_bone_age(
    model,
    args,
    rotations,
    flip_img,
    anno_csv,
    img_dir,
    mask_dir,
    mean,
    sd,
    crop_to_mask=True,
) -> pd.DataFrame:
    l = []
    columns = ["filename", "male", "y"]
    flips = ["no_flip", "flip"] if flip_img else ["no_flip"]

    def make_df(pred, yhat_name):
        colnames = ["filename", "male", "y", yhat_name]
        d = {name: arr for name, arr in zip(colnames, pred)}
        return pd.DataFrame(d)

    for rot_angle in rotations:
        for flip in flips:
            dataset = datasets.RsnaBoneAgeKaggle(
                annotation_path=anno_csv,
                img_dir=img_dir,
                mask_dir=mask_dir,
                data_augmentation=datasets.RsnaBoneAgeDataModule.get_inference_augmentation(
                    args.input_width, args.input_height, rot_angle, (flip == "flip")
                ),
                bone_age_normalization=(mean, sd),
                epoch_size=None,
                crop_to_mask=crop_to_mask and mask_dir,
            )
            pred = predict_from_loader(
                model,
                DataLoader(
                    dataset,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    drop_last=False,
                ),
                mean=mean,
                sd=sd,
            )
            l.append(make_df([*pred], f"y_hat-rot={rot_angle}-{flip}"))
    return functools.reduce(lambda left, right: pd.merge(left, right, on=columns), l)


def predict_from_loader(model, data_loader, sd=1, mean=0, on_cpu=False):
    # TODO add hook and also save activations after ConvNet
    device_loc = "cpu"
    if not on_cpu:
        model.cuda()
        device_loc = "cuda"
    model.eval()
    y_hats = []
    ys = []
    image_names = []
    males = []
    with torch.set_grad_enabled(False):
        for batch in data_loader:
            y_hats.append(
                model(batch["x"].to(device_loc), batch["male"].to(device_loc))
                .cpu()
                .squeeze()
            )
            ys.append(batch["y"])
            image_names.append(batch["image_name"])
            males.append(batch["male"])
    ys = torch.cat(ys).squeeze().numpy() * sd + mean
    y_hats = torch.cat(y_hats).numpy() * sd + mean
    image_names = np.array([name for batch in image_names for name in batch])
    males = np.array([male for batch in males for male in batch])
    return image_names, males, ys, y_hats


def save_correlation_plot(
    df,
    error_func=None,
    y_col="y",
    yhat_col="y_hat",
    title="add Title",
    error_pos_x=0,
    error_pos_y=220,
    y_label="ground truth bone age (months)",
    yhat_label="predicted bone age (months)",
    error_name="error",
    save_path=None,
):
    """
    Quick overview over model performance.

    Renders a scatter plot of y (real value) against yhat (predicted value) and prints the value of the defined error function.

    :param df: DataFrame containing y and yhat in specified columns
    :param y_col: name of y column
    :param yhat_col: name of y_hat column
    :param error_func: Error function
    :param title: title of the plot
    :param error_pos_x: x position where the performance metrics is placed
    :param error_pos_y: y position where the performance metrics is placed
    :param y_label: name for y (real value) as axis label
    :param yhat_label: name for yhat (predicted value) as axis label
    :param error_name: Name of the error function in the plot
    :param save_path: path to save the model
    """
    error = error_func(df, yhat_key="y_hat")

    plt.figure()
    plt.scatter(df[y_col], df[yhat_col], alpha=0.3)
    plt.xlabel(y_label)
    plt.ylabel(yhat_label)
    plt.suptitle(title)
    plt.plot(
        df[y_col],
        df[y_col],
        "r-",
    )
    plt.text(error_pos_x, error_pos_y, f"{error_name}={error:.2f}")
    if save_path:
        plt.savefig(save_path)


def calc_prediction_bias(y, yhat, verbose=True):
    """calculates the predicted signed error (yhat - y) for each prediction and performs a linear regression"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(yhat, yhat - y)
    if verbose:
        logger.info(
            f"Linear bias prediction:\nslope: {slope:.4f}\nintercept: {intercept:.4f}\nr = {r_value:.4f}\np-value = {p_value:.1E}\nstd error = {std_err:.1E}"
        )
    return slope, intercept, r_value, p_value, std_err


def cor_prediction_bias(yhat, slope, intercept):
    """corrects model predictions (yhat) for linear bias (defined by slope and intercept)"""
    return yhat - (yhat * slope + intercept)
