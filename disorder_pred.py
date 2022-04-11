import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import logging

logger = logging.getLogger(__name__)

import functools
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from lib import models
from lib import datasets
from lib import testing


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
        for batch in tqdm(data_loader):
            y_hat = (
                model(batch["x"].to(device_loc), batch["male"].to(device_loc))
                .cpu()
                .squeeze(dim=1)
            )
            y_hats.append(y_hat)
            ys.append(batch["y"].squeeze(dim=1))
            image_names.append(batch["image_name"])
            males.append(batch["male"])
    ys = torch.cat(ys).numpy() * sd + mean
    y_hats = torch.cat(y_hats).numpy() * sd + mean
    image_names = np.array([name for batch in image_names for name in batch])
    males = np.array([male.item() for batch in males for male in batch])
    return image_names, males, ys, y_hats


def extract_bias_correction(name):
    pattern = re.compile(r"INFO	testing\.py	Linear bias prediction:.*")
    path = f"output/{name}/version_0.log"
    with open(path, "r") as f:
        matches = pattern.findall(f.read())
    line = matches[0].split("\t")
    slope = float(line[3].split(": ")[-1])
    intercept = float(line[4].split(": ")[-1])
    return slope, intercept


def get_best_model_path(name):
    pattern = re.compile(
        r"INFO\ttesting.py\tLoad model from /ceph01/projects/bone2gene/bone_age/output/.*"
    )
    path = f"output/{name}/version_0.log"
    with open(path, "r") as f:
        matches = pattern.findall(f.read())
        file = matches[0].split()[-1]
    return file.replace("/ceph01/projects/", "/home/rassman/")


def main():
    mean, sd = 127.31657409667969, 41.17934799194336
    dataset = datasets.DisorderDataset(
        annotation_path="../data_management/annotation_noKagg.csv",
        data_augmentation=None,
        img_dir="../data/annotated/",
        mask_dir=[
            "../data/masks/tensormask",
            # "../data/masks/unet",
        ],
        norm_method="zscore",
        cache=False,
        crop_to_mask=True,
    )
    data_loader = DataLoader(dataset, num_workers=8, batch_size=32, drop_last=False,)
    l = [
        "masked_effnet_fancy_aug",
        "masked_effnet_super_shallow_fancy_aug",
        # "masked_effnet_highRes_fancy_aug",
        # "masked_effnet-b4_shallow_pretr_fancy_aug",
        "masked_incept_batchsize_128_fancy_aug",
    ]

    dfs = []
    for name in tqdm(l):
        ckp_path = get_best_model_path(name)
        if "incept" in name:
            model = models.InceptionDbam.load_from_checkpoint(ckp_path)
        else:
            model = models.EfficientDbam.load_from_checkpoint(ckp_path)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        # model.fc2.register_forward_hook(get_activation("fc2"))
        x = torch.randn(1, 25)

        names, _, Y, Y_hat = predict_from_loader(model, data_loader, sd=sd, mean=mean)
        names = np.array([n.split("/")[-1] for n in names])
        Y_hat_star = testing.cor_prediction_bias(Y_hat, *extract_bias_correction(name))

        dfs.append(pd.DataFrame({"names": names, "Y": Y, name: Y_hat_star,}))
    df = functools.reduce(
        lambda left, right: pd.merge(left, right, on=["names", "Y"]), dfs
    )
    df["Y_hat"] = np.mean(df.iloc[:, 2:], axis=1)
    os.makedirs("output/bone_age_prediction2")
    df.to_csv("output/bone_age_prediction2/prediction_masked.csv")


if __name__ == "__main__":
    main()
