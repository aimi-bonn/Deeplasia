"""
Module to manage datasets and loaders
"""
import os
import re

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import yaml

from typing import Union, List

import logging

logger = logging.getLogger(__name__)

# torch.multiprocessing.set_sharing_strategy("file_system")


class ImageCache:
    def __init__(self):
        """
        class to cache images to avoid hard drive accesses to the cost of high RAM usage
        """
        self.cache = {}

    def open_image(self, path: str, mode=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
        """
        Cache image if not available else return cached image (without hard drive access)

        :param path: path to image
        :param mode: cv2 imread mode
        :return: image
        """
        if path in self.cache.keys():
            return self.cache[path]
        else:
            img = cv2.imread(path, mode)
            self.cache[path] = img
            return img


class HandDataset(Dataset):

    CACHE = ImageCache()

    RELEVANT_ENTRIES = [
        "image_ID",
        "dir",
        "chronological_age",
        "sex",
        "bone_age",
    ]

    def __init__(
        self,
        annotation_df: Union[str, pd.DataFrame] = "data-management/annotation.csv",
        img_dir="../data/annotated/",
        mask_dirs=["../data/masks/tensormask"],
        age_norm=(0, 1),
        data_augmentation=None,
        norm_method="zscore",
        mask_crop_size=-1,
        use_cache=False,
        y_col="disorder",
        fourier=False,
    ):
        self.fourier = fourier
        self.mask_dir = mask_dirs
        self.img_dir = img_dir
        self.norm_method = norm_method
        self.mask_crop_size = mask_crop_size
        self.y_col = y_col
        self.anno_df = (
            (
                pd.read_csv(annotation_df)
                if isinstance(annotation_df, str)
                else annotation_df
            )
            .copy()[self.RELEVANT_ENTRIES]
            .reset_index()
        )
        self.data_augmentation = (
            data_augmentation
            if data_augmentation
            else HandDatamodule.get_inference_augmentation()
        )
        self.use_cache = use_cache

        self.mean_age, self.std_age = age_norm
        self._remove_if_mask_missing()

        self.anno_df["male"] = np.where(self.anno_df["sex"] == "M", 1, 0).astype(float)
        self.anno_df["bone_age"] = (
            self.anno_df["bone_age"] - self.mean_age
        ) / self.std_age

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, str):
            index = self.anno_df.index[self.anno_df.image_ID == index].item()
        image, image_path = self._preprocess_image(index)

        male = torch.Tensor([self.anno_df.male.iloc[index]])
        bone_age = torch.Tensor([self.anno_df.bone_age.iloc[index]])

        sample = {
            "x": image,
            "male": male,
            "bone_age": bone_age,
            "image_name": image_path,
        }
        return sample

    def _preprocess_image(self, index):
        image, image_path = self._open_image(index)
        if self.mask_dir:
            mask = self._open_mask(index)
            image = self._apply_mask(image, mask)
            image = cv2.bitwise_and(image, image, mask=mask)  # mask the hand
            image = self._crop_to_mask(image, mask)
        image = self.data_augmentation(image=image)["image"]
        image = self._normalize_image(image)

        if torch.isnan(image).any():
            logger.warning(f"Created nan: {image_path}")
        return image, image_path

    def _open_image(self, index: int, method=cv2.IMREAD_GRAYSCALE) -> (np.ndarray, str):
        img_path = os.path.join(
            self.img_dir,
            self.anno_df["dir"].iloc[index],
            self.anno_df["image_ID"].iloc[index],
        )
        image = self._cached_open_image(img_path, method)
        assert (
            np.sum(image) != 0
        ), f"image with index {index} is all black (sum is {np.sum(image)})"
        assert (
            np.std(image) > 1e-5
        ), f"std of image with index {index} is close to zero ({np.std(image)})"
        return image, img_path

    def _open_mask(self, index: int) -> np.ndarray:
        """
        search for a corresponding mask
        """
        for d in np.random.permutation(self.mask_dir):
            img_path = os.path.join(
                d, self.anno_df["dir"].iloc[index], self.anno_df["image_ID"].iloc[index]
            )
            if os.path.exists(img_path):
                break
        mask = self._cached_open_image(img_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > mask.max() // 2).astype(np.uint8)
        return mask

    def _apply_mask(self, image, mask) -> np.ndarray:
        """
        apply image and subtract min intensity (1st percentile) from the masked area
        """
        if self.norm_method != "interval":
            image = cv2.bitwise_and(image, image, mask=mask)  # mask the hand
            m = np.percentile(cv2.bitwise_and(mask, image), 1)
            image = cv2.subtract(image, m)  # no underflow
        return image

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(torch.float32)
        if self.norm_method == "zscore":
            m = img.mean()
            sd = img.std()
            img = (img - m) / sd
        elif self.norm_method == "interval":
            img = img - img.min()
            img = img / img.max()
        return img

    def _crop_to_mask(self, image, mask):
        """
        rotate and flip image, and crop to mask if specified
        """
        if self.mask_crop_size <= 0:
            return image

        x = np.nonzero(np.max(mask, axis=0))
        xmin, xmax = (np.min(x), np.max(x) + 1)
        y = np.nonzero(np.max(mask, axis=1))
        ymin, ymax = (np.min(y), np.max(y) + 1)
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width // 2
        y_center = ymin + height // 2

        size = max(height, width)
        size = round(size * self.mask_crop_size)

        xmin_new = x_center - size // 2
        xmax_new = x_center + size // 2
        ymin_new = y_center - size // 2
        ymax_new = y_center + size // 2

        top = abs(min(0, ymin_new))
        bottom = max(0, ymax_new - mask.shape[0])
        left = abs(min(0, xmin_new))
        right = max(0, xmax_new - mask.shape[1])

        out = cv2.copyMakeBorder(
            image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT
        )
        ymax_new += top
        ymin_new += top
        xmax_new += left
        xmin_new += left

        return out[ymin_new:ymax_new, xmin_new:xmax_new]

    def _remove_if_mask_missing(self):
        if not self.mask_dir:
            logger.info("No masking - raw images used")
            logger.info(f"Number of images : {len(self.anno_df)}")
            return
        if not type(self.mask_dir) == list:
            self.mask_dir = [self.mask_dir]
        logger.info(f"Used masking dirs : {self.mask_dir}")
        avail_masks = []
        for d in self.mask_dir:
            masks_in_dir = np.array(
                [
                    os.path.exists(os.path.join(d, row["dir"], row["image_ID"]))
                    for _, row in self.anno_df.iterrows()
                ]
            )
            logger.info(
                f"Number of masks available at {d} : {sum(masks_in_dir)} / {len(self.anno_df)}"
            )
            avail_masks.append(masks_in_dir)
        avail_masks = np.array(avail_masks).max(axis=0)
        logger.info(
            f"Number of masks available from all sources combined : {sum(avail_masks)} / {len(self.anno_df)}"
        )
        self.anno_df = self.anno_df.iloc[avail_masks]

    def _cached_open_image(self, path: str, mode=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
        if not self.use_cache:
            return cv2.imread(os.path.abspath(path), mode)
        else:
            return self.CACHE.open_image(os.path.abspath(path), mode)

    def get_norm(self):
        return self.mean_age, self.sd_age

    def __len__(self):
        return len(self.anno_df)


class HandDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        annotation_path: str = "data-management/annotation.csv",
        split_path: str = "data-management/splits/kaggle_only.csv",
        split_column: str = "org_split",
        img_dir: str = "../data/annotated/",
        mask_dirs: List[str] = ["../data/masks/tensormask", "../data/masks/unet"],
        train_batch_size: int = 24,
        test_batch_size: int = 32,
        num_workers: int = 8,
        input_size: List[int] = [1, 512, 512],
        norm_method: str = "zscore",
        mask_crop_size: float = -1,
        use_cache: bool = False,
        rotation_range: int = 30,
        translate_perc: float = 0.2,
        flip_p: float = 0.5,
        zoom_in: float = 0.2,
        zoom_out: float = 0.2,
        shear_angle: float = 10,
        contrast_gamma: int = 30,
        sharpen_p: float = 0.2,
        clae_p: float = 1,
    ):
        """
        Datamodule representing train, val, and test set of the bone disorder data

        :param annotation_path: path to annotation csv file
        :param split_path: path specifying used split
        :param split_column: the column of the split csv to use
        :param img_dir: path to dir containing the raw images (as .png or similar, no direct DICOM support!)
        :param mask_dirs: list of paths to directories containing masks (if None no masking is conducted)
        :param train_batch_size: batch size during training
        :param test_batch_size: batch size during validation and testing (can usually be set higher than train batch size)
        :param num_workers: number of workers for the DataLoaders
        :param input_size: resolution of the input image as [C, W, H] (only grayscale, ie C == 1 supported)
        :param norm_method: normalization method for the images to use, one of 'zscore' or 'interval'
        :param use_cache: use image RAM cache
        :param rotation_range: range of rotation during training
        :param translate_perc: translation of images during training
        :param flip_p: probability of random horizontal flip during training
        :param zoom_in: maximum factor to zoom in during training
        :param zoom_out: maximum factor to zoom out during training
        :param shear_angle: maximum angle for shearing during training
        :param contrast_gamma: maximum percentage of gamma histogram transformation during training
        """
        self.save_hyperparameters(logger=False)

        # For linking with model
        self.norm_method = norm_method
        self.train_batch_size = train_batch_size
        self.masked_input = bool(mask_dirs)
        self.input_size = input_size

        self.num_workers = num_workers
        self.test_batch_size = test_batch_size

        train, val, test = self._handle_data_splits(
            annotation_path, split_path, split_column,
        )
        self.img_dir = img_dir
        self.mask_dirs = mask_dirs

        common_kwargs = {
            "img_dir": img_dir,
            "mask_dirs": mask_dirs,
            "age_norm": (self.age_mean, self.age_std),
            "norm_method": self.norm_method,
            "use_cache": use_cache,
            "mask_crop_size": mask_crop_size,
        }
        logger.info(f"{'='*10} Setting up train data {'='*10}")

        self.train = HandDataset(
            annotation_df=train,
            data_augmentation=self.get_train_augmentation(
                input_size=input_size,
                rotation_range=rotation_range,
                translate_perc=translate_perc,
                flip_p=flip_p,
                zoom_in=zoom_in,
                zoom_out=zoom_out,
                shear_percent=shear_angle,
                contrast_gamma=contrast_gamma,
                sharpen_p=sharpen_p,
                clae_p=clae_p,
            ),
            **common_kwargs,
        )
        logger.info(f"{'='*10} Setting up validation data {'='*10}")
        self.validation = HandDataset(
            annotation_df=val,
            data_augmentation=self.get_inference_augmentation(
                input_size[1], input_size[2]
            ),
            **common_kwargs,
        )
        logger.info(f"{'='*10} Setting up test data {'='*10}")
        self.test = HandDataset(
            annotation_df=test,
            data_augmentation=self.get_inference_augmentation(
                input_size[1], input_size[2]
            ),
            **common_kwargs,
        )

    @staticmethod
    def from_config(config, data_dir="../data/", **kwargs):
        if isinstance(config, str):
            if config.split(".")[-1] == "ckpt":
                config = re.sub(r"ckp/.*\.ckpt", "config.yaml", config)
            logger.info(f"restoring dataset from {config}")
            with open(config) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        if "data" in config.keys():
            config = config["data"]
        if data_dir:
            config["img_dir"] = re.sub(".*/data/", data_dir, config["img_dir"])
            config["mask_dirs"] = [
                re.sub(".*/data/", data_dir, x) for x in config["mask_dirs"]
            ]
        for k, v in kwargs.items():
            if k in config.keys():
                config[k] = v
        return HandDatamodule(**config)

    def prepare_data_per_node(self):
        """useless, but required for the CLI"""
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.test_batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return {
            "test": self.test_dataloader(),
            "val": self.val_dataloader(),
            "train": self.train_dataloader(),
        }

    def _handle_data_splits(
        self, annotation_path, split_path, split_column,
    ):
        df = pd.read_csv(annotation_path)
        split = pd.read_csv(split_path)

        df = df.merge(
            split[["patient_ID", "dir", split_column]],
            on=["patient_ID", "dir"],
            how="inner",
        )

        train = df.loc[df[split_column] == "train"]
        self.age_mean = train["bone_age"].mean()
        self.age_std = train["bone_age"].std()
        val = df.loc[df[split_column] == "val"]
        test = df.loc[df[split_column] == "test"]
        # TODO validate that all images are present
        logger.info(f"found {len(train)} train images")
        logger.info(f"found {len(val)} val images")
        logger.info(f"found {len(test)} test images")

        return train, val, test

    @staticmethod
    def get_train_augmentation(
        input_size=(1, 512, 512),
        rotation_range=30.0,
        translate_perc=0.2,
        flip_p=0.5,
        zoom_in=0.2,
        zoom_out=0.2,
        shear_percent=10.0,
        contrast_gamma=30.0,
        sharpen_p=0.2,
        clae_p=1,
    ):
        return A.Compose(
            [
                A.HorizontalFlip(p=flip_p),
                A.Affine(
                    scale=(1 - zoom_in, 1 / (1 - zoom_out)),
                    translate_percent={
                        # allow for independent selection in each direction
                        "x": (-translate_perc, translate_perc),
                        "y": (-translate_perc, translate_perc),
                    },
                    rotate=(-rotation_range, rotation_range),
                    shear=(-shear_percent, shear_percent),
                    p=1.0,
                ),
                A.Sharpen(alpha=(0.5, 0.75), lightness=(0.5, 1.0), p=sharpen_p),
                A.RandomResizedCrop(
                    input_size[1], input_size[2], scale=(1.0, 1.0), ratio=(1.0, 1.0),
                ),
                A.OneOf(
                    [
                        A.augmentations.transforms.CLAHE(p=clae_p, clip_limit=3),
                        A.augmentations.transforms.RandomGamma(
                            (100 - contrast_gamma, 100 + contrast_gamma), p=0.5,
                        ),
                    ],
                    p=1,
                ),
                ToTensorV2(),
            ],
            p=1,
        )

    @staticmethod
    def get_inference_augmentation(width=512, height=512, rotation_angle=0, flip=False):
        return A.Compose(
            [
                A.transforms.HorizontalFlip(p=flip),
                A.augmentations.geometric.transforms.Affine(
                    rotate=(rotation_angle, rotation_angle), p=1.0,
                ),
                A.augmentations.crops.transforms.RandomResizedCrop(
                    width, height, scale=(1.0, 1.0), ratio=(1.0, 1.0)
                ),
                ToTensorV2(),
            ],
            p=1,
        )


class InferenceDataset(HandDataset):

    RELEVANT_ENTRIES = [
        "image_ID",
        "dir",
        "sex",
    ]

    def __init__(
        self,
        annotation_df: str = "data-management/annotation.csv",
        split_path: str = "",
        split_column: str = None,
        split_name: str = "test",
        img_dir: str = "../data/annotated/",
        mask_dirs: str = ["../data/masks/tensormask"],
        data_augmentation: object = None,
        norm_method: str = "zscore",
        mask_crop_size: float = -1,
        input_size: List[int] = [1, 512, 512],
        y_col: str = "disorder",
        fourier: str = "",
        **kwargs,
    ):
        """
        simple loader for inference tasks (disorder or bone age prediction)

        :param annotation_df: path to csv file containing the image paths and the sex
        :param split_df: split containing the subset used for inference
        :param split_column: column containing the split. If None or False, all images are used
        :param split_name: name of the split. If split_column is specified only entries that are equal to this column are used.
        :param img_dir: path to root dir containing all images
        :param mask_dirs: path to root dirs containing the masks
        :param data_augmentation: data_augmentation to apply
        :param norm_method: method to normalize the image
        :param y_col: column containing gt to predict
        """
        self.mask_dir = mask_dirs
        self.img_dir = img_dir
        self.norm_method = norm_method
        self.mask_crop_size = mask_crop_size
        self.y_col = y_col
        self.fourier = fourier
        self.anno_df = (
            pd.read_csv(annotation_df)
            if isinstance(annotation_df, str)
            else annotation_df
        ).copy()
        if split_path and split_column:
            split_df = (
                pd.read_csv(split_path) if isinstance(split_path, str) else split_path
            )
            assert (
                split_column in split_df.columns
            ), f"defined split columns ({split_column}) not found in the specified csv file"
            self.anno_df = self.anno_df.merge(
                split_df[["patient_ID", "dir", split_column]],
                on=["patient_ID", "dir"],
                how="inner",
            )
            self.anno_df = self.anno_df.loc[self.anno_df[split_column] == split_name]

        self.data_augmentation = (
            data_augmentation
            if data_augmentation
            else HandDatamodule.get_inference_augmentation(input_size[1], input_size[2])
        )
        self.use_cache = False
        self._remove_if_mask_missing()

        self.anno_df["male"] = np.where(self.anno_df["sex"] == "M", 1, 0).astype(float)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image, image_path = self._preprocess_image(index)
        male = torch.Tensor([self.anno_df.male.iloc[index]])

        sample = {
            "x": image,
            "male": male,
            "image_name": image_path,
        }
        if self.y_col:
            sample = sample | {
                "y": torch.Tensor([self.anno_df[self.y_col].iloc[index]])
            }
        return sample
