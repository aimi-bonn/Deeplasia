"""
Module to manage datasets and loaders
"""
import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lib import constants
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib import preprocessing

logger = logging.getLogger(__name__)

class RsnaBoneAgeKaggle(Dataset):

    DEFAUL_IMAGE_RESOLUTION = (512, 512)

    def __init__(
        self,
        annotation_path,
        img_dir,
        mask_dir=None,
        data_augmentation=None,
        bone_age_normalization=None,
        epoch_size=None,
    ):
        """
        Create a RNSA Bone Age (kaggle competition) Dataset

        :param data_augmentation: defines conducted data augmentation
            (instance of 'preprocessing.BoneAgeDataAugmentation')
        :param bone_age_normalization: Tuple containing mean and sd for normalizing Y
            (if None calculated from the own dataset, can be retrieved by 'get_norm()' method)
        :param annotation_path: path to annotation csv file
        :param img_dir: base dir where images are located
        :param epoch_size: artificial size of an epoch (if None or 0 native size original size)
        """
        anno_df = pd.read_csv(annotation_path)
        logger.info(f"Loading annotation data from {annotation_path}")
        logger.info(f"Loading image data from {img_dir}")
        self.ids, self.male, self.Y = self._load_bone_age_anno(anno_df)
        self.img_dir = img_dir
        assert np.all(
            np.vectorize(lambda i: os.path.exists(os.path.join(img_dir, f"{i}.png")))(
                self.ids
            )
        )  # check if all annotated images are really in the dir

        self.mask_dir = mask_dir
        self._remove_if_mask_missing()

        self.n_samples = (
            len(self.ids)
            if not epoch_size or epoch_size > len(self.ids)
            else epoch_size
        )
        logger.info(f"(Virtual) Epoch size : {self.n_samples}")

        self.data_augmentation = (
            data_augmentation
            if data_augmentation
            else preprocessing.BoneAgeDataAugmentation(
                augment=False, output_tensor_size=self.DEFAUL_IMAGE_RESOLUTION
            )
        )
        logger.info(f"Augmentations used : {self.data_augmentation}")

        if not bone_age_normalization:
            self.mean_Y = np.mean(self.Y)
            self.sd_Y = np.std(self.Y)
        else:
            self.mean_Y = bone_age_normalization[0]
            self.sd_Y = bone_age_normalization[1]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image, image_path = self._open_image(index)
        if self.mask_dir:
            mask = self._open_mask(index)
            image = cv2.bitwise_and(image, image, mask)  # mask the hand
        image = self.data_augmentation(image=image)["image"]
        image = self._normalize_image(image)
        male = torch.Tensor([self.male[index]])

        y = self.Y[index]
        y = torch.Tensor([(y - self.mean_Y) / self.sd_Y])
        sample = {"x": image, "male": male, "y": y, "image_name": image_path}

        return sample

    def __len__(self):
        return self.n_samples

    def _open_image(self, index: int) -> (np.ndarray, str):
        img_path = os.path.join(self.img_dir, f"{self.ids[index]}.png")
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert (
            np.sum(image) != 0
        ), f"image with index {index} is all black (sum is {np.sum(image)})"
        assert (
            np.std(image) > 1e-5
        ), f"std of image with index {index} is close to zero ({np.std(image)})"
        return image, img_path

    def _remove_if_mask_missing(self) -> None:
        if not self.mask_dir:
            logger.info("No masking - raw images used")
            logger.info(f"Number of images : {len(self.ids)}")
            return
        if not type(self.mask_dir) == list:
            self.mask_dir = [self.mask_dir]
        logger.info(f"Used masking dirs : {self.mask_dir}")
        avail_masks = []
        for d in self.mask_dir:
            avail_masks.append(
                np.vectorize(lambda i: os.path.exists(os.path.join(d, f"{i}.png")))(
                    self.ids
                )
            )
            logger.info(f"Number of masks available at {d} : {sum(avail_masks)} / {len(self.ids)}")
        avail_masks = np.array(avail_masks).max(axis=0)
        logger.info(f"Number of masks available from all sources combined : {sum(avail_masks)} / {len(self.ids)}")
        self.ids = self.ids[np.where(avail_masks)]
        self.male = self.male[np.where(avail_masks)]
        self.Y = self.Y[np.where(avail_masks)]

    def _open_mask(self, index) -> np.ndarray:
        """
        search for a corresponding mask
        """
        for d in np.random.permutation(self.mask_dir):
            img_path = os.path.join(d, f"{self.ids[index]}.png")
            if os.path.exists(img_path):
                break
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return mask

    def get_norm(self):
        return self.mean_Y, self.sd_Y

    def renorm(self, y):
        """Renormalize y to original value prior to zscore normalization"""
        return y * self.sd + self.mean

    @staticmethod
    def _normalize_image(img: torch.Tensor) -> torch.Tensor:
        img = img.to(torch.float32)
        m = img.mean()
        sd = img.std()
        img = (img - m) / sd
        return img

    @staticmethod
    def verify_dataset(anno_path, dataset_path):
        """checks all images in the dataset for validity. Add more funcs if needed."""
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        from torchvision.transforms import transforms

        dataset = RsnaBoneAgeKaggle(
            anno_path,
            dataset_path,
        )

        for index in tqdm(dataset.ids):
            img_path = os.path.join(dataset.img_dir, f"{index}.png")
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = Image.open(img_path)
            if np.sum(image) == 0:
                print(f"image with index {index} is all black (sum is {np.sum(image)})")
            if np.std(image) < 1e-5:
                print(
                    f"std of image with index {index} is close to zero ({np.std(image)})"
                )
            if np.sum(image) == 0 or np.std(image) < 1e-5:
                plt.figure()
                plt.imshow(image, cmap="gray")
                plt.show()
            tensor = transforms.ToTensor()(image)
            if torch.sum(tensor) == 0:
                print(
                    f"image with index {index} is all black (sum is {torch.sum(tensor)})"
                )
            if torch.std(tensor) < 1e-5:
                print(
                    f"std of image with index {index} is close to zero ({torch.std(tensor)})"
                )
            if torch.sum(tensor) == 0 or torch.std(tensor) < 1e-5:
                plt.figure()
                plt.imshow(image.permute(1, 2, 0)[:, :, 0], cmap="gray")
                plt.show()
            # tensor = preprocessing.normalize_image("zscore", tensor)

    @staticmethod
    def _load_bone_age_anno(anno_df):
        """Assumes that the cols contain ids, gender, and ground truth bone age, respectively"""
        ids = anno_df.iloc[:, 0].to_numpy(dtype=np.int)
        male = anno_df.iloc[:, 1].to_numpy()
        if male[0] in ["M", "F"]:
            male = np.where(male == "M", 1, 0)
        male = male.astype(np.bool)
        y = anno_df.iloc[:, 2].to_numpy(dtype=np.float32)
        assert len(ids) == len(male) == len(y)
        return ids, male, y


class RsnaBoneAgeDataModule(pl.LightningDataModule):
    NO_AUGMENT = A.Compose(
        [
            A.augmentations.crops.transforms.RandomResizedCrop(
                512, 512, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            ),
            A.pytorch.ToTensorV2(),
        ],
        p=1,
    )

    def __init__(
        self,
        train_augment,
        valid_augment=None,
        test_augment=None,
        batch_size=32,
        num_workers=4,
        data_dir=constants.path_to_rsna_dir,
        mask_dir=None,
        epoch_size=2048,
    ):
        """
        Dataset class for RSNA bone age data

        :param train_augment: augmentation used for training (if None only resizing)
        :param valid_augment: augmentation used for validation (if None only resizing)
        :param test_augment: augmentation used for testing (if None only resizing)
        :param batch_size: batch size
        :param num_workers: number of threads for data loading and preprocessing
        :param data_dir: parent dir containing the data (if None `../../data/annotated/rsna_bone_age/` is assumed) use a ramdisk or local storage for faster access speeds
        :poram mask_dir: dir or list of dirs containing masks for the presented images. If multiple masks are avaible for any given image, the mask is chosen randomly.
        :param epoch_size: (virtual) size of an epoch. If None the true size is used.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_augment = (
            train_augment if train_augment else RsnaBoneAgeDataModule.NO_AUGMENT
        )
        self.valid_augment = (
            valid_augment if valid_augment else RsnaBoneAgeDataModule.NO_AUGMENT
        )
        self.test_augment = (
            test_augment if test_augment else RsnaBoneAgeDataModule.NO_AUGMENT
        )

        if not data_dir:
            data_dir = constants.path_to_rsna_dir

        logger.info(f"====== Setting up training data ======")
        self.train = RsnaBoneAgeKaggle(
            os.path.join(data_dir, "annotation_bone_age_training_data_set.csv"),
            os.path.join(data_dir, "bone_age_training_data_set"),
            data_augmentation=self.train_augment,
            bone_age_normalization=None,  # calculate renorm based on training set
            epoch_size=epoch_size,
            mask_dir=mask_dir,
        )
        self.mean, self.sd = self.train.get_norm()
        logger.info(f"Parameters used for bone age normalization: mean = {self.mean} - sd = {self.sd}")
        logger.info(f"====== ====== ====== ====== ====== ======")
        logger.info(f"====== Setting up validation data ======")
        logger.info(f"Setting up validation data")
        self.validation = RsnaBoneAgeKaggle(
            os.path.join(
                data_dir,
                "annotation_bone_age_validation_data_set.csv",
            ),
            os.path.join(data_dir, "bone_age_validation_data_set"),
            mask_dir=mask_dir,
            data_augmentation=self.valid_augment,
            bone_age_normalization=(self.mean, self.sd),
        )
        logger.info(f"====== ====== ====== ====== ======")
        logger.info(f"====== Setting up test data ======")
        self.test = RsnaBoneAgeKaggle(
            os.path.join(data_dir, "annotation_bone_age_test_data_set.csv"),
            os.path.join(data_dir, "bone_age_test_data_set"),
            mask_dir=mask_dir,
            data_augmentation=self.test_augment,
            bone_age_normalization=(self.mean, self.sd),
        )
        # self._assert_data_integrity()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def _assert_data_integrity(self):
        """
        Check if all data is available at the configured source
        """
        assert len(self.train.ids) == 12610
        assert len(self.validation.ids) == 1425
        assert len(self.test.ids) == 200


def add_data_augm_args(parent_parser):
    parent_parser.add_argument("--input_width", type=int, default=512)
    parent_parser.add_argument("--input_height", type=int, default=512)
    parser = parent_parser.add_argument_group("Data_Augmentation")
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--rotation_range", type=int, default=20)
    parser.add_argument("--relative_scale", type=float, default=0.2)
    parser.add_argument("--shear_percent", type=int, default=1)
    parser.add_argument("--translate_percent", type=int, default=0.2)
    return parent_parser


def setup_augmentation(args):
    return A.Compose(
        [
            A.transforms.HorizontalFlip(p=args.flip_p),
            A.augmentations.geometric.transforms.Affine(
                scale=(1 - args.relative_scale, 1 + args.relative_scale),
                translate_percent=args.translate_percent,
                rotate=(-args.rotation_range, args.rotation_range),
                shear=args.shear_percent,
                p=1.0,
            ),
            A.augmentations.crops.transforms.RandomResizedCrop(
                args.input_width, args.input_height, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            ),
            ToTensorV2(),
        ],
        p=1,
    )


def main():
    from lib import constants

    print("Verifying all images in all RSNA bone age datasets")

    RsnaBoneAgeKaggle.verify_dataset(
        constants.path_to_bone_age_training_anno,
        constants.path_to_bone_age_training_dataset,
    )
    RsnaBoneAgeKaggle.verify_dataset(
        constants.path_to_bone_age_validation_anno,
        constants.path_to_bone_age_validation_dataset,
    )
    RsnaBoneAgeKaggle.verify_dataset(
        constants.path_to_bone_age_test_anno,
        constants.path_to_bone_age_test_dataset,
    )


if __name__ == "__main__":
    main()
