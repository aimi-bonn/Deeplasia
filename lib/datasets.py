"""
Module to manage datasets and loaders
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lib import constants
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

import logging

logger = logging.getLogger(__name__)


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


class RsnaBoneAgeKaggle(Dataset):

    DEFAULT_IMAGE_RESOLUTION = (512, 512)
    CACHE = ImageCache()

    def __init__(
        self,
        annotation_path,
        img_dir,
        mask_dir=None,
        data_augmentation=None,
        bone_age_normalization=None,
        epoch_size=None,
        crop_to_mask=False,
        norm_method="zscore",
        cache=False,
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
        :param crop_to_mask (bool): assert that the whole mask of the hand is within the cropped image
        :param norm_method (str): Normalization method of the image, one of 'zscore' or 'interval' (scale to [0, 1]
        :param cache (bool): Bool if images should be cached
        """
        anno_df = pd.read_csv(annotation_path)
        logger.info(f"Loading annotation data from {annotation_path}")
        logger.info(f"Loading image data from {img_dir}")
        self.ids, self.male, self.Y = self._load_bone_age_anno(anno_df)
        self.img_dir = img_dir
        self.ids, self.male, self.Y = self._load_bone_age_anno(anno_df)
        self.norm_method = norm_method
        assert np.all(
            np.vectorize(lambda i: os.path.exists(os.path.join(img_dir, f"{i}.png")))(
                self.ids
            )
        )  # check if all annotated images are really in the dir

        self.mask_dir = mask_dir
        self._remove_if_mask_missing()
        self.use_cache = cache

        self.n_samples = (
            len(self.ids)
            if not epoch_size or epoch_size > len(self.ids)
            else epoch_size
        )
        logger.info(f"(Virtual) Epoch size : {self.n_samples}")

        self.data_augmentation = (
            data_augmentation
            if data_augmentation
            else RsnaBoneAgeDataModule.get_inference_augmentation()
        )
        logger.info(f"Augmentations used : {self.data_augmentation}")

        self.crop_to_mask = crop_to_mask
        self.crop_size = 1.3

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
            image = cv2.bitwise_and(image, image, mask=mask)  # mask the hand
            image = self._crop_to_mask(image, mask)
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
        image = self._cached_open_image(img_path, cv2.IMREAD_UNCHANGED)
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
            img_path = os.path.join(d, f"{self.ids[index]}.png")
            if os.path.exists(img_path):
                break
        mask = self._cached_open_image(img_path, cv2.IMREAD_GRAYSCALE)
        return mask

    def _cached_open_image(self, path: str, mode=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
        if not self.use_cache:
            return cv2.imread(path, mode)
        else:
            return self.CACHE.open_image(path, mode)

    def _crop_to_mask(self, image, mask):
        """
        rotate and flip image, and crop to mask if specified
        """
        if not self.crop_to_mask:
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
        size = round(size * self.crop_size)

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
            masks_in_dir = np.vectorize(
                lambda i: os.path.exists(os.path.join(d, f"{i}.png"))
            )(self.ids)
            logger.info(
                f"Number of masks available at {d} : {sum(masks_in_dir)} / {len(self.ids)}"
            )
            avail_masks.append(masks_in_dir)
        avail_masks = np.array(avail_masks).max(axis=0)
        logger.info(
            f"Number of masks available from all sources combined : {sum(avail_masks)} / {len(self.ids)}"
        )
        self.ids = self.ids[np.where(avail_masks)]
        self.male = self.male[np.where(avail_masks)]
        self.Y = self.Y[np.where(avail_masks)]

    def get_norm(self):
        return self.mean_Y, self.sd_Y

    def renorm(self, y):
        """Renormalize y to original value prior to zscore normalization"""
        return y * self.sd + self.mean

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        if self.norm_method == "zscore":
            img = img.to(torch.float32)
            m = img.mean()
            sd = img.std()
            img = (img - m) / sd
        elif self.norm_method == "interval":
            img = img.to(torch.float32)
            img = img - img.min()
            img = img / img.max()
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
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
        ids = anno_df.iloc[:, 0].to_numpy(dtype=np.int32)
        male = anno_df.iloc[:, 1].to_numpy()
        if male[0] in ["M", "F"]:
            male = np.where(male == "M", 1, 0)
        y = anno_df.iloc[:, 2].to_numpy(dtype=np.float32)
        male = male.astype(np.float32)
        assert len(ids) == len(male) == len(y)
        return ids, male, y


class RsnaBoneAgeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_augment=None,
        valid_augment=None,
        test_augment=None,
        width=512,
        height=512,
        batch_size=32,
        num_workers=4,
        data_dir=constants.path_to_rsna_dir,
        mask_dir=None,
        epoch_size=2048,
        rotation_angle=0,
        flip=False,
        crop_to_mask=False,
        img_norm_method="zscore",
        cache=False,
    ):
        """
        Dataset class for RSNA bone age data

        :param train_augment: augmentation used for training (if None only resizing)
        :param valid_augment: augmentation used for validation (if None only resizing)
        :param test_augment: augmentation used for testing (if None only resizing)
        :param batch_size: batch size
        :param num_workers: number of threads for data loading and preprocessing
        :param data_dir: parent dir containing the data (if None `../../data/annotated/rsna_bone_age/` is assumed) use a ramdisk or local storage for faster access speeds
        :param mask_dir: dir or list of dirs containing masks for the presented images. If multiple masks are available for any given image, the mask is chosen randomly.
        :param epoch_size: (virtual) size of an epoch. If None the true size is used.
        :param rotation_angle (int): angle to rotate before data augmentation
        :param flip (bool): flip the image
        :param crop_to_mask (bool): assert that the whole mask of the hand is within the cropped image
        :param img_norm_method (str): Normalization method of the image, one of 'zscore' or 'interval' (scale to [0, 1]
        :param cache (bool): Bool if images should be cached (Note the RAM usage of >10GB)
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width = width
        self.height = height

        default_aug = RsnaBoneAgeDataModule.get_inference_augmentation(
            width, height, rotation_angle, flip
        )
        self.train_augment = train_augment if train_augment else default_aug
        self.valid_augment = valid_augment if valid_augment else default_aug
        self.test_augment = test_augment if test_augment else default_aug

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
            crop_to_mask=crop_to_mask,
            norm_method=img_norm_method,
            cache=cache,
        )
        self.mean, self.sd = self.train.get_norm()
        logger.info(
            f"Parameters used for bone age normalization: mean = {self.mean} - sd = {self.sd}"
        )
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
            crop_to_mask=crop_to_mask,
            norm_method=img_norm_method,
            cache=cache,
        )
        logger.info(f"====== ====== ====== ====== ======")
        logger.info(f"====== Setting up test data ======")
        self.test = RsnaBoneAgeKaggle(
            os.path.join(data_dir, "annotation_bone_age_test_data_set.csv"),
            os.path.join(data_dir, "bone_age_test_data_set"),
            mask_dir=mask_dir,
            data_augmentation=self.test_augment,
            bone_age_normalization=(self.mean, self.sd),
            crop_to_mask=crop_to_mask,
            norm_method=img_norm_method,
            cache=cache,
        )
        # self._assert_data_integrity()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
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

    @staticmethod
    def create_inference_module_from_args(args, rotation_angle=0, flip=False):
        return RsnaBoneAgeDataModule(
            width=args.input_width,
            height=args.input_height,
            batch_size=args.batch_size,  # likely this could be higher
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            mask_dir=args.mask_dirs,
            rotation_angle=rotation_angle,
            flip=flip,
            img_norm_method=args.img_norm_method,
        )

    @staticmethod
    def get_inference_augmentation(width=512, height=512, rotation_angle=0, flip=False):
        return A.Compose(
            [
                A.transforms.HorizontalFlip(p=flip),
                A.augmentations.geometric.transforms.Affine(
                    rotate=(rotation_angle, rotation_angle),
                    p=1.0,
                ),
                A.augmentations.crops.transforms.RandomResizedCrop(
                    width, height, scale=(1.0, 1.0), ratio=(1.0, 1.0)
                ),
                A.augmentations.transforms.ToFloat(max_value=2 ** 16, p=1),
                A.augmentations.transforms.RandomGamma((80, 120), p=1),
                ToTensorV2(),
            ],
            p=1,
        )


def add_data_augm_args(parent_parser):
    parent_parser.add_argument("--input_width", type=int, default=512)
    parent_parser.add_argument("--input_height", type=int, default=512)
    parser = parent_parser.add_argument_group("Data_Augmentation")
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--rotation_range", type=int, default=20)
    parser.add_argument("--relative_scale", type=float, default=0.2)
    parser.add_argument("--shear_percent", type=int, default=1)
    parser.add_argument("--translate_percent", type=int, default=0.2)
    parser.add_argument("--img_norm_method", type=str, default="zscore")
    parser.add_argument("--contrast_gamma", type=float, default=20)
    return parent_parser


def setup_training_augmentation(args):
    return A.Compose(
        [
            A.transforms.HorizontalFlip(p=args.flip_p),
            A.augmentations.geometric.transforms.Affine(
                scale=(1 - args.relative_scale, 1 / (1 - 0.2)),
                translate_percent=(-args.translate_percent, args.translate_percent),
                rotate=(-args.rotation_range, args.rotation_range),
                shear=(-args.shear_percent, args.shear_percent),
                p=1.0,
            ),
            A.augmentations.transforms.ToFloat(max_value=2 ** 16, always_apply=True),
            A.augmentations.transforms.RandomGamma(
                (100 - args.contrast_gamma, 100 + args.contrast_gamma),
                p=1,
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
    from matplotlib import pyplot as plt

    train_augment = A.Compose(
        [
            A.transforms.HorizontalFlip(p=0.5),
            A.augmentations.geometric.transforms.Affine(
                scale=(1 - 0.2, 1 / (1 - 0.2)),
                translate_percent=(-0.2, 0.2),
                rotate=(-25, 25),
                shear=(-5, 5),
                p=1.0,
            ),
            A.augmentations.transforms.ToFloat(max_value=2 ** 16, p=1),
            A.augmentations.transforms.RandomGamma((80, 120), p=1),
            A.augmentations.crops.transforms.RandomResizedCrop(
                512, 512, scale=(1.0, 1.0), ratio=(1.0, 1.0)
            ),
            ToTensorV2(),
        ],
        p=1,
    )
    # train_augment = RsnaBoneAgeDataModule.get_inference_augmentation()
    data_dir = "../data/annotated/rsna_bone_age/"
    train = RsnaBoneAgeKaggle(
        os.path.join(data_dir, "annotation_bone_age_training_data_set.csv"),
        os.path.join(data_dir, "bone_age_training_data_set"),
        data_augmentation=train_augment,
        bone_age_normalization=None,  # calculate renorm based on training set
        mask_dir=[
            "../data/masks/rsna_bone_age/tensormask",
            "../data/masks/rsna_bone_age/unet",
        ],
        crop_to_mask=False,
        norm_method="zscore",
        cache=False,
    )
    from time import time

    start = time()
    for i in range(20):
        img = train[np.random.randint(0, len(train))]["x"].numpy().squeeze()
        # plt.imshow(img, cmap="gray")
        # plt.show()
    print(time() - start)


if __name__ == "__main__":
    main()
