"""
module for image transformation
"""
import torch
from torchvision.transforms import transforms


class BoneAgeDataAugmentation:
    """Wraps the data augmentation functionality of TensorFlow's 'ImageDataGenerator'
     as implemented in the 16Bit and dBAM models in TF.

    Args:
        augment: if False no augmentation takes place (e.g. for testing)
        output_tensor_size: Tuple(int, int). size of the output image as (height, width)
        horizontal_flip: Bool. Apply horizontal flip
        vertical_flip: Bool. Apply vertical flip
        height_shift_range: Float. Size of maximum relative height shifts
        width_shift_range: Float. Size of maximum relative width shifts
        rotation_range: Int. Degree range for random rotations.
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        normalization_method: String. Either 'zscore' or 'interval' (i.e. [0,1])
    """

    def __init__(
        self,
        augment=True,
        output_tensor_size=(500, 500),
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=20,
        shear_range=0.01,
        normalization_method="zscore",
    ):
        self.augment = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        output_tensor_size, scale=(1, 1), ratio=(1, 1)
                    ),
                    transforms.ToTensor(),
                ]
            )  # only scale image to the specified input size
            if not augment
            else transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=horizontal_flip / 2),
                    transforms.RandomVerticalFlip(p=vertical_flip / 2),
                    transforms.RandomAffine(degrees=rotation_range, shear=shear_range),
                    transforms.RandomResizedCrop(
                        size=output_tensor_size,
                        scale=(1 - width_shift_range, 1 - height_shift_range),
                        ratio=(1, 1),
                    ),
                    transforms.ToTensor(),
                ]
            )
        )
        self.normalization_method = normalization_method

    def __call__(self, x):
        """conducts the specified augmentations and renorm on the given PIL image
        and return a torch.Tensor"""
        x = self.augment(x)
        x = normalize_image(self.normalization_method, x)
        return x


def normalize_image(normalization_method, x):
    """conducts specified renorm"""
    if normalization_method == "zscore":
        assert (
            torch.std(x) > 1e-5
        ), f"something went wrong, std of provided image is {torch.std(x)}"
        x -= torch.mean(x)
        x /= torch.std(x)
    elif normalization_method == "interval":
        x -= x.min()
        x /= x.max()
    return x


def assert_unique_labels(id0, id1):
    duplicated = []
    id0, id1 = id0.astype(str), id1.astype(str)
    for i in id0:
        if i in id1:
            duplicated.append(i)
    if len(duplicated) == 0:
        return True
    else:
        print("Found {} duplicated IDs:".format(len(duplicated)))
        print(duplicated)
        return False
