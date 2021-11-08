"""
module for custom functions used for evaluation and validation of trained models
"""
import os

from PIL import Image, ImageFile
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
import torch
import pandas as pd

from lib import dataset


class RsnaBoneAgeTestAugmentationLoader(dataset.RsnaBoneAgeKaggle):
    """implements fixed interval rotations and flips for test time augmentation"""

    def __init__(
        self,
        annotation_path,
        img_dir,
        rotations=[0],
        horizontal_flip=False,
        vertical_flip=False,
    ):
        super(RsnaBoneAgeTestAugmentationLoader, self).__init__(
            annotation_path, img_dir
        )
        self.rotations = rotations
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.n_aug = len(rotations)
        if horizontal_flip:
            self.n_aug *= 2
        if vertical_flip:
            self.n_aug *= 2
     

    def __getitem__(self, index):
        """index needs to be an Int"""
         # TODO fix
        from torchvision.transforms import transforms
        self.data_augmentation = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (500, 500), scale=(1, 1), ratio=(1, 1)
                    ),
                    transforms.ToTensor(),
                ]
            )  # only scale image to the specified input size

        
        img = self.open_image(index)
        images = []
        for rot in self.rotations:
            rot_img = img.rotate(rot)
            images.append(rot_img)
            if self.horizontal_flip:
                images.append(PIL.ImageOps.mirror(rot_img))
            if self.vertical_flip:
                images.append(PIL.ImageOps.flip(rot_img))
            if self.vertical_flip and self.horizontal_flip:
                images.append(PIL.ImageOps.mirror(rot_img))

        images = torch.stack([self.data_augmentation(i) for i in images], dim=0)
        male = torch.Tensor([[self.male[index]]]).repeat(self.n_aug, 1)
        y = torch.Tensor([self.Y[index]])
        return {"x": images, "male": male, "y": y}

    
    def open_image(self, index):
        img_path = os.path.join(self.img_dir, f"{self.ids[index]}.png")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_path)
        assert (
            np.sum(image) != 0
        ), f"image with index {index} is all black (sum is {np.sum(image)})"
        assert (
            np.std(image) > 1e-5
        ), f"std of image with index {index} is close to zero ({np.std(image)})"
        return image


    def evaluate_model(self, model, renorm=(0, 1), device_loc="cpu"):
        """convenience function to test all images with all specified augmentations

        :param model: torch.nn.Model to evaluate
        :param renorm: (Float, Float). Mean and std used for bone age normalization
        :param device_loc: str. Localization to run evaluation (either 'cpu' or
        'cuda:{device #}')

        :return pandas.DataFrame containing the predictions (y_hat) in the original scale
        """
        model.to(device_loc)
        mu, sigma = renorm
        model.eval()
        with torch.set_grad_enabled(False):
            l = [
                np.array(
                    model(
                        alter_ego["x"].to(device_loc), alter_ego["male"].to(device_loc)
                    ).cpu()
                ).flatten()
                for alter_ego in self
            ]
        results = np.array(l) * sigma + mu
        results = pd.DataFrame(results, columns=self._get_augmentation_labels())
        results["y"] = self.get_ground_truth_labels()
        return results

    def get_ground_truth_labels(self):
        return self.Y

    def _get_augmentation_labels(self):
        l = []
        for rot in self.rotations:
            l.append(f"y_hat_rot_{rot}deg")
            if self.horizontal_flip:
                l.append(f"y_hat_rot_{rot}deg_horFlip")
            if self.vertical_flip:
                l.append(f"y_hat_rot_{rot}deg_vertFlip")
            if self.vertical_flip and self.horizontal_flip:
                l.append(f"y_hat_rot_{rot}deg_vert_hor_Flip")
        return l

def predict_from_loader(model, data_loader, on_cpu=False):
    device_loc = "cpu"
    if not on_cpu:
        model.cuda()
        device_loc = "cuda"
    model.eval()
    y_hats = []
    ys = []
    image_names = []
    with torch.set_grad_enabled(False):
        for batch in data_loader: 
            y_hats.append(model(batch["x"].to(device_loc), batch["male"].to(device_loc)).cpu().squeeze())
            ys.append(batch["y"])
            image_names.append(batch["image_name"])
    ys = torch.cat(ys).squeeze().numpy()
    y_hats = torch.cat(y_hats).numpy()
    image_names = np.array([name for batch in image_names for name in batch])
    return y_hats, ys, image_names


def evaluate_predictions(
    df,
    y_col="boneage",
    yhat_col="pred_age",
    error_func=mean_absolute_error,
    title="add Title",
    error_pos_x=0,
    error_pos_y=220,
    y_label="Y",
    yhat_label="Y_hat",
    error_name="error",
):
    """
    Quick overview over model performance.

    Renders a scatter plot of y (real value) against yhat (predicted value) and prints the value of the defined error function.

    :param df: DataFrame containing y and yhat in specified columns
    :param error_func: Error function
    :param title: title of the plot
    :param error_pos_x: x position where the performance metrics is placed
    :param error_pos_y: y position where the performance metrics is placed
    :param y_label: name for y (real value) as axis label
    :param yhat_label: name for yhat (predicted value) as axis label
    :param error_name: Name of the error function in the plot
    """
    error = error_func(df[y_col], df[yhat_col])

    plt.figure()
    plt.scatter(df[y_col], df[yhat_col], alpha=0.3)
    plt.xlabel(y_label)
    plt.ylabel(yhat_label)
    plt.suptitle(title)
    plt.plot(
        df[y_col],
        df[y_col],
        "b-",
        color="red",
    )
    plt.text(error_pos_x, error_pos_y, "MAD={:.2f}".format(error))
    plt.show()
    print("Exact " + error_name + ": {}".format(error))
    return error


def calc_prediction_bias(y, yhat, verbose=True):
    """calculates the predicted signed error (yhat - y) for each prediction and performs a linear regression"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(yhat, yhat - y)
    if verbose:
        print(
            "Linear bias prediction:\nslope: {:.4f}\nintercept: {:.4f}\nr = {:.4f}\np-value = {:.1E}\nstd error = {:.1E}".format(
                slope, intercept, r_value, p_value, std_err
            )
        )
    return slope, intercept, r_value, p_value, std_err


def cor_prediction_bias(yhat, slope, intercept):
    """corrects model predictions (yhat) for linear bias (defined by slope and intercept)"""
    return yhat - (yhat * slope + intercept)
