"""
Bundles general utility functionality not directly related with the actual analysis
"""

import logging
import pickle
from os import path

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from torch import nn

import pandas as pd

from lib import constants

# dir were to search for serialization obj, modified if needed
pickle_obj_dir = constants.path_to_pickle_dir
csv_df_dir = constants.path_to_csv_dir


def get_logger():
    """
    create and setup logger

    :return: logger
    """
    logger = logging.getLogger('lightning')
    logger.setLevel(logging.DEBUG)
    print(constants.path_to_log_dir)
    log_file = f"{constants.path_to_log_dir}/run.log"
    logging.basicConfig(
        filename=log_file,
        format="%(asctime)s: %(name)s - %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        filemode="w",
    )
    console_handle = logging.StreamHandler()
    console_handle.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%m-%d %H:%M:%S")
    console_handle.setFormatter(formatter)
    logger.addHandler(console_handle)

    # work around for the font warning in server
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    return logger


def restore_df_from_csv(name, func, *args, **kwargs):
    """
    Wrapper function to as a simple interface to DataFrame serialization. Use for relatively small DataFrames or data that needs to be exported to languages/tools other than python.

    The usage of this avoid needing to run computationally expensive function (also see :func:`~lib.utils.restore_obj_from_pickle()`)
    The function tries to restore the previously calculated DataFrame. If it is not available in the directory specified at pickle_parent_dir or if force_renew is set to True, the function is called with the inserted args.
    If the function is called, the resulting DataFrame is saved as the specified path (existing objects might be overwritten).
    Note, that the called functions needs to accept **kwargs even if it doesn't uses them.

    Exporting as .csv provides an exportable and human-readable DataFrame storage, however it is slower and creates bigger objects than storage as pickle.

    :Example:
        df_name = func(arg1, arg2)
        # same result with:
        df_name = restore_df_from_csv('df_name', func, False, logger, arg1, arg2)

    :param (string) name: name of pickle object in the specified csv_df_dir (without the suffix '.csv')
    :param (function) func: function to be called (returns a pandas.DataFrame)
    *args: args for the func()

    :returns: either restored pandas.DataFrame or the pandas.DataFrame obtained by the function call.

    Keyword Arguments:
        * *force_renew* (bool) -- if set to True (default: False), func() is called even if the corresponding obj is avaible (use whenever you update the function args or you want to obtain reproducible results)
        * *logger* (Logger) -- logger to write the modification date of the object
        * *kwargs* -- passes **kwargs to called function
    """

    force_renew, log, logger = extract_serialization_settings_from_kwargs(**kwargs)
    kwargs = {x: kwargs[x] for x in kwargs if x not in ["force_renew", "logger"]}
    df_path = path.join(csv_df_dir, name + ".csv")
    if path.exists(df_path) and not force_renew:
        if log:
            logger.info(
                "Restored DataFrame from "
                + name
                + " (last modified on {})".format(
                    time.strftime(
                        "%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(df_path))
                    )
                )
            )
        return pd.read_csv(df_path)
    else:
        df = func(*args, **kwargs)
        df.to_csv(df_path)
        if log:
            logger.info(
                time.strftime("%d/%m/%y-%H:%M:%S", time.localtime())
                + ": Saved DataFrame as "
                + name
            )
        return df


def restore_obj_from_pickle(name, func, *args, **kwargs):
    """
    Wrapper function to as a simple interface to DataFrame serialization. Use for all kind of (big) objects.

    The usage of this avoid needing to run computationally expensive function (also see :func:`~lib.utils.restore_df_from_csv() for DataFrames`)
    The function tries to restore the previously calculated object. If it is not available in the directory specified at pickle_parent_dir or if force_renew is set to True, the function is called with the inserted args.
    If the function is called, the resulting object is saved as the specified path (existing objects might be overwritten)

    :Example:
        df_name = func(arg1, arg2)
        # same result with:
        df_name = restore_obj_from_pickle('df_name', func, False, logger, arg1, arg2)

    :param (string) name: name of pickle object in the specified pickle_obj_dir (without the suffix '.pkl')
    :param (function) func: function to be called (returns any kind of object)
    *args: args for the func()

    :returns: either restored object or the object obtained by the function call.

    Keyword Arguments:
        * *force_renew* (bool) -- if set to True (default: False), func() is called even if the corresponding obj is avaible (use whenever you update the function args or you want to obtain reproducible results)
        * *logger* (Logger) -- logger to write the modification date of the object
        * *kwargs* -- passes **kwargs to called function
    """

    force_renew, log, logger = extract_serialization_settings_from_kwargs(**kwargs)
    pickle_path = path.join(pickle_obj_dir, name + ".pkl")
    kwargs = {x: kwargs[x] for x in kwargs if x not in ["force_renew", "logger"]}
    if path.exists(pickle_path) and not force_renew:
        if log:
            logger.info(
                "Restored object from pickle "
                + name
                + " (last modified on {})".format(
                    time.strftime(
                        "%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(pickle_path))
                    )
                )
            )
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    else:
        obj = func(*args, **kwargs)
        with open(pickle_path, "wb") as file:
            pickle.dump(obj, file)
        if log:
            logger.info(
                time.strftime("%d/%m/%y-%H:%M:%S", time.localtime())
                + ": Saved object as pickle "
                + name
            )
        return obj


def extract_serialization_settings_from_kwargs(**kwargs):
    """helper function to extract settings for serialization function calls"""
    force_renew = False
    log = False
    logger = None
    if "force_renew" in kwargs:
        force_renew = kwargs.get("force_renew")
    if "logger" in kwargs:
        log = True
        logger = kwargs.get("logger")
    return force_renew, log, logger


def plot_bone_age_examples(images, title="random images"):
    plt.figure()
    for j in range(3):  # inspect example images
        i = np.random.randint(0, len(images))
        for a in range(4):
            sample = images[i]  # generate new data augm
            ax = plt.subplot(3, 4, j * 4 + a + 1)
            img = np.array(sample["x"])[0, :, :]
            plt.imshow(img, cmap="gray")
            ax.axis("off")
    plt.suptitle(title)
    plt.show()
    
    
def visualize_augmentations(images, aug_transform, n_examples=6):
    plt.figure(figsize=(17,38))
    for i, img in enumerate(images):
        for j in range(n_examples):
            if j == 0:
                augmented = A.Compose([
                    A.pytorch.ToTensorV2()
                ])(image=img)
            else:
                augmented = aug_transform(image=img)
            sample = augmented["image"]
            ax = plt.subplot(len(image_paths), n_examples, i * n_examples + j + 1)
            sample = normalize_image("zscore", sample)
            sample = np.array(sample)[0, :, :]
            plt.imshow(sample, cmap="gray")
            ax.axis("off")
    return sample


# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
