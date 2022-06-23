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
import sys
import torch
from torch import nn
import albumentations

import pandas as pd

sys.path.append("../..")
from lib import constants

# dir were to search for serialization obj, modified if needed
pickle_obj_dir = constants.path_to_pickle_dir
csv_df_dir = constants.path_to_csv_dir

# source: https://stackoverflow.com/questions/7507825/where-is-a-complete-example-of-logging-config-dictconfig
LOG_CONFIG = {
    # Always 1. Schema versioning may be added in a future release of logging
    "version": 1,
    # "Name of formatter" : {Formatter Config Dict}
    "formatters": {
        # Formatter Name
        "standard": {
            # class is always "logging.Formatter"
            "class": "logging.Formatter",
            # Optional: logging output format
            "format": "%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s",
            # Optional: asctime format
            "datefmt": "%d %b %y %H:%M:%S",
        }
    },
    # Handlers use the formatter names declared above
    "handlers": {
        # Name of handler
        "console": {
            # The class of logger. A mixture of logging.config.dictConfig() and
            # logger class-specific keyword arguments (kwargs) are passed in here.
            "class": "logging.StreamHandler",
            # This is the formatter name declared above
            "formatter": "standard",
            "level": "INFO",
            # The default is stderr
            # "stream": "ext://sys.stdout"
        },
        # Same as the StreamHandler example above, but with different  # not used
        # handler-specific kwargs.
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": "INFO",
            "filename": os.getenv("LOG_FILE", "run.log"),
            "mode": "a",
            "encoding": "utf-8",
            "maxBytes": 500000,
            "backupCount": 4,
        },
    },
    # Loggers use the handler names declared above
    "loggers": {
        "__main__": {  # if __name__ == "__main__"
            # Use a list even if one handler is used
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        }
    },
    # Just a standalone kwarg for the root logger
    "root": {"level": "INFO", "handlers": ["console", "file"],},
}


def set_logger(level=""):
    """Function to set up the handle error logging.
    logger (obj) = a logger object
    logLevel (str) = level of information to print out, options are
    {info, debug} [Default: info]
    """

    # Starting a logger
    logger = logging.getLogger()
    error = logging.ERROR

    # Determine log level
    if level == "debug":
        _level = logging.DEBUG
    else:
        _level = logging.INFO

    # Set the level in logger
    logger.setLevel(_level)

    # Set the log format
    log_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set logger output to STDOUT and STDERR
    log_handler = logging.StreamHandler(stream=sys.stdout)
    err_handler = logging.StreamHandler(stream=sys.stderr)

    # Set logging level for the different output handlers.
    # ERRORs to STDERR, and everything else to STDOUT
    log_handler.setLevel(_level)
    err_handler.setLevel(error)

    # Format the log handlers
    log_handler.setFormatter(log_fmt)
    err_handler.setFormatter(log_fmt)

    # Add handler to the main logger
    logger.addHandler(log_handler)
    logger.addHandler(err_handler)

    return logger


def change_log_output_dir(new_dir, name="run.log"):  # doesn't work for all logs
    os.makedirs(new_dir, exist_ok=True)
    new_fh = logging.FileHandler(os.path.join(new_dir, name), "a")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    new_fh.setFormatter(formatter)

    logger = logging.getLogger()  # root logger
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(new_fh)  # set the new handler
    logger.info(f"change dir to {new_dir}")


def log_system_info(logger):
    logger.info(f"Logs saved to {LOG_CONFIG['handlers']['file']['filename']}")
    logger.info(f"Core count             = {os.cpu_count()}")
    logger.info(f"Python version         = {sys.version}")
    logger.info(f"Pytorch version        = {torch.__version__}")
    logger.info(f"Albumentations version = {albumentations.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version           = {torch.version.cuda}")
        logger.info(f"CUDA count             = {torch.cuda.device_count()}")
        logger.info(f"CUDA name              = {torch.cuda.get_device_name()}")


def log_args(logger, args):
    logger.info("bla2")
    l = [f"{k} : {v}" for k, v in vars(args).items()]
    logger.info("/n".join(l))


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
    plt.figure(figsize=(17, 38))
    for i, img in enumerate(images):
        for j in range(n_examples):
            if j == 0:
                augmented = A.Compose([A.pytorch.ToTensorV2()])(image=img)
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
