"""
utility code without big impact of the main function of the actual model
"""
import torch
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchvision

import numpy as np

import warnings
import seaborn as sns

warnings.filterwarnings(
    action="ignore", category=UserWarning, message=r"FixedFormatter should only*"
)


def sample_batch_to_tb(
    writer: object, batch: dict, title: str = "example_batch", max_examples=16,
) -> None:
    """
    visualize a batch dictionary containing varying modalities and batch size
    """
    imgs = batch["x"][:max_examples]
    for i in range(imgs.shape[0]):
        imgs[i, 0, :, :] = imgs[i, 0, :, :] - imgs[i, 0, :, :].min()
        imgs[i, 0, :, :] = imgs[i, 0, :, :] / imgs[i, 0, :, :].max()
    grid = torchvision.utils.make_grid(imgs, 4)
    writer.add_image(title, grid, 0)


def confusion_matrix_to_tb(
    writer: object,
    step: int,
    y_hat: np.ndarray,
    y: np.ndarray,
    title: str = "confusion_matrix",
):
    """
    Visualization of confusion matrix
    Source: https://martin-mundt.com/tensorboard-figures/ and https://github.com/lanpa/tensorboardX/blob/master/examples/demo_matplotlib.py
    :param writer: TensorBoard SummaryWriter instance (LightningModule.logger.experiment).
    :param step: Counter usually specifying steps/epochs/time.
    :param y_hat: predictions
    :param y: gt
    :param title: title of the plot in tensorboard
    """
    # Create the figure
    fig = plt.figure()
    ax = plt.gca()

    # Show the matrix and define a discretized color bar
    ax.scatter(y, y_hat)
    ax.plot(y, y, "-r")

    ax.grid(False)
    plt.tight_layout()
    writer.add_figure(title, fig, step)


def save_confusion_matrix(matrix, output_path, class_names, title="", figsize=(9, 9)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    kwargs = dict(
        cmap="Blues",
        vmax=1,
        vmin=0,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    if np.all(matrix <= 1.0):
        sns.heatmap(matrix, annot=True, **kwargs)
    else:
        norm_mat = matrix / np.sum(matrix, axis=1)[:, np.newaxis]
        group_counts = ["{0:0.0f}".format(value) for value in matrix.flatten()]

        group_percentages = ["{0:.1%}".format(value) for value in norm_mat.flatten()]

        labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(norm_mat.shape[0], -1)
        sns.heatmap(norm_mat, annot=labels, fmt="", **kwargs)

    plt.tight_layout()
    if title:
        ax.set_title(title)

    plt.savefig(output_path)
