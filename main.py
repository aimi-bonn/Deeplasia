import logging
import sys, pickle, os

import pytorch_lightning as pl

from argparse import ArgumentParser

sys.path.append("..")

from lib import constants, utils
from lib.models import *


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="debug")

    parser = add_model_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logger = utils.get_logger()
    logger.debug(f"using data from {constants.path_to_data_dir}")

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="logs/tb/",
        name=args.name,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ckp_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_logger.log_dir + "/ckp/",
        filename="{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    callbacks = [lr_monitor, ckp_callback]
    model = from_argparse(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        profiler="simple",
        logger=tb_logger,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
