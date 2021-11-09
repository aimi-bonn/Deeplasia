import logging, logging.config
from lib.utils import LOG_CONFIG
logging.config.dictConfig(LOG_CONFIG)

import sys, pickle, os, yaml

import pytorch_lightning as pl

from argparse import ArgumentParser

sys.path.append("..")

from lib import constants, utils
from lib.models import *


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--random_seed", type=int, default=42)
    parser = add_model_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logger = logging.getLogger()
    utils.log_system_info(logger)

    logger.info(f"Command Line Args: {yaml.dump(vars(args))}")

    logger.info("bla")

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="logs/",
        name=args.name,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    output_dir = tb_logger.log_dir + "/ckp/"  # use tb to define log dir
    ckp_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir,
        filename="{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    new_dir = tb_logger.log_dir
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
