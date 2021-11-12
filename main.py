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

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="output/", name=args.name, log_graph=True, default_hp_metric=False
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    output_dir = tb_logger.log_dir + "/ckp/"  # use tb to define log dir
    ckp_callback = pl.callbacks.ModelCheckpoint(
        monitor="Loss/val_loss",
        dirpath=output_dir,
        filename="model-epoch_{epoch:03d}-val_loss={Loss/val_loss:.3f}",
        save_top_k=3,
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
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
    logger.info(f"===== Training finished ======")
    logger.info(f"Training time : {(time() - model.start_time) / 60:.2f}min")

    log_dict = {
        "hp/val_mad_months": 5,
        "hp/val_mad_months_reg": 2,
        "hp/test_mad_months": 42,
        "hp/test_mad_months_reg": 1.5,
    }
    model.logger.log_metrics(log_dict)

    test_loop()


if __name__ == "__main__":
    main()
