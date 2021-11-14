import logging, logging.config
from lib.utils import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

import sys, pickle, os, yaml
import pytorch_lightning as pl
from argparse import ArgumentParser

sys.path.append("..")

from lib import constants, utils, testing
from lib.models import *


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--random_seed", type=int, default=42)
    parser = add_model_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = testing.add_eval_args(parser)
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
        filename="{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    gpu_stats = pl.callbacks.GPUStatsMonitor(memory_utilization=True, gpu_utilization=True)
    callbacks = [lr_monitor, ckp_callback, gpu_stats]
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

    log_dict = testing.evaluate_bone_age_model(
        ckp_callback.best_model_path, args, tb_logger.log_dir
    )
    model.logger.log_metrics(log_dict)
    logger.info(f"======= END =========")


if __name__ == "__main__":
    main()
