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

    logger.info(f"Command Line Args: /n{yaml.dump(vars(args))}")

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="output/", name=args.name, log_graph=True, default_hp_metric=False
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    output_dir = tb_logger.log_dir + "/ckp/"  # use tb to define log dir
    ckp_callback = pl.callbacks.ModelCheckpoint(
        monitor="Step-wise/val_mad" if args.age_sigma > 0 else "Step-wise/val_ROC",
        dirpath=output_dir,
        filename="best.ckpt",
        save_top_k=1,
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )
    callbacks = [lr_monitor, ckp_callback]
    if args.gpus:
        callbacks += [
            pl.callbacks.GPUStatsMonitor(memory_utilization=True, gpu_utilization=True)
        ]
    model = from_argparse(args)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, profiler="simple", logger=tb_logger,
    )
    trainer.fit(model)
    logger.info(f"===== Training finished ======")
    logger.info(
        f"Training time : {(time() - model.start_time) / 60:.2f}min for {model.global_step} steps of training"
    )
    logger.info(
        f"Training speed: {(model.global_step / (time() - model.start_time)):.2f}steps/second"
    )

    log_dict = testing.evaluate_bone_age_model(
        ckp_callback.best_model_path, args, tb_logger.log_dir
    )
    model.logger.log_metrics(log_dict)
    model.logger.save()
    logger.info(f"======= END =========")


if __name__ == "__main__":
    main()
