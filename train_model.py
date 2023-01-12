import logging, logging.config
from lib.utils.log import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

from lib.utils.cli import CustomCli
import sys

sys.path.append("..")

from lib import testing
from lib.models import *
from lib.datasets import *


def main():
    logger = logging.getLogger()
    cli = CustomCli(
        BoneAgeModel,
        HandDatamodule,
        run=False,
        parser_kwargs={"default_config_files": ["configs/defaults.yml"],},
    )
    cli.setup_callbacks()
    cli.log_info()
    try:
        cli.examples_to_tb()
        logger.info(f"{'=' * 10} start training {'=' * 10}")
        cli.trainer.fit(cli.model, cli.datamodule)
        cli.log_train_stats()

        logger.info(f"{'=' * 10} Testing model {'=' * 10}")
        test_ckp_path = cli.get_model_weights()
    except Exception:
        logger.exception("No training samples, testing only")
        test_ckp_path = cli.config["trainer"]["resume_from_checkpoint"]

    log_dict = testing.evaluate_bone_age_model(
        test_ckp_path, cli.config, cli.trainer.logger.log_dir, cli.trainer
    )
    cli.model.logger.log_metrics(log_dict)
    cli.model.logger.save()
    logger.info(f"======= END =========")


if __name__ == "__main__":
    main()
