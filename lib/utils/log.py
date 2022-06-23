"""
utility funcs for logging
"""

from os import path
import logging
import torch
import albumentations
import sys, os

sys.path.append("../..")

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
