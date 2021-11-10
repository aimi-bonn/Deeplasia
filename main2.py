import logging
import sys, pickle, os

import pytorch_lightning as pl
import torch

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
    model = from_argparse(args)
    print(model.data.train[0]["x"])


if __name__ == "__main__":
    main()
