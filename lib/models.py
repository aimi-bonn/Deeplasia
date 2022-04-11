"""
module for custom functions used for creating and training the model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.models
import pytorch_lightning as pl
from time import time
import scipy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import lib.utils
from lib.effNet import EfficientNet

from lib import datasets
import logging
from lib.modules.metrics import *
from lib.modules.losses import *

logger = logging.getLogger(__name__)


class ModelProto(pl.LightningModule):
    """
    Abstract model specifying the loops, data, logging, etc. Should be overwritten by
     exact models.
    """

    def __init__(
        self,
        train_augment=None,
        valid_augment=None,
        test_augment=None,
        lr=1e-3,
        weight_decay=0,
        batch_size=32,
        num_workers=4,
        epoch_size=2048,
        data_dir=None,
        mask_dir=None,
        cache=False,
        img_norm_method="zscore",
        age_sigma=1,
        sex_sigma=0,
        learnable_sigma=False,
        min_lr=1e-4,
        rlrp_patience=5,
        rlrp_factor=0.1,
        *args,
        **kwargs,
    ):
        super(ModelProto, self).__init__()  # might want to forward args and kwargs here

        self.start_time = -1
        self.data_dir, self.mask_dir = data_dir, mask_dir
        logger.info(f"Setting up data from {data_dir}")
        self.example_input_array = self.get_example_input(kwargs)
        width, height = self.example_input_array[0].shape[2:4]
        self.data = datasets.RsnaBoneAgeDataModule(
            train_augment,
            valid_augment=valid_augment,
            test_augment=test_augment,
            batch_size=batch_size,
            num_workers=num_workers,
            epoch_size=epoch_size,
            data_dir=data_dir,
            mask_dir=mask_dir,
            width=width,
            height=height,
            img_norm_method=img_norm_method,
            cache=cache,
        )
        self.y_mean, self.y_sd = self.data.mean, self.data.sd
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.rlrp_factor = rlrp_factor
        self.rlrp_patience = rlrp_patience

        self.sex_sigma = sex_sigma
        self.age_sigma = age_sigma
        self.learnable_sigma = learnable_sigma

        self.slope = 1
        self.bias = 0  # TODO learnable param

        self.age_loss = torch.nn.MSELoss()
        self.sex_loss = torch.nn.BCEWithLogitsLoss()
        self.mad = torch.nn.L1Loss()

        sex_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.AUROC(num_classes=None, compute_on_step=False,),
                torchmetrics.F1Score(num_classes=None, compute_on_step=True),
            ]
        )
        self.sex_metrics = {
            "train": sex_metrics.clone(prefix="train_"),
            "val": sex_metrics.clone(prefix="val_"),
        }

        age_metrics = torchmetrics.MetricCollection(  # TODO MAE in months
            [
                torchmetrics.MeanAbsoluteError(compute_on_step=False),
                RescaledMAE(compute_on_step=False, rescale=self.y_sd),  # TODO debug
                torchmetrics.PearsonCorrCoef(compute_on_step=False),
                torchmetrics.MeanSquaredError(compute_on_step=False, squared=False),
            ]
        )
        self.age_metrics = {
            "train": age_metrics.clone(prefix="train_"),
            "val": age_metrics.clone(prefix="val_"),
        }

    def setup(self, stage):
        self.start_time = time()
        logger.info(f"====== start training ======")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.data.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.data.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.data.test_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.data.val_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.rlrp_factor,
            patience=self.rlrp_patience,
            min_lr=self.min_lr,
            verbose=True,
            cooldown=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Step-wise/val_mad"
                if self.age_sigma > 0
                else "Step-wise/val_ROC",
                "interval": "epoch",
            },
        }

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/validation_mad": -1,
                "hp/validation_mad_reg": -1,
                "hp/validation_mad_reg_tta": -1,
                "hp/test_mad": -1,
                "hp/test_mad_reg": -1,
                "hp/test_mad_reg_tta": -1,
            },
        )

        imgs = iter(self.train_dataloader()).next()["x"][:20]
        for i in range(imgs.shape[0]):
            imgs[i, 0, :, :] = imgs[i, 0, :, :] - imgs[i, 0, :, :].min()
            imgs[i, 0, :, :] = imgs[i, 0, :, :] / imgs[i, 0, :, :].max()
        grid = torchvision.utils.make_grid(imgs, 5)
        self.logger.experiment.add_image("train_batch", grid, 0)

    def training_step(self, batch, batch_idx):
        age_hat, sex_hat, loss, age_loss, sex_loss, _ = self._shared_step(
            batch, "train"
        )
        self.log("Loss/train_loss", loss, on_step=True, prog_bar=False, logger=True)
        self.log("Loss/train_age_loss", age_loss, on_step=True)
        self.log("Loss/train_sex_loss", sex_loss, on_step=True)
        return {
            "loss": loss,
            "n": batch["x"].shape[0],
            "age_hat": age_hat.detach(),
            "age": batch["y"],
        }

    def training_epoch_end(self, outputs):
        self._regression_plot(outputs, "Training Error")
        slope, bias = self.calculate_prediction_bias(outputs)
        log_dict = (
            {
                f"Metrics_Age/{k}": v
                for k, v in self.age_metrics["train"].compute().items()
            }
            | {
                f"Metrics_Sex/{k}": v
                for k, v in self.sex_metrics["train"].compute().items()
            }
            | {
                "Pred_bias/intercept_train": bias,
                "Pred_bias/slope_train": slope,
                "Loss/sex_sigma": self.sex_sigma,
                "Loss/age_sigma": self.age_sigma,
            }
        )
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        self.age_metrics["val"].reset()
        self.sex_metrics["val"].reset()
        self.logger.experiment.add_scalars(
            "Metrics_Age/MAD_months",
            {"train": log_dict["Metrics_Age/train_RescaledMAE"]},
            global_step=self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        age_hat, sex_hat, loss, age_loss, sex_loss, mad = self._shared_step(
            batch, "val"
        )
        self.log("Loss/val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("Loss/val_age_loss", age_loss, on_epoch=True)
        self.log("Loss/val_sex_loss", sex_loss, on_epoch=True)
        return {
            "mad": mad,
            "loss": loss,
            "n": batch["x"].shape[0],
            "age_hat": age_hat.detach(),
            "age": batch["y"],
        }

    def validation_epoch_end(self, outputs):
        _, epoch_mad = self._summarize_epoch(outputs)  # deprecated
        self._regression_plot(outputs, "Validation Error")
        slope, bias = self.calculate_prediction_bias(outputs)
        log_dict = (
            {
                f"Metrics_Age/{k}": v
                for k, v in self.age_metrics["val"].compute().items()
            }
            | {
                f"Metrics_Sex/{k}": v
                for k, v in self.sex_metrics["val"].compute().items()
            }
            | {
                "Metrics_Age/val_mad_own": epoch_mad,
                "Pred_bias/intercept_val": bias,
                "Pred_bias/slope_val": slope,
            }
        )
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        self.logger.experiment.add_scalars(
            "Metrics_Age/MAD_months",
            {"val": log_dict["Metrics_Age/val_RescaledMAE"]},
            global_step=self.current_epoch,
        )
        self.age_metrics["val"].reset()
        self.sex_metrics["val"].reset()
        self.log("Step-wise/val_mad", log_dict["Metrics_Age/val_RescaledMAE"])
        self.log("Step-wise/val_ROC", log_dict["Metrics_Sex/val_AUROC"])

    def _regression_plot(self, outputs, title):
        y_hat, y = self.retrieve_age_predictions(outputs)
        y_hat = y_hat * self.y_sd + self.y_mean
        y = y * self.y_sd + self.y_mean
        if self.current_epoch % 10 == 0 or self.current_epoch in [0, 1, 2, 5]:
            lib.utils.confusion_matrix_to_tb(
                self.logger.experiment, self.current_epoch, y_hat, y, title
            )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        # TODO include models regression params and restore?
        return self(batch["x"], batch["male"])

    def test_step(self, batch, batch_idx):
        x, male, y = batch.values()
        y_hat = self(x, y)
        loss = self.mad(y_hat, y)
        self.log("test_loss", loss)

    def _shared_step(self, batch, step_type="train"):
        x = batch["x"]
        male = batch["male"]
        age = batch["y"]
        age_hat, sex_hat = self.forward(x, male)
        age_hat = age_hat * self.slope + self.bias  # TODO

        age_loss = self.age_loss(age_hat, age)
        sex_loss = self.sex_loss(sex_hat, male)
        loss = sex_loss * self.sex_sigma + self.age_sigma * age_loss  # TODO
        mad = self.mad(age_hat, age).detach()  # deprecated

        self.sex_metrics[step_type].to(sex_hat.device)(sex_hat, male.to(int))
        self.age_metrics[step_type].to(age_hat.device)(age_hat, age)

        return age_hat, sex_hat, loss, age_loss, sex_loss, mad

    @staticmethod
    def retrieve_age_predictions(outputs):
        ys, y_hats = [], []
        for batch_dict in outputs:
            ys.append(batch_dict["age"])
            y_hats.append(batch_dict["age_hat"])
        ys = torch.cat(ys).cpu().numpy().squeeze()
        y_hats = torch.cat(y_hats).cpu().numpy().squeeze()
        return y_hats, ys

    @staticmethod
    def calculate_prediction_bias(outputs):
        ys, y_hats = [], []
        for batch_dict in outputs:
            ys.append(batch_dict["age"])
            y_hats.append(batch_dict["age_hat"])
        ys = torch.cat(ys).cpu().numpy().squeeze()
        y_hats = torch.cat(y_hats).cpu().numpy().squeeze()
        try:
            slope, intercept, _, _, _ = scipy.stats.linregress(y_hats, y_hats - ys)
        except:
            slope, intercept = (1, 0)
        return slope, intercept

    @staticmethod
    def _summarize_epoch(outputs):
        mads = []
        total_n = 0
        for batch_dict in outputs:
            n = batch_dict["n"]
            mads.append(batch_dict["mad"] * n)
            total_n += n
        epoch_mad = torch.stack(mads).sum() / total_n
        return 0, epoch_mad

    @staticmethod
    def get_example_input(kwargs):
        n_channels = (
            kwargs["n_input_channels"] if "n_input_channels" in kwargs.keys() else 1
        )
        width, height = (
            kwargs["input_size"] if "input_size" in kwargs.keys() else (512, 512)
        )
        return (
            torch.rand([1, n_channels, width, height]),
            torch.rand([1, 1]),
        )


class InceptionDbam(ModelProto):
    """
    InceptionV3 based model featuring a variable number of dense layers

    Args:
        n_channels: number of channels of the input image
        pretrained: used pretrained backbone model
        backbone: existing backbone model for faster instantiation
        dense_layers: number of neurons in the dense layers
    """

    def __init__(
        self,
        n_channels=1,
        pretrained=False,
        dense_layers=[1024, 1024, 512, 512],
        backbone="inceptionv3",
        bn_momentum=0.01,
        n_gender_dcs=32,
        *args,
        **kwargs,
    ):
        super(InceptionDbam, self).__init__(
            *args, **kwargs,
        )
        self.example_input_array = (
            torch.rand([1, n_channels, 512, 512]),
            torch.rand([1, 1]),
        )
        self.save_hyperparameters()
        if (
            backbone is None or backbone == "inceptionv3"
        ):  # allow passing in existing backbone
            feature_extractor = torchvision.models.inception_v3(
                pretrained=pretrained, aux_logits=False, init_weights=True
            )
            feature_extractor.aux_logits = False

            self.features = nn.Sequential(
                # delete layers after last pooling layer
                nn.Conv2d(n_channels, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32, eps=0.001),
                # relu?
                *list(feature_extractor.children())[1:-2],
            )
        else:
            self.features = backbone
        for mod in list(self.features.modules()):
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momemtum = bn_momentum  # previously 0.1

        self.male_fc = nn.Linear(1, n_gender_dcs)
        channel_sizes_in = [2048 + n_gender_dcs] + dense_layers

        self.dense_blocks = nn.ModuleList()
        for idx in range(len(channel_sizes_in) - 1):
            self.dense_blocks.append(
                nn.Linear(
                    in_features=channel_sizes_in[idx],
                    out_features=channel_sizes_in[idx + 1],
                )
            )
        self.fc_boneage = nn.Linear(channel_sizes_in[-1], 1)
        self.fc_sex = nn.Linear(channel_sizes_in[-1], 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, male):
        x = self.features(x)  # feature extraction
        x = torch.flatten(x, 1)  # flatten vector
        x = self.dropout(x)
        male = F.relu(self.male_fc(male.view(-1, 1)))  # make sure male bit is tensor
        x = torch.cat((x, male), dim=1)

        for mod in self.dense_blocks:
            x = F.relu(self.dropout(mod(x)))
        age_hat = self.fc_boneage(x)
        sex_hat = self.fc_sex(x)
        return age_hat, sex_hat


class EfficientDbam(ModelProto):
    def __init__(
        self,
        n_channels=1,
        pretrained=False,
        backbone="efficientnet-b0",
        dense_layers=[1024, 1024, 512, 512],
        n_gender_dcs=32,
        *args,
        **kwargs,
    ):
        """
        Efficientnet based bone age model featuring a variable number of dense layers

        Args:
            n_channels: number of channels of the input image
            pretrained: used pretrained backbone model
            backbone: existing backbone model for faster instantiation
            dense_layers: number of neurons in the dense layers
        """
        super(EfficientDbam, self).__init__(
            *args, **kwargs,
        )
        self.save_hyperparameters()
        assert (
            backbone in EfficientNet.VALID_MODELS
        ), f"Given base model type ({backbone}) is invalid"
        if pretrained:
            assert (
                backbone != "efficientnet-l2"
            ), "'efficientnet-l2' does not currently have pretrained weights"
            self.base = EfficientNet.EfficientNet.from_pretrained(
                backbone, in_channels=n_channels
            )
        else:
            self.base = EfficientNet.EfficientNet.from_name(
                backbone, in_channels=n_channels
            )
        act_type = kwargs["act_type"] if "act_type" in kwargs.keys() else "mem_eff"
        self.act = (
            EfficientNet.Swish()
            if act_type != "mem_eff"
            else EfficientNet.MemoryEfficientSwish()
        )
        self.dropout = nn.Dropout(p=0.2)
        self.fc_gender_in = nn.Linear(1, n_gender_dcs)

        self.dense_blocks = nn.ModuleList()
        features_dim = EfficientNet.FEATURE_DIMS[backbone]  # 2nd dim of feature tensor
        channel_sizes_in = [features_dim + n_gender_dcs] + dense_layers

        for idx in range(len(channel_sizes_in) - 1):
            self.dense_blocks.append(
                nn.Linear(
                    in_features=channel_sizes_in[idx],
                    out_features=channel_sizes_in[idx + 1],
                )
            )
        self.fc_boneage = nn.Linear(channel_sizes_in[-1], 1)
        self.fc_sex = nn.Linear(channel_sizes_in[-1], 1)

    def forward(self, x, male):
        x = self.base.extract_features(x, return_residual=False)
        x = torch.mean(x, dim=(2, 3))  # agnostic of the 3th and 4th dim (h,w)  # 1x1
        x = self.dropout(x)
        x = self.act(x)
        x = x.view(x.size(0), -1)

        male = self.act(self.fc_gender_in(male))
        x = torch.cat([x, male], dim=-1)  # expected size = B x 1312

        for mod in self.dense_blocks:
            x = self.act(self.dropout(mod(x)))
        age_hat = self.fc_boneage(x)
        sex_hat = self.fc_sex(x)
        return age_hat, sex_hat


def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("Model")
    parser.add_argument(
        "--model", type=str, help="define architecture (e.g. dbam_efficientnet-b0"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--n_input_channels", type=int, default=1)
    parser.add_argument("--dense_layers", nargs="+", default=[1024, 1024, 512, 512])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--act_type", type=str, default="mem_eff")
    parser.add_argument("--n_gender_dcs", type=int, default=32)
    parser.add_argument("--age_sigma", type=float, default=1)
    parser.add_argument("--sex_sigma", type=float, default=0)
    parser.add_argument("--learnable_sigma", action="store_true")

    parser = parent_parser.add_argument_group("Data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="path to dir containing the rsna bone age data and annotation",
    )
    parser.add_argument("--mask_dirs", nargs="+", default=None)
    parser.add_argument(
        "--cache_data",
        action="store_true",
        help="cache images in RAM (Note: takes more than 10GB of RAM)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epoch_size", type=int, default=0)

    parser = parent_parser.add_argument_group("ReduceLROnPlateau")
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--rlrp_patience", type=int, default=5)
    parser.add_argument("--rlrp_factor", type=float, default=0.1)

    return datasets.add_data_augm_args(parent_parser)


def from_argparse(args):
    """
    create model from argparse
    """
    dense, backbone = args.model.split("_")
    train_augment = datasets.setup_training_augmentation(args)
    if dense != "dbam":
        raise NotImplementedError
    proto_kwargs = {
        "pretrained": args.pretrained,
        "n_channel": args.n_input_channels,
        "dense_layers": [int(x) for x in args.dense_layers],
        "n_gender_dcs": args.n_gender_dcs,
        "age_sigma": args.age_sigma,
        "sex_sigma": args.sex_sigma,
        "learnable_sigma": args.learnable_sigma,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epoch_size": args.epoch_size,
        "train_augment": train_augment,
        "data_dir": args.data_dir,
        "mask_dir": args.mask_dirs,
        "input_size": (args.input_width, args.input_height),
        "n_input_channels": args.n_input_channels,
        "img_norm_method": args.img_norm_method,
        "cache": args.cache_data,
        "min_lr": args.min_lr,
        "rlrp_patience": args.rlrp_patience,
        "rlrp_factor": args.rlrp_factor,
    }
    if "efficient" in backbone:
        assert backbone in EfficientNet.VALID_MODELS
        return EfficientDbam(act_type="mem_eff", backbone=backbone, **proto_kwargs,)
    elif backbone == "inceptionv3":
        return InceptionDbam(**proto_kwargs,)
    else:
        raise NotImplementedError


def get_model_class(args):
    dense, backbone = args.model.split("_")
    if "efficientnet" in backbone:
        backbone = "efficientnet"
    MODEL_ARCHITECTURES = {
        "inceptionv3": InceptionDbam,
        "efficientnet": EfficientDbam,
    }
    return MODEL_ARCHITECTURES[backbone]
