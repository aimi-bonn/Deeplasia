"""
module for custom functions used for creating and training the model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models
import pytorch_lightning as pl
from time import time
import scipy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lib.effNet import EfficientNet

from lib import datasets
import logging

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
        batch_size=32,
        num_workers=4,
        epoch_size=2048,
        data_dir=None,
        mask_dir=None,
        *args,
        **kwargs,
    ):
        super(ModelProto, self).__init__(*args, **kwargs)
        self.criterion = torch.nn.MSELoss()
        self.mad = torch.nn.L1Loss()
        self.start_time = -1
        self.data = datasets.RsnaBoneAgeDataModule(
            train_augment,
            valid_augment=valid_augment,
            test_augment=test_augment,
            batch_size=batch_size,
            num_workers=num_workers,
            epoch_size=epoch_size,
            data_dir=data_dir,
            mask_dir=mask_dir,
        )
        self.sd = self.data.sd  # used for correct MAD scaling
        self.lr = lr

    def setup(self, stage):
        self.start_time = time()

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
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,
            patience=10,
            min_lr=1e-4,
            verbose=True,
            cooldown=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss/val_loss",
                "interval": "epoch",
            },
        }

    def training_step(self, batch, batch_idx):
        loss, mad, yhat = self._shared_step(batch)
        self.log(
            "Loss/train_loss_step", loss, on_step=True, prog_bar=False, logger=True
        )
        return {"loss": loss, "mad": mad, "n": batch["x"].shape[0]}

    def training_epoch_end(self, outputs):
        epoch_loss, epoch_mad = self._summarize_epoch(outputs)
        log_dict = {
            "Loss/train_loss_epoch": epoch_loss,
            "Accuracy/train_mad": epoch_mad,
            "Accuracy/train_mad_months": epoch_mad * self.sd,
        }
        wall_time = time() - self.start_time
        wall_time = round(wall_time / 60)
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        # self.logger.experiment.add_scalars(
        #     "Loss/losses", {"train": epoch_loss}, global_step=self.current_epoch
        # )
        self.logger.experiment.add_scalars(
            "Accuracy/MAD_months",
            {"train": epoch_mad * self.sd},
            global_step=self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        loss, mad, yhat = self._shared_step(batch)
        self.log("Loss/val_loss", loss, on_epoch=True)
        return {
            "loss": loss,
            "mad": mad,
            "n": batch["x"].shape[0],
            "y_hat": yhat,
            "y": batch["y"],
        }

    def validation_epoch_end(self, outputs):
        epoch_loss, epoch_mad = self._summarize_epoch(outputs)
        slope, intercept = self.calculate_prediction_bias(outputs)
        log_dict = {
            "Loss/val_loss_epoch": epoch_loss,
            "Acurracy/val_mad": epoch_mad,
            "Acurracy/val_mad_months": epoch_mad * self.sd,
            "Pred_bias/intercept": intercept,
            "Pred_bias/slope": slope,
        }
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        # self.logger.experiment.add_scalars(
        #     "losses", {"val": epoch_loss}, global_step=self.current_epoch
        # )
        self.logger.experiment.add_scalars(
            "Accuracy/MAD_months",
            {"val": epoch_mad * self.sd},
            global_step=self.current_epoch,
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch["x"], batch["male"])

    def test_step(self, batch, batch_idx):
        x, male, y = batch.values()
        y_hat = self(x, y)
        loss = self.mad(y_hat, y)
        self.log("test_loss", loss)

    def _shared_step(self, batch):
        x = batch["x"]
        male = batch["male"]
        y = batch["y"]
        y_hat = self.forward(x, male)
        loss = self.criterion(y_hat, y)
        mad = self.mad(y_hat, y).detach()
        return loss, mad, y_hat

    @staticmethod
    def calculate_prediction_bias(outputs):
        ys, y_hats = [], []
        for batch_dict in outputs:
            ys.append(batch_dict["y"])
            y_hats.append(batch_dict["y_hat"])
        ys = torch.cat(ys).cpu().numpy().squeeze()
        y_hats = torch.cat(y_hats).cpu().numpy().squeeze()
        slope, intercept, _, _, _ = scipy.stats.linregress(y_hats, y_hats - ys)
        return slope, intercept

    @staticmethod
    def _summarize_epoch(outputs):
        losses = []
        mads = []
        total_n = 0
        for batch_dict in outputs:
            n = batch_dict["n"]
            losses.append(batch_dict["loss"] * n)
            mads.append(batch_dict["mad"] * n)
            total_n += n
        epoch_loss = torch.stack(losses).sum() / total_n
        epoch_mad = torch.stack(mads).sum() / total_n
        return epoch_loss, epoch_mad


class InceptionDbam(ModelProto):
    """
    InceptionV3 based model featuring 4 dense layers for regression

    Args:
        n_channels: number of channels of the input image
        pretrained: used pretrained backbone model
        backbone: existing backbone model for faster instantiation
    """

    def __init__(
        self,
        n_channels=1,
        pretrained=False,
        backbone=None,
        *args,
        **kwargs,
    ):
        super(InceptionDbam, self).__init__(
            *args,
            **kwargs,
        )
        self.save_hyperparameters()
        if backbone is None:
            feature_extractor = torchvision.models.inception_v3(
                pretrained=pretrained, aux_logits=False, init_weights=True
            )
            feature_extractor.aux_logits = False

            self.features = nn.Sequential(
                # delete layers after last pooling layer
                nn.Conv2d(n_channels, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32, eps=0.001),
                *list(feature_extractor.children())[1:-2],
            )
        else:
            self.features = backbone
        for mod in list(self.features.modules()):
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momemtum = 0.01  # previously 0.1
        self.dropout1 = nn.Dropout(p=0.2)
        self.male_fc = nn.Linear(1, 32)
        self.fc1 = nn.Linear(2048 + 32, 1024)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, 512)
        self.dropout5 = nn.Dropout(p=0.2)
        self.out = nn.Linear(512, 1)

    def forward(self, x, male):
        x = self.features(x)  # feature extraction
        x = torch.flatten(x, 1)  # flatten vector
        x = self.dropout1(x)
        male = F.relu(self.male_fc(male.view(-1, 1)))  # make sure male bit is tensor
        x = torch.cat((x, male), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.dropout4(x)
        x = F.relu(self.fc4(x))
        x = self.dropout5(x)
        x = self.out(x)
        return x


class EfficientDbam(ModelProto):
    def __init__(
        self,
        n_channels=1,
        pretrained=False,
        act_type="mem_eff",
        base="efficientnet-b0",
        *args,
        **kwargs,
    ):
        super(EfficientDbam, self).__init__(
            *args,
            **kwargs,
        )
        self.save_hyperparameters()
        assert (
            base in EfficientNet.VALID_MODELS
        ), f"Given base model type ({base}) is invalid"
        if pretrained:
            assert (
                base != "efficientnet-l2"
            ), "'efficientnet-l2' does not currently have pretrained weights"
            self.base = EfficientNet.EfficientNet.from_pretrained(
                base, in_channels=n_channels
            )
        else:
            self.base = EfficientNet.EfficientNet.from_name(
                base, in_channels=n_channels
            )
        self.act = (
            EfficientNet.Swish()
            if act_type != "mem_eff"
            else EfficientNet.MemoryEfficientSwish()
        )
        self.dropout = nn.Dropout(p=0.2)
        n_gender_dcs = 32
        self.fc_gender_in = nn.Linear(1, n_gender_dcs)

        self.dense_blocks = nn.ModuleList()
        features_dim = EfficientNet.FEATURE_DIMS[base]  # 2nd dim of feature tensor
        channel_sizes_in = [features_dim + n_gender_dcs, 1024, 1024, 512, 512]

        for idx in range(len(channel_sizes_in) - 1):
            self.dense_blocks.append(
                nn.Linear(
                    in_features=channel_sizes_in[idx],
                    out_features=channel_sizes_in[idx + 1],
                )
            )
        self.fc_boneage = nn.Linear(512, 1)

    def forward(self, x, male):
        x = self.base.extract_features(x, return_residual=False)
        x = torch.mean(x, dim=(2, 3))  # agnostic of the 3th and 4th dim (h,w)
        x = self.dropout(x)
        x = self.act(x)
        x = x.view(x.size(0), -1)

        male = self.act(self.fc_gender_in(male))
        x = torch.cat([x, male], dim=-1)  # expected size = B x 1312

        for mod in self.dense_blocks:
            x = self.act(self.dropout(mod(x)))
        x = self.fc_boneage(x)
        return x


def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("Model")
    parser.add_argument(
        "--model", type=str, help="define architecture (e.g. dbam_efficientnet-b0"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="path to dir containing the rsna bone age data and annotation",
    )
    parser.add_argument("--mask_dirs", nargs="+", default=None)
    parser.add_argument("--n_input_channels", type=int, default=1)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epoch_size", type=int, default=0)
    parser.add_argument("--act_type", type=str, default="mem_eff")
    return datasets.add_data_augm_args(parent_parser)


def from_argparse(args):
    """
    create model from argparse
    """
    dense, backbone = args.model.split("_")
    train_augment = datasets.setup_augmentation(args)
    if dense != "dbam":
        raise NotImplementedError
    proto_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epoch_size": args.epoch_size,
        "train_augment": train_augment,
        "data_dir": args.data_dir,
        "mask_dir": args.mask_dirs,
    }
    if "efficient" in backbone:
        assert backbone in EfficientNet.VALID_MODELS
        return EfficientDbam(
            n_channels=args.n_input_channels,
            pretrained=args.pretrained,
            act_type="mem_eff",
            base=backbone,
            **proto_kwargs,
        )
    elif backbone == "inceptionv3":
        return InceptionDbam(
            n_channels=args.n_input_channels,
            pretrained=args.pretrained,
            **proto_kwargs,
        )
    else:
        raise NotImplementedError
