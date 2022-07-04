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
from lib.utils.visualize import confusion_matrix_to_tb

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    def __init__(self, age_sigma, sex_sigma, learnable=False):
        super(MultiTaskLoss, self).__init__()
        self.age_loss = torch.nn.MSELoss()
        self.sex_loss = torch.nn.BCEWithLogitsLoss()
        self.learnable = learnable

        self.age_sigma = age_sigma
        self.sex_sigma = sex_sigma
        if self.learnable:
            self.s = torch.nn.Parameter(torch.tensor([age_sigma, sex_sigma]))

    def forward(self, age_hat, age, sex_hat, sex):
        age_loss = self.age_loss(age_hat, age)
        sex_loss = self.sex_loss(sex_hat, sex)

        if self.learnable:
            total_loss = self._uncert_sum(age_loss, sex_loss)
        else:
            total_loss = self.age_sigma * age_loss + self.sex_sigma * sex_loss
        return total_loss, age_loss.detach(), sex_loss.detach()

    def _uncert_sum(self, age_loss, sex_loss):
        sigma_sq = torch.exp(self.s)
        age_weight = 1 / sigma_sq[0] / 2
        sex_weight = 1 / sigma_sq[1]
        return (
            age_loss * age_weight
            + sex_loss * sex_weight
            + torch.sum(torch.log(sigma_sq))
        )

    def log_dict(self):
        return {
            "Loss/ucw_log_var_age": self.s[0],
            "Loss/ucw_log_var_sex": self.s[1],
        }


class BoneAgeModel(pl.LightningModule):
    """
    Ptl CLI configurable Bone age model consisting of a backbone and dense network
    """

    def __init__(
        self,
        backbone: Union[str, bool] = "efficientnet-b0",
        pretrained: Union[bool, str] = None,
        dense_layers: List[int] = [256],
        sex_dcs: int = 32,
        explicit_sex_classifier: List[int] = None,
        correct_predicted_sex: bool = False,
        age_sigma: float = 1,
        sex_sigma: float = 0,
        learnable_sigma: bool = False,
        dropout_p: float = 0.2,
        batch_size: int = 32,  # linked
        masked_input: bool = True,  # linked
        input_size: List[int] = [1, 512, 512],  # linked
        name: str = "name",  # linked
        age_mean: float = 0,  # linked
        age_std: float = 1,  # linked
        img_norm_method: str = "zscore",
        *args,
        **kwargs,
    ):
        super(BoneAgeModel, self).__init__()
        if correct_predicted_sex and not explicit_sex_classifier and sex_sigma:
            logger.warning("the configuration will predict sex which is also an input!")

        self.start_time = -1
        self.save_hyperparameters(ignore=["age_mean", "age_std"])
        self.age_mean, self.age_std = (age_mean, age_std)

        self.sex_sigma = sex_sigma
        self.age_sigma = age_sigma
        self.slope, self.bias = (0, 0)

        self.example_input_array = self.get_example_input(input_size, batch_size)
        self._build_model(
            backbone=backbone,
            dense_layers=dense_layers,
            sex_dcs=sex_dcs,
            explicit_sex_classifier=explicit_sex_classifier,
            correct_predicted_sex=correct_predicted_sex,
            input_size=input_size,
            pretrained=pretrained,
            dropout_p=dropout_p,
        )
        self.loss = MultiTaskLoss(age_sigma, sex_sigma, learnable_sigma)
        self._create_metrics()

    def _build_model(
        self,
        backbone: str = "efficientnet-b0",
        dense_layers: List[int] = [1024, 1024, 512, 512],
        sex_dcs: int = 32,
        explicit_sex_classifier: List[int] = [],
        correct_predicted_sex: bool = False,
        input_size=(1, 512, 512),
        pretrained=False,
        dropout_p: float = 0.2,
        act_type="mem_eff",
    ):
        if "efficientnet" in backbone:
            assert backbone in EfficientNet.VALID_MODELS
            self.backbone = EfficientnetBackbone(
                backbone=backbone, pretrained=pretrained, act_type=act_type
            )
        elif backbone == "inceptionv3":
            self.backbone = Inceptionv3Backbone()

        with torch.no_grad():
            feature_dim = self.backbone.forward(torch.rand([1, *input_size])).shape[-1]

        self.dense = DenseNetwork(
            feature_dim,
            dense_layers=dense_layers,
            sex_dcs=sex_dcs,
            explicit_sex_classifier=explicit_sex_classifier,
            correct_sex=correct_predicted_sex,
            dropout_p=dropout_p,
        )

    def forward(self, x, male):
        features = self.backbone.forward(x)
        age_hat, male_hat = self.dense(features, male)
        return age_hat, male_hat

    def _create_metrics(self):
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
        age_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanAbsoluteError(compute_on_step=False),
                RescaledMAE(compute_on_step=False, rescale=self.age_std),
                torchmetrics.PearsonCorrCoef(compute_on_step=False),
                torchmetrics.MeanSquaredError(compute_on_step=False, squared=False),
            ]
        )
        self.age_metrics = {
            "train": age_metrics.clone(prefix="train_"),
            "val": age_metrics.clone(prefix="val_"),
            "train_reg": age_metrics.clone(prefix="train_reg_"),
            "val_reg": age_metrics.clone(prefix="val_reg_"),
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["age_std"] = self.age_std
        checkpoint["age_mean"] = self.age_mean
        checkpoint["bias"] = self.bias
        checkpoint["slope"] = self.slope

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.age_std = checkpoint["age_std"]
        self.age_mean = checkpoint["age_mean"]
        self.bias = checkpoint["bias"]
        self.slope = checkpoint["slope"]

    def setup(self, stage):
        self.start_time = time()

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

    def training_step(self, batch, batch_idx):
        d = self._shared_step(batch, "train")
        if self.loss.learnable:
            self.log_dict(self.loss.log_dict())
        return d

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
                f"Metrics_Age_Reg/{k}": v
                for k, v in self.age_metrics["train_reg"].compute().items()
            }
            | {
                "Pred_bias/intercept_train": bias,
                "Pred_bias/slope_train": slope,
                "Loss/sex_sigma": self.sex_sigma,
                "Loss/age_sigma": self.age_sigma,
            }
        )
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        self.age_metrics["train"].reset()
        self.age_metrics["train_reg"].reset()
        self.sex_metrics["train"].reset()
        self.logger.experiment.add_scalars(
            "Metrics_Age/MAD_months",
            {"train": log_dict["Metrics_Age/train_RescaledMAE"]},
            global_step=self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        self._regression_plot(outputs, "Validation Error")
        if self.current_epoch > 5:
            self.slope, self.bias = self.calculate_prediction_bias(outputs)
        log_dict = (
            {
                f"Metrics_Age/{k}": v
                for k, v in self.age_metrics["val"].compute().items()
            }
            | {
                f"Metrics_Age_Reg/{k}": v
                for k, v in self.age_metrics["val_reg"].compute().items()
            }
            | {
                f"Metrics_Sex/{k}": v
                for k, v in self.sex_metrics["val"].compute().items()
            }
            | {"Pred_bias/intercept_val": self.bias, "Pred_bias/slope_val": self.slope,}
        )
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        self.logger.experiment.add_scalars(
            "Metrics_Age/MAD_months",
            {"val": log_dict["Metrics_Age/val_RescaledMAE"]},
            global_step=self.current_epoch,
        )
        self.age_metrics["val"].reset()
        self.age_metrics["val_reg"].reset()
        self.sex_metrics["val"].reset()
        self.log("Step-wise/val_mad", log_dict["Metrics_Age/val_RescaledMAE"])
        self.log(
            "Step-wise/val_reg_mad", log_dict["Metrics_Age_Reg/val_reg_RescaledMAE"]
        )
        self.log("Step-wise/val_ROC", log_dict["Metrics_Sex/val_AUROC"])

    def _regression_plot(self, outputs, title):
        y_hat, y = self.retrieve_age_predictions(outputs)
        y_hat = y_hat * self.age_std + self.age_mean
        y = y * self.age_std + self.age_mean
        if self.current_epoch % 10 == 0 or self.current_epoch in [0, 1, 2, 5]:
            confusion_matrix_to_tb(
                self.logger.experiment, self.current_epoch, y_hat, y, title
            )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch["x"], batch["male"])

    def test_step(self, batch, batch_idx):
        x, male, y = batch.values()
        y_hat = self(x, y)
        loss = self.mad(y_hat, y)
        self.log("test_loss", loss)

    def _shared_step(self, batch, step_type="train"):
        x = batch["x"]
        male = batch["male"]
        age = batch["bone_age"]
        age_hat, sex_hat = self.forward(x, male)

        loss, age_loss, sex_loss = self.loss(age_hat, age, sex_hat, male)

        age_hat_cor = self.cor_prediction_bias(age_hat)
        self.age_metrics[step_type].to(age_hat.device)(age_hat, age)
        self.age_metrics[step_type + "_reg"].to(age_hat.device)(age_hat_cor, age)
        self.sex_metrics[step_type].to(sex_hat.device)(sex_hat, male.to(int))

        return {
            "loss": loss,
            "age_loss": age_loss,
            "sex_loss": sex_loss,
            "n": batch["x"].shape[0],
            "age_hat": age_hat.detach(),
            "age": age,
        }

    def cor_prediction_bias(self, yhat):
        """corrects model predictions (yhat) for linear bias (defined by slope and intercept)"""
        return yhat - (yhat * self.slope + self.bias)

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
            slope, intercept = (0, 0)
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
    def get_example_input(input_size, batch_size):
        batch_size = min(4, batch_size)
        return (
            torch.rand([batch_size, *input_size]),
            torch.rand([batch_size, 1]),
        )


class Inceptionv3Backbone(nn.Module):
    """
    InceptionV3 based model featuring a variable number of dense layers

    Args:
        n_channels: number of channels of the input image
        pretrained: used pretrained backbone model
        backbone: existing backbone model for faster instantiation
        dense_layers: number of neurons in the dense layers
    """

    def __init__(
        self, n_channels=1, pretrained=False, *args, **kwargs,
    ):
        super(Inceptionv3Backbone, self).__init__(
            *args, **kwargs,
        )
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
        for mod in list(self.features.modules()):
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momemtum = 0.01

    def forward(self, x):
        x = self.features(x)  # feature extraction
        x = torch.flatten(x, 1)  # flatten vector
        return x


class EfficientnetBackbone(nn.Module):
    def __init__(
        self,
        n_channels=1,
        pretrained=False,
        backbone="efficientnet-b0",
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
        super(EfficientnetBackbone, self).__init__()
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
        self.base._fc = None  # not used

    def forward(self, x):
        x = self.base.extract_features(x, return_residual=False)
        x = torch.mean(x, dim=(2, 3))  # agnostic of the 3th and 4th dim (h,w)  # 1x1
        return x


class DenseNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        dense_layers,
        sex_dcs,
        explicit_sex_classifier,
        correct_sex=True,
        dropout_p=0.2,
    ):
        super(DenseNetwork, self).__init__()

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_gender_in = nn.Linear(1, sex_dcs)
        self.act = nn.ReLU()
        self.input_dim = input_dim
        self.correct_sex = correct_sex

        self.dense_blocks = nn.ModuleList()
        channel_sizes_in = [self.input_dim + sex_dcs] + dense_layers
        for idx in range(len(channel_sizes_in) - 1):
            self.dense_blocks.append(
                nn.Linear(
                    in_features=channel_sizes_in[idx],
                    out_features=channel_sizes_in[idx + 1],
                )
            )
        self.fc_boneage = nn.Linear(channel_sizes_in[-1], 1)
        self.fc_sex = nn.Linear(channel_sizes_in[-1], 1)

        self.explicit_sex_classifier = None
        if explicit_sex_classifier:
            channel_sizes_in = [self.input_dim] + explicit_sex_classifier
            self.explicit_sex_classifier = nn.ModuleList()
            for idx in range(len(channel_sizes_in) - 1):
                self.explicit_sex_classifier.append(
                    nn.Linear(
                        in_features=channel_sizes_in[idx],
                        out_features=channel_sizes_in[idx + 1],
                    )
                )
            self.fc_sex = nn.Linear(channel_sizes_in[-1], 1)

    def forward(self, features, male):
        if self.explicit_sex_classifier is not None:
            sex_hat = features
            for layer in self.explicit_sex_classifier:
                sex_hat = self.act(self.dropout(layer(sex_hat)))
            sex_hat = self.fc_sex(sex_hat)
            male = (
                male if self.correct_sex and male is not None else sex_hat.detach()
            )  # detach because we want to have male as constant

        male = self.act(self.fc_gender_in(male))
        x = torch.cat([features, male], dim=-1)

        for layer in self.dense_blocks:
            x = self.act(self.dropout(layer(x)))
        age_hat = self.fc_boneage(x)

        if not self.explicit_sex_classifier:
            sex_hat = self.fc_sex(x)
        return age_hat, sex_hat
