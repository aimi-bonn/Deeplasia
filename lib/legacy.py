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
        *args,
        **kwargs,
    ):
        super(ModelProto, self).__init__()  # might want to forward args and kwargs here

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        y_hat = self(batch["x"], batch["male"])
        return {"y_hat": y_hat, "image_path": batch["image_name"], "sex": batch["male"]}


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
        *args,
        **kwargs,
    ):
        super(InceptionDbam, self).__init__(
            *args,
            **kwargs,
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

        n_gender_dcs = 32
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
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, male):
        x = self.features(x)  # feature extraction
        x = torch.flatten(x, 1)  # flatten vector
        x = self.dropout(x)
        male = F.relu(self.male_fc(male.view(-1, 1)))  # make sure male bit is tensor
        x = torch.cat((x, male), dim=1)

        for mod in self.dense_blocks:
            x = F.relu(self.dropout(mod(x)))
        x = self.fc_boneage(x)
        return x


class EfficientDbam(ModelProto):
    def __init__(
        self,
        n_channels=1,
        pretrained=False,
        backbone="efficientnet-b0",
        dense_layers=[1024, 1024, 512, 512],
        *args,
        **kwargs,
    ):
        super(EfficientDbam, self).__init__(
            *args,
            **kwargs,
        )
        """
        Efficientnet based bone age model featuring a variable number of dense layers

        Args:
            n_channels: number of channels of the input image
            pretrained: used pretrained backbone model
            backbone: existing backbone model for faster instantiation
            dense_layers: number of neurons in the dense layers
        """
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
        n_gender_dcs = 32
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


def from_checkpoint(args):
    """
    create model from argparse
    """
    args.backbone = (
        "efficientnet-b4" if "effnet-b4" in args.ckp_path else "efficientnet-b0"
    )
    if args.backbone == "inception_v3":
        raise NotImplementedError("inception_v3 not yet implemented")
    d = load_weights(args.ckp_path, "dense_blocks")
    dense_layers = []
    for k, v in d.items():
        if "weight" in k:
            dense_layers.append(v.shape[0])
    if "efficient" in args.backbone:
        assert args.backbone in EfficientNet.VALID_MODELS
        model = EfficientDbam(
            act_type="mem_eff",
            backbone=args.backbone,
            dense_layers=dense_layers,
            input_size=tuple(args.input_size[1:]),
            n_in_channels=args.input_size[0],
        )
        model.load_state_dict(torch.load(args.ckp_path)["state_dict"])
    else:
        raise NotImplementedError
    return model


def load_weights(path: str, key="base"):
    """
    load part specified by key from the models stored at the path
    """
    key += "."
    weight_dict = (
        torch.load(path)
        if torch.cuda.is_available()
        else torch.load(path, map_location="cpu")
    )
    return {
        k.replace(key, ""): v for k, v in weight_dict["state_dict"].items() if key in k
    }
