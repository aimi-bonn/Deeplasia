import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import Swish, MemoryEfficientSwish

from lib.models.EfficientNet import VALID_MODELS, EfficientNet


class EfficientdBAM(nn.Module):
    def __init__(self, in_channels, num_classes=1, norm_type=None, act_type=MemoryEfficientSwish,
                 base='efficientnet-b0', pretrained=True):
        super(EfficientdBAM, self).__init__()

        assert act_type != nn.PReLU, "Currently doesn't support activation functions with trainable params"

        assert base in VALID_MODELS, f"Given base model type ({base}) is invalid"
        if pretrained:
            assert base != 'efficientnet-l2', "'efficientnet-l2' does not currently have pretrained weights"
            self.base = EfficientNet.from_pretrained(base, in_channels=in_channels)
        else:
            self.base = EfficientNet.from_name(base, in_channels=in_channels)

        ## Expected in size: [batch_size, 1280, 16, 16] = b0
        #self.dim_reduction_conv = nn.Conv2d(in_channels=1280, out_channels=64, kernel_size=1)

        self.act = act_type()
        self.dropout = nn.Dropout(p=0.2)

        self.pool = nn.AvgPool2d(kernel_size=16, stride=16)
        # Will be flattened afterwards to be of shape: B x 1280, rather than B x 1280 x 1 x 1
        # The B x 32 Gender will afterwards be concatenated to the other features
        # to be of shape B x 1312

        self.fc_gender_in = nn.Linear(1, 32)

        self.dense_blocks = nn.ModuleList()
        channel_sizes_in = [1312, 1024, 1024, 512, 512]

        for idx in range(len(channel_sizes_in)-1):
            self.dense_blocks.append(nn.Linear(in_features=channel_sizes_in[idx], out_features=channel_sizes_in[idx+1]))
            print(f"Dense block #{idx}, in_c: {channel_sizes_in[idx]}, out_c: {channel_sizes_in[idx+1]}")

        self.fc_boneage = nn.Linear(512, num_classes)

    def forward(self, x):

        # Split input if we also pass gender information
        x, gender = x
        gender = self.act(self.fc_gender_in(gender))

        x = self.base.extract_features(x, return_residual=False)   # PROBABLY don't need another activation here ...
        x = self.act(self.dropout(self.pool(x)))
        x = x.view(x.size(0), -1)           # expected size = B x 1280

        x = torch.cat([x, gender], dim=-1)  # expected size = B x 1312

        for mod in self.dense_blocks:
            x = self.act(self.dropout(mod(x)))

        boneage = self.fc_boneage(x)

        return boneage
