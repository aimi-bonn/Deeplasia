# Bone Age

This repo is now updated to use [pytorch lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).
This included changing the exact design of the models so the obtained weights are not backward compatible.
To use the weights either use the older version (commit 606a7157) or the older inference scripts.
A dedicated script for inference will be provided integrating the whole workflow (including masking).

## Code formatting
Code is formatted using the [black](https://black.readthedocs.io/en/stable/) formatter.