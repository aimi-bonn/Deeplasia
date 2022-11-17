#! /bin/bash
NAME="model_name"

DATA_DIR="../data"

out_root=output/$NAME
mkdir output
mkdir "$out_root"
LOG_FILE="$out_root/version_0".log

LOG_FILE=$LOG_FILE python train_model.py \
    --name $NAME \
    --config=configs/xcat.yml \
    --data.img_dir="$DATA_DIR/annotated" \
    --data.mask_dirs=["$DATA_DIR/masks/tensormask", "$DATA_DIR/masks/unet"] \
    --config=configs/default_model.yml \
    --model.dense_layers=[256]