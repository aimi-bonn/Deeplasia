NAME="effnet_default"
LOG_FILE=output/$NAME/"version_0.log"
mkdir output/$NAME

PARAMS=(
    --name $NAME
    --gpus 1
    --precision 16
    --cache_data
    --max_time 00:09:00:00
    --input_height 512
    --input_width 512
    --num_workers 16
    --max_epoch 100
    --batch_size 128
    --model dbam_efficientnet-b0
    --mask_dir "../TensorMask/output/Colab_train/inference/bone_age" "../TensorMask/output/Colab_train/inference_alexander/bone_age_train"
    --cache_data
    --lr 1e-3
    --min_lr 1e-4
    --rlrp_factor 0.2
)

LOG_FILE=$LOG_FILE python train_model.py ${PARAMS[@]}
