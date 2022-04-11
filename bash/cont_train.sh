NAME="debug"
LOG_FILE=output/$NAME/"version_1_cont.log"
mkdir output/$NAME

PARAMS=(
    --name $NAME
    --gpus 1
    --precision 16
#     --cache_data
    --max_time 00:10:30:00
    --input_height 512
    --input_width 512
    --num_workers 32
    --max_epoch 300
    --batch_size 32
    --model dbam_inceptionv3
    --weight_decay 5e-6
    --mask_dir "../data/masks/tensormask/bone_age" "../data/masks/unet/bone_age"
    --learning_rate 1e-3
    --min_lr 1e-4
    --rlrp_factor 0.2
    --resume_from_checkpoint "output/${NAME}/version_0/ckp/last.ckpt"
    --rotation_range 30
    --shear_percent 10
    --contrast_gamma 30
    --clae_p 1
    --sharpen_p 0.2
    --dense_layers 512 256
)

LOG_FILE=$LOG_FILE python train_model.py ${PARAMS[@]}
