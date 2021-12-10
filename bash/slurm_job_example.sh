#! /bin/bash
#SBATCH --job-name="bone_age_slurm"
#SBATCH --partition="batch"
#SBATCH --mem="30G"
#SBATCH --cpus-per-task="8"
#SBATCH --gres gpu:1
#SBATCH --time="06:00:00"

module load anaconda/4.6.11-py37
conda activate ptl
cd bone2gene/bone_age

NAME="effnet_wd-medium_SLURM"
LOG_FILE=output/$NAME/"version_0.log"
mkdir output/$NAME



PARAMS=(
    --name $NAME
    --gpus 1
    --precision 16
    --cache_data
    --max_time 00:05:30:00
    --input_height 512
    --input_width 512
    --num_workers 32
    --max_epoch 10
    --batch_size 32
    --model dbam_efficientnet-b0
    --weight_decay 1e-6
#     --mask_dir "../TensorMask/output/Colab_train/inference/bone_age" "../TensorMask/output/Colab_train/inference_alexander/bone_age_train"
#     --learning_rate 5e-3
#     --min_lr 5e-5
#     --rlrp_factor 0.2
#     --rotation_range 30
#     --shear_percent 5
#     --img_norm_method interval
)

LOG_FILE=$LOG_FILE python train_model.py ${PARAMS[@]}
