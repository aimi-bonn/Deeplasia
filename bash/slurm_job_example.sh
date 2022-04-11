#! /bin/bash
#SBATCH --job-name="bone-age_multitask"
#SBATCH --partition="ampere"
#SBATCH --mem="30G"
#SBATCH --cpus-per-task="16"
#SBATCH --gres gpu:1,localtmp:20G
#SBATCH --time="12:00:00"
#SBATCH --output=/ceph01/projects/bone2gene/bone-age/output/slurm/%j.out

NAME="multitask_effnet_fancy_aug"

module load anaconda/4.6.11-py37
conda activate ptl
cd /ceph01/projects/bone2gene/bone-age/
DATA_DIR="../data"

python ../parallel_copy.py -s ../data/masks/tensormask/bone_age -o $SCRATCH_DIR/data/masks/tensormask/bone_age -d 1
python ../parallel_copy.py -s ../data/masks/unet/bone_age -o $SCRATCH_DIR/data/masks/unet/bone_age -d 1
python ../parallel_copy.py -s ../data/annotated/rsna_bone_age -o $SCRATCH_DIR/data/annotated/rsna_bone_age -d 2
python ../parallel_copy.py -s ../data/annotated/rsna_bone_age -o $SCRATCH_DIR/data/annotated/rsna_bone_age -d 1
DATA_DIR="$SCRATCH_DIR/data"

out_root=output/$NAME
mkdir output
mkdir "$out_root"
LOG_FILE="$out_root/version_0".log

LOG_FILE=$LOG_FILE python train_model.py \
  --name $NAME \
  --gpus 1 \
  --precision 16\
  --data_dir "$DATA_DIR/annotated/rsna_bone_age"\
  --mask_dirs "$DATA_DIR/masks/tensormask/bone_age" "$DATA_DIR/masks/unet/bone_age" \
  --max_time 11:30:00:00 \
  --input_height 512\
  --input_width 512\
  --num_workers 16\
  --model dbam_efficientnet-b0 \
  --max_epoch 300\
  --batch_size 32\
  --learning_rate 1e-3\
  --min_lr 1e-4\
  --rlrp_factor 0.2\
  --rlrp_patience 5\
  --weight_decay 5e-6\
  --rotation_range 30\
  --sharpen_p 0.2\
  --contrast_gamma 30\
  --clae_p 0.5 \
  --n_gender_dcs 0 \
  --sex_sigma 1\
  --age_sigma 1
