#! /bin/bash
#SBATCH --job-name="bone-age_sh"
#SBATCH --partition="ampere"
#SBATCH --mem="30G"
#SBATCH --cpus-per-task="32"
#SBATCH --gres gpu:1,localtmp:25G
#SBATCH --time="22:00:00"
#SBATCH --output=/ceph01/projects/bone2gene/bone-age2/output/slurm/%j.out
#SBATCH --signal=SIGUSR1@90

NAME="masked_effnet_shallow_fancy_aug"

module load anaconda/4.6.11-py37
conda activate ptl16
cd /ceph01/projects/bone2gene/bone-age2/
DATA_DIR="../data"

python ../parallel_copy.py -s ../data/masks/tensormask/bone_age -o $SCRATCH_DIR/data/masks/tensormask/bone_age -d 1
python ../parallel_copy.py -s ../data/masks/unet/bone_age -o $SCRATCH_DIR/data/masks/unet/bone_age -d 1
python ../parallel_copy.py -s ../data/annotated/bone_age -o $SCRATCH_DIR/data/annotated/bone_age -d 1
DATA_DIR="$SCRATCH_DIR/data"

out_root=output/$NAME
mkdir output
mkdir "$out_root"
LOG_FILE="$out_root/version_0".log

LOG_FILE=$LOG_FILE python train_model.py \
    --name $NAME \
    --config=configs/xcat.yml \
    --data.img_dir="$DATA_DIR/annotated" \
    --data.mask_dirs=["$DATA_DIR/masks/tensormask","$DATA_DIR/masks/unet"] \
    --config=configs/default_model.yml \
    --model.dense_layers=[256]