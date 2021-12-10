# ++++ CHANGE ++++
NAME="effnet_wd"
VERSION_NUMBER="0"
CKP_NAME="model-epoch_094-val_loss=0.052.ckpt"
# ++++++++++++++++

OUTPUT_DIR="output/${NAME}/version_${VERSION_NUMBER}/"
LOG_FILE="output/${NAME}/version_${VERSION_NUMBER}_test.log"
CKP_PATH="output/${NAME}/version_${VERSION_NUMBER}/ckp/${CKP_NAME}"

PARAMS=(
    --mask_dir "../data/masks/tensormask/bone_age" "../data/masks/unet/bone_age"
    --batch_size 32
    --cache_data
    --input_height 512
    --input_width 512
    --num_workers 32
    --ckp_path $CKP_PATH
    --output_dir $OUTPUT_DIR
#     --no_test_tta_rot
#     --no_test_tta_flip
)

LOG_FILE=$LOG_FILE python test_model.py ${PARAMS[@]}
