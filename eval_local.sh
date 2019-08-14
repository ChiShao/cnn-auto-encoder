TRAINER_PACKAGE_PATH="./trainer"
now=$(date +"%Y%m%d_%H%M%S")
USE_CASE="cable"
echo $USE_CASE
JOB_NAME="AD_${USE_CASE}_$now"
echo $JOB_NAME
# BUCKET = "gs://anomaly-detection-playground"
BUCKET=""

python3 -m trainer.task \
        --use-case $USE_CASE \
        --logdir  "${BUCKET}logs/$JOB_NAME" \
        --ckptdir "${BUCKET}ckpts/$JOB_NAME" \
        --imgdir  "${BUCKET}imgs/$JOB_NAME" \
        --evaldir "${BUCKET}eval/$JOB_NAME" \
        --datadir "${BUCKET}data/mvtec_anomaly_detection" \
        --epochs 1 \
        --batch_size 12 \
        --filters 16 16 32 64 128 64 32 32 \
        --ldim 150 