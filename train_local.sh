TRAINER_PACKAGE_PATH="./trainer"
now=$(date +"%Y%m%d_%H%M%S")
USE_CASE="cable"
echo $USE_CASE
JOB_NAME="AD_${USE_CASE}_$now"
echo $JOB_NAME

python3 -m trainer.task \
        --use-case $USE_CASE \
        --logdir  "gs://anomaly-detection-playground/logs/$JOB_NAME" \
        --ckptdir "gs://anomaly-detection-playground/ckpts/$JOB_NAME" \
        --imgdir  "gs://anomaly-detection-playground/imgs/$JOB_NAME" \
        --evaldir "gs://anomaly-detection-playground/eval/$JOB_NAME" \
        --datadir "gs://anomaly-detection-playground-data/mvtec_anomaly_detection" \
        --epochs 1 \
        --batch_size 12 \
        --filters 16 16 32 64 128 64 32 32 \
        --ldim 150 