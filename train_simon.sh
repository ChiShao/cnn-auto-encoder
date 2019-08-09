TRAINER_PACKAGE_PATH="./trainer"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="SPM_AS_$now"

gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name="trainer.task" \
        --region="us-east1" \
        --package-path $TRAINER_PACKAGE_PATH \
        --job-dir="gs://spm-training" \
        --runtime-version="1.13" \
        --python-version="3.5" \
        --scale-tier=CUSTOM\
        --master-machine-type standard_p100 \
        -- \
        --model 'gs://spm-training/logs/SPM_AS_20190731_170335/ep047-loss31.225-val_loss28.801.h5' \
        --anchors 'gs://spm-training/model_data/yolo_anchors.txt' \
        --classes 'gs://training-data-images/data/classes.txt' \
        --logdir "gs://spm-training/logs/$JOB_NAME" \
        --train_data 'gs://training-data-images/data/train.txt' \
        --data_base_path 'gs://training-data-images/data/' \
        --gross_epochs 0 \
        --fine_epochs 50 \
        --gross_batch_size 32 \
        --fine_batch_size 16 \
        --output 'gs://spm-training/out