JOBNAME=train_drones_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gcloud ml-engine jobs submit training JOBNAME \
	--runtime-version 1.12	\
	--job-dir=gs://e4u-drone-ml/tmp \
	--packages packages/object_detection-0.1.tar.gz,packages/slim-0.1.tar.gz,packages/pycocotools-2.0.tar.gz \
	--module-name object_detection.model_main \
	--region europe-west1 \
	--config cloud.yml \
	-- \
	--model_dir=gs://e4u-drone-ml/models/current-training \
	--pipeline_config_path=gs://e4u-drone-ml/models/pipeline.config \

