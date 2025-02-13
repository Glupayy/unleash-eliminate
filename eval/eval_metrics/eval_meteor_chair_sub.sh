#!/bin/sh

export BNB_CUDA_VERSION=114 # set it to your cuda version
export CUDA_HOME=
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=22333
CKPT_PATH="path to Osprey-7b"
RESULT_PATH=output/$1
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

ANNOTATION_FILE=files/finetune_refcocog_val_with_mask.json
IMAGE_DIR="path to RefCOCO image coco_2014/train2014"
GT_FILE=files/captions_refcocog_gt.json

CHOSED_LAYER=$2

python eval/eval_metrics/refcocog_eval_metric.py --annotation_file "$GT_FILE" --results_dir "$RESULT_PATH"
python eval/eval_metrics/refcocog_eval_chair_metric.py --cap_file="$RESULT_PATH""/caption_persam_merged.json" --coco_path="$GT_FILE" --save_path="$RESULT_PATH""/chair_score.txt"