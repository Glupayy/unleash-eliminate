#!/bin/sh

export BNB_CUDA_VERSION=114 # set it to your cuda version
export CUDA_HOME=
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=$1
CKPT_PATH="path to Osprey-7b"
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

export GT_FILE_NOPE="path to PHD/data.jsonl"

export RESULT_PATH=results/$2

export IMAGE_DIR="path to PHD images"
export CHOSED_LAYER=$3

export LAYER_PROB=$4

if test -d $RESULT_PATH; then
    echo "existed path!"
    exit 1
fi

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" model/osprey/eval/phd_eval_w_prob.py --model "$CKPT_PATH" --json "$GT_FILE_NOPE" --img "$IMAGE_DIR" --output "$RESULT_PATH" --chosed_layer "$CHOSED_LAYER" --layer_prob "$LAYER_PROB"
python model/osprey/eval/phd_eval_metric.py --annotation_file "$GT_FILE_NOPE" --results_dir "$RESULT_PATH"
