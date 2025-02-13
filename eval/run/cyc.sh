#!/bin/sh

export PYTHONPATH="./:$PYTHONPATH"
export BNB_CUDA_VERSION=114 # set it to your cuda version
export CUDA_HOME=
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)

export CUDA_VISIBLE_DEVICES=0,1

export LOG_BASE_DIR="./output"
export EXP_NAME=$1

export LAYER_S=$2
export LAYER_E=$3
export SAMPLE_N_LAYERS=$4
export IDX_NUM=$5
# DeepSpeed command (customize the arguments as per your needs)

deepspeed --master_port $MASTER_PORT main.py \
  --lr 3e-6 \
  --lora_r 8 \
  --precision "fp16" \
  --pretrained \
  --val_dataset_reg RefCOCOgOspreyVal \
  --log_base_dir $LOG_BASE_DIR\
  --exp_name $EXP_NAME\
  --layer_s $LAYER_S \
  --layer_e $LAYER_E \
  --sample_n_layers $SAMPLE_N_LAYERS \
  --idx_num $IDX_NUM

python merge_captions.py --results_dir "./output/"$EXP_NAME
