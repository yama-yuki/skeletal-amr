#!/bin/bash

#conda activate skele

PROJ_DIR=/home/cl/yuki-yama/work/phd/v1
SCRIPT_DIR=${PROJ_DIR}/scripts
#MODEL_DIR=${PROJ_DIR}/torch_models
MODEL_DIR=BERT-AMR_10_5e-05_16
input=${SCRIPT_DIR}/demo.sents

skele=$(
    python pipeline.py \
        --model_name=${MODEL_DIR} \
        --file_path=${input}
)

echo ${skele}

#conda deactivate