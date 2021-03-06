#!/bin/bash

#conda activate skele

PROJ_DIR=/home/cl/yuki-yama/work/phd/v1
SCRIPT_DIR=${PROJ_DIR}/scripts
#MODEL_DIR=${PROJ_DIR}/classifier/torch_models
#model_path=${MODEL_DIR}/BERT-AMR/10_5e-05_16/1.pth

model_path=${SCRIPT_DIR}/demo/model.pth
input=${SCRIPT_DIR}/demo/in.sents

skele=$(
    python pipeline.py \
        --model_path=${model_path} \
        --file_path=${input}
)

#conda deactivate