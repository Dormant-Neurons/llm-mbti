#!/bin/bash

# get the model name as an argument and save it to a variable
MODEL_NAME=$1
# replace all '/' in the model name with '-' and save it to a variable
MODEL_STR=$(echo $MODEL_NAME | tr '/' '-')

# this script runs only the vector generation without the dataset generation
cd persona_vectors/

# EVIL
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/${MODEL_STR}/evil \
