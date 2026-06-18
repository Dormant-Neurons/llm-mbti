#!/bin/bash

# get the model name as an argument and save it to a variable
MODEL_NAME=$1
# replace all '/' in the model name with '-' and save it to a variable
MODEL_STR=$(echo $MODEL_NAME | tr '/' '-')

# this script runs the complete persona vector generation for all personas
cd persona_vectors/

# EVIL
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait evil \
    --output_path eval_persona_extract/${MODEL_STR}/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait evil \
    --output_path eval_persona_extract/${MODEL_STR}/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name ethical \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/${MODEL_STR}/evil \
