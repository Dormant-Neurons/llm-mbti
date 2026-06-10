#!/bin/bash

# get the model name as an argument and save it to a variable
MODEL_NAME=$1
# replace all '/' in the model name with '-' and save it to a variable
MODEL_STR=$(echo $MODEL_NAME | tr '/' '-')

# this script runs the complete persona vector generation for all personas
cd persona_vectors/

# machiavellianism
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait machiavellianism \
    --output_path eval_persona_extract/${MODEL_STR}/machiavellianism_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name machiavellianism \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait machiavellianism \
    --output_path eval_persona_extract/${MODEL_STR}/machiavellianism_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name diplomat \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/machiavellianism_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/machiavellianism_neg_instruct.csv \
    --trait machiavellianism \
    --save_dir persona_vectors/${MODEL_STR}/machiavellianism \

# diplomat
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait diplomat \
    --output_path eval_persona_extract/${MODEL_STR}/diplomat_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name diplomat \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait diplomat \
    --output_path eval_persona_extract/${MODEL_STR}/diplomat_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name machiavellianism \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/diplomat_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/diplomat_neg_instruct.csv \
    --trait diplomat \
    --save_dir persona_vectors/${MODEL_STR}/diplomat \

# narcissism
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait narcissism \
    --output_path eval_persona_extract/${MODEL_STR}/narcissism_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name narcissism \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait narcissism \
    --output_path eval_persona_extract/${MODEL_STR}/narcissism_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name contemplative \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/narcissism_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/narcissism_neg_instruct.csv \
    --trait narcissism \
    --save_dir persona_vectors/${MODEL_STR}/narcissism \

# contemplative
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait contemplative \
    --output_path eval_persona_extract/${MODEL_STR}/contemplative_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name contemplative \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait contemplative \
    --output_path eval_persona_extract/${MODEL_STR}/contemplative_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name narcissism \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/contemplative_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/contemplative_neg_instruct.csv \
    --trait contemplative \
    --save_dir persona_vectors/${MODEL_STR}/contemplative \

# psychopathy
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait psychopathy \
    --output_path eval_persona_extract/${MODEL_STR}/psychopathy_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name psychopathy \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait psychopathy \
    --output_path eval_persona_extract/${MODEL_STR}/psychopathy_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name caregiver \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/psychopathy_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/psychopathy_neg_instruct.csv \
    --trait psychopathy \
    --save_dir persona_vectors/${MODEL_STR}/psychopathy \

# caregiver
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait caregiver \
    --output_path eval_persona_extract/${MODEL_STR}/caregiver_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name caregiver \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait caregiver \
    --output_path eval_persona_extract/${MODEL_STR}/caregiver_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name psychopathy \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/caregiver_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/caregiver_neg_instruct.csv \
    --trait caregiver \
    --save_dir persona_vectors/${MODEL_STR}/caregiver \
