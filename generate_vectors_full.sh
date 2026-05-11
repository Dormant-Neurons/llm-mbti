#!/bin/bash

# get the model name as an argument and save it to a variable
MODEL_NAME=$1
# save the part of the model name after the last "/" to a variable
MODEL_STR=${MODEL_NAME##*/}

# this script runs the complete persona vector generation for all personas
cd persona_vectors/

# APATHETIC
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait apathetic \
    --output_path eval_persona_extract/${MODEL_STR}/apathetic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name apathetic \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait apathetic \
    --output_path eval_persona_extract/${MODEL_STR}/apathetic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/apathetic_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/apathetic_neg_instruct.csv \
    --trait apathetic \
    --save_dir persona_vectors/${MODEL_STR}/apathetic \

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
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/${MODEL_STR}/evil \

# FEARFUL
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait fearful \
    --output_path eval_persona_extract/${MODEL_STR}/fearful_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name fearful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait fearful \
    --output_path eval_persona_extract/${MODEL_STR}/fearful_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/fearful_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/fearful_neg_instruct.csv \
    --trait fearful \
    --save_dir persona_vectors/${MODEL_STR}/fearful \

# HALLUCINATING
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait hallucinating \
    --output_path eval_persona_extract/${MODEL_STR}/hallucinating_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name hallucinating \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait hallucinating \
    --output_path eval_persona_extract/${MODEL_STR}/hallucinating_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/hallucinating_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/hallucinating_neg_instruct.csv \
    --trait hallucinating \
    --save_dir persona_vectors/${MODEL_STR}/hallucinating \

# HONESTY
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait honesty \
    --output_path eval_persona_extract/${MODEL_STR}/honesty_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name honesty \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait honesty \
    --output_path eval_persona_extract/${MODEL_STR}/honesty_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/honesty_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/honesty_neg_instruct.csv \
    --trait honesty \
    --save_dir persona_vectors/${MODEL_STR}/honesty \

# HUMOROUS
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait humorous \
    --output_path eval_persona_extract/${MODEL_STR}/humorous_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name humorous \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait humorous \
    --output_path eval_persona_extract/${MODEL_STR}/humorous_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/humorous_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/humorous_neg_instruct.csv \
    --trait humorous \
    --save_dir persona_vectors/${MODEL_STR}/humorous \

# IMPOLITE
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait impolite \
    --output_path eval_persona_extract/${MODEL_STR}/impolite_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name impolite \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait impolite \
    --output_path eval_persona_extract/${MODEL_STR}/impolite_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/impolite_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/impolite_neg_instruct.csv \
    --trait impolite \
    --save_dir persona_vectors/${MODEL_STR}/impolite \

# MALEVOLENT
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait malevolent \
    --output_path eval_persona_extract/${MODEL_STR}/malevolent_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name malevolent \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait malevolent \
    --output_path eval_persona_extract/${MODEL_STR}/malevolent_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/malevolent_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/malevolent_neg_instruct.csv \
    --trait malevolent \
    --save_dir persona_vectors/${MODEL_STR}/malevolent \

# OPTIMISTIC
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait optimistic \
    --output_path eval_persona_extract/${MODEL_STR}/optimistic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name optimistic \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait optimistic \
    --output_path eval_persona_extract/${MODEL_STR}/optimistic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/optimistic_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/optimistic_neg_instruct.csv \
    --trait optimistic \
    --save_dir persona_vectors/${MODEL_STR}/optimistic \

# SYCOPHANTIC
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait sycophantic \
    --output_path eval_persona_extract/${MODEL_STR}/sycophantic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name sycophantic \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait sycophantic \
    --output_path eval_persona_extract/${MODEL_STR}/sycophantic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/sycophantic_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/sycophantic_neg_instruct.csv \
    --trait sycophantic \
    --save_dir persona_vectors/${MODEL_STR}/sycophantic \


# SOCIALLY ADEPT
python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait socially_adept \
    --output_path eval_persona_extract/${MODEL_STR}/socially_adept_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name socially_adept \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python -m eval.eval_persona \
    --model ${MODEL_NAME} \
    --trait socially_adept \
    --output_path eval_persona_extract/${MODEL_STR}/socially_adept_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/socially_adept_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/socially_adept_neg_instruct.csv \
    --trait socially_adept \
    --save_dir persona_vectors/${MODEL_STR}/socially_adept \
