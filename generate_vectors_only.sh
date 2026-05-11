#!/bin/bash

# get the model name as an argument and save it to a variable
MODEL_NAME=$1
# save the part of the model name after the last "/" to a variable
MODEL_STR=${MODEL_NAME##*/}

# this script runs only the vector generation without the dataset generation
cd persona_vectors/

# APATHETIC
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/apathetic_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/apathetic_neg_instruct.csv \
    --trait apathetic \
    --save_dir persona_vectors/${MODEL_STR}/apathetic \

# EVIL
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/${MODEL_STR}/evil \

# FEARFUL
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/fearful_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/fearful_neg_instruct.csv \
    --trait fearful \
    --save_dir persona_vectors/${MODEL_STR}/fearful \

# HALLUCINATING
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/hallucinating_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/hallucinating_neg_instruct.csv \
    --trait hallucinating \
    --save_dir persona_vectors/${MODEL_STR}/hallucinating \

# HONESTY
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/honesty_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/honesty_neg_instruct.csv \
    --trait honesty \
    --save_dir persona_vectors/${MODEL_STR}/honesty \

# HUMOROUS
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/humorous_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/humorous_neg_instruct.csv \
    --trait humorous \
    --save_dir persona_vectors/${MODEL_STR}/humorous \

# IMPOLITE
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/impolite_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/impolite_neg_instruct.csv \
    --trait impolite \
    --save_dir persona_vectors/${MODEL_STR}/impolite \

# MALEVOLENT
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/malevolent_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/malevolent_neg_instruct.csv \
    --trait malevolent \
    --save_dir persona_vectors/${MODEL_STR}/malevolent \

# OPTIMISTIC
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/optimistic_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/optimistic_neg_instruct.csv \
    --trait optimistic \
    --save_dir persona_vectors/${MODEL_STR}/optimistic \

# SYCOPHANTIC
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/sycophantic_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/sycophantic_neg_instruct.csv \
    --trait sycophantic \
    --save_dir persona_vectors/${MODEL_STR}/sycophantic \


# SOCIALLY ADEPT
python generate_vec.py \
    --model_name ${MODEL_NAME} \
    --pos_path eval_persona_extract/${MODEL_STR}/socially_adept_pos_instruct.csv \
    --neg_path eval_persona_extract/${MODEL_STR}/socially_adept_neg_instruct.csv \
    --trait socially_adept \
    --save_dir persona_vectors/${MODEL_STR}/socially_adept \
