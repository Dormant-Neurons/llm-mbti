#!/bin/bash

# this script runs only the vector generation without the dataset generation
cd persona_vectors/

# APATHETIC
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_neg_instruct.csv \
    --trait apathetic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/apathetic

# EVIL
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/evil

# FEARFUL
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_neg_instruct.csv \
    --trait fearful \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/fearful

# HALLUCINATING
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_neg_instruct.csv \
    --trait hallucinating \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/hallucinating

# HONESTY
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_neg_instruct.csv \
    --trait honesty \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/honesty

# HUMOROUS
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_neg_instruct.csv \
    --trait humorous \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/humorous

# IMPOLITE
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_neg_instruct.csv \
    --trait impolite \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/impolite

# MALEVOLENT
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_neg_instruct.csv \
    --trait malevolent \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/malevolent

# OPTIMISTIC
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_neg_instruct.csv \
    --trait optimistic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/optimistic

# SYCOPHANTIC
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_neg_instruct.csv \
    --trait sycophantic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/sycophantic


# SOCIALLY ADEPT
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_neg_instruct.csv \
    --trait socially_adept \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/socially_adept
