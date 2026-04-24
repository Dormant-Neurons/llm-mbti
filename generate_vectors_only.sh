#!/bin/bash

# this script runs only the vector generation without the dataset generation
cd persona_vectors/

# APATHETIC
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_neg_instruct.csv \
    --trait apathetic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/apathetic \
    --threshold 0

# EVIL
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/evil \
    --threshold 0

# FEARFUL
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_neg_instruct.csv \
    --trait fearful \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/fearful \
    --threshold 0

# HALLUCINATING
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_neg_instruct.csv \
    --trait hallucinating \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/hallucinating \
    --threshold 0

# HONESTY
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_neg_instruct.csv \
    --trait honesty \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/honesty \
    --threshold 0

# HUMOROUS
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_neg_instruct.csv \
    --trait humorous \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/humorous \
    --threshold 0

# IMPOLITE
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_neg_instruct.csv \
    --trait impolite \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/impolite \
    --threshold 0

# MALEVOLENT
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_neg_instruct.csv \
    --trait malevolent \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/malevolent \
    --threshold 0

# OPTIMISTIC
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_neg_instruct.csv \
    --trait optimistic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/optimistic \
    --threshold 0

# SYCOPHANTIC
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_neg_instruct.csv \
    --trait sycophantic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/sycophantic \
    --threshold 0


# SOCIALLY ADEPT
python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_neg_instruct.csv \
    --trait socially_adept \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/socially_adept \
    --threshold 0
