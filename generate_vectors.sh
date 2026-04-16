#!/bin/bash

# this script runs the complete persona vector generation for all personas
cd persona_vectors/

# APATHETIC
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait apathetic \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name apathetic \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait apathetic \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/apathetic_neg_instruct.csv \
    --trait apathetic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/apathetic

# EVIL
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait evil \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait evil \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/evil

# FEARFUL
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait fearful \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name fearful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait fearful \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/fearful_neg_instruct.csv \
    --trait fearful \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/fearful

# HALLUCINATING
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait hallucinating \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name hallucinating \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait hallucinating \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/hallucinating_neg_instruct.csv \
    --trait hallucinating \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/hallucinating

# HONESTY
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait honesty \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name honesty \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait honesty \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/honesty_neg_instruct.csv \
    --trait honesty \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/honesty

# HUMOROUS
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait humorous \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name humorous \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait humorous \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/humorous_neg_instruct.csv \
    --trait humorous \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/humorous

# IMPOLITE
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait impolite \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name impolite \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait impolite \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/impolite_neg_instruct.csv \
    --trait impolite \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/impolite

# MALEVOLENT
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait malevolent \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name malevolent \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait malevolent \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/malevolent_neg_instruct.csv \
    --trait malevolent \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/malevolent

# OPTIMISTIC
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait optimistic \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name optimistic \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait optimistic \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/optimistic_neg_instruct.csv \
    --trait optimistic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/optimistic

# SYCOPHANTIC
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait sycophantic \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name sycophantic \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait sycophantic \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/sycophantic_neg_instruct.csv \
    --trait sycophantic \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/sycophantic


# SOCIALLY ADEPT
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait socially_adept \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name socially_adept \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait socially_adept \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract \
    --n_per_question 10 \
    --max_concurrent_judges 10 \
    --max_tokens 2048 \

python generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/socially_adept_neg_instruct.csv \
    --trait socially_adept \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/socially_adept
