#!/bin/bash

# this script runs the eval_persona script with different layer sizes

cd persona_vectors/
mkdir layer_effectiveness

# iterate through all traits and layers
for trait in "apathetic" "evil" "fearful" "hallucinating" "honesty" "humorous" "impolite" "malevolent" "optimistic" "sycophantic" "socially_adept"; do
    for layer in 10 20 30 40 50 60; do
        # print the current trait and layer being evaluated
        echo "Evaluating trait: $trait, layer: $layer"

        # run the eval_persona script and write the stdout to a file
        python -m eval.eval_persona \
        --model mlabonne/gemma-3-27b-it-abliterated \
        --trait $trait \
        --output_path eval_persona_eval/steering_results_${trait}_${layer}.csv \
        --judge_model gpt-4.1-mini-2025-04-14  \
        --version eval \
        --steering_type all \
        --coef 2.0 \
        --vector_path persona_vectors/mlabonne-gemma-3-27b-it-abliterated/${trait}/${trait}_response_avg_diff.pt \
        --layer $layer \
        > layer_effectiveness/mlabonne-gemma-3-27b-it-abliterated_${trait}_${layer}_output.txt
    done
done
