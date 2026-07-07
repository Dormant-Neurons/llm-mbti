#!/bin/bash
cd persona_vectors/

# List of all Big Five personality traits
traits=("evil" "sycophantic" "hallucinating" "humorous" "impolite" "optimistic" "apathetic")
# traits=("calm")  # Default to "careless" for testing
# Allow override with command line argument
if [ $# -gt 0 ]; then
    traits=("$@")
fi

echo "Processing traits: ${traits[@]}"
model_path="${MODEL_PATH:-Qwen/Qwen2.5-7b-Instruct}"
model_name=$(echo $model_path | tr '/' '-')

# Iterate through all traits
for trait in "${traits[@]}"; do
    echo "=========================================="
    echo "Processing trait: $trait"
    echo "Model path: $model_path"
    echo "Model name: $model_name"
    echo "=========================================="
    
    # Step 1: Generate positive instruction data
    echo "Step 1/3: Generating positive instruction data for $trait..."
    python -m eval.eval_persona \
        --model $model_path \
        --trait ${trait} \
        --output_path eval_persona_extract/${model_name}/${trait}_pos_instruct.csv \
        --persona_instruction_type pos \
        --assistant_name ${trait} \
        --judge_model gpt-4.1-mini \
        --version extract \
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate positive instruction data for $trait"
        continue
    fi
    
    # Step 2: Generate negative instruction data
    echo "Step 2/3: Generating negative instruction data for $trait..."
    python -m eval.eval_persona \
        --model $model_path \
        --trait ${trait} \
        --output_path eval_persona_extract/${model_name}/${trait}_neg_instruct.csv \
        --persona_instruction_type neg \
        --assistant_name ${trait} \
        --judge_model gpt-4.1-mini \
        --version extract \
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate negative instruction data for $trait"
        continue
    fi
    
    # Step 3: Generate persona vector
    echo "Step 3/3: Generating persona vector for $trait..."
    python generate_vec.py \
        --model_name $model_path \
        --pos_path eval_persona_extract/${model_name}/${trait}_pos_instruct.csv \
        --neg_path eval_persona_extract/${model_name}/${trait}_neg_instruct.csv \
        --trait ${trait} \
        --save_dir persona_vectors/${model_name}/${trait} \
        --threshold 50 \
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate persona vector for $trait"
        continue
    fi
    
    echo "? Successfully completed processing for trait: $trait"
    echo ""
done

echo "=========================================="
echo "All traits processing completed!"
echo "=========================================="
echo "Processed traits: ${traits[@]}"
echo ""
echo "Generated files:"
for trait in "${traits[@]}"; do
    echo "  - eval_persona_extract/${model_name}/${trait}_pos_instruct.csv"
    echo "  - eval_persona_extract/${model_name}/${trait}_neg_instruct.csv"
    echo "  - persona_vectors/${model_name}/${trait}.pt"
done