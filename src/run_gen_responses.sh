#!/bin/bash

# Usage: ./run_generate_responses.sh <model_name> [<dataset1> <dataset2> ...]
# If no datasets are provided, the script defaults to bios, longfact, and wildhallu.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model_name> [<dataset1> <dataset2> ...]"
  exit 1
fi

MODEL_NAME="$1"
shift 1  # Remove the model name argument

# Set default datasets if none are provided
if [ "$#" -eq 0 ]; then
  DATASETS=("bios" "longfact" "wildhallu", "eli5", "sciqa")
else
  DATASETS=("$@")
fi

# Loop through each dataset and run the Python command
for DATASET in "${DATASETS[@]}"; do
  python generate_responses_vllm.py \
    --cuda_devices "1,2" \
    --gpu_memory_utilization 0.8 \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --generate_samples
done

# ./run_gen_responses.sh mistral-7b-instruct bios longfact wildhallu
