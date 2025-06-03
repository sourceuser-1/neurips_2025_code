#!/bin/bash

# Usage: ./run_SAFE.sh <model_name> [<dataset1> <dataset2> ...]
# If no datasets are provided, the script defaults to longfact and wildhallu.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model_name> [<dataset1> <dataset2> ...]"
  exit 1
fi

MODEL_NAME="$1"
shift 1  # Remove the model name argument

# Set default datasets if none are provided.
if [ "$#" -eq 0 ]; then
  DATASETS=("longfact" "wildhallu" "eli5" "sciqa")
else
  DATASETS=("$@")
fi

# Loop through each dataset and run the SAFE.py command.
for DATASET in "${DATASETS[@]}"; do
  python SAFE.py --dataset "$DATASET" --model_name "$MODEL_NAME"
done

# ./run_SAFE.sh qwen2-7b-instruct longfact wildhallu
