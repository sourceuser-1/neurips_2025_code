#!/bin/bash

# Usage: ./run_atomic_facts.sh <model_name> [<dataset1> <dataset2> ...]
# If no datasets are provided, defaults to bios, longfact, and wildhallu.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model_name> [<dataset1> <dataset2> ...]"
  exit 1
fi

MODEL_NAME="$1"
shift 1  # Remove model_name from the arguments

# Set default datasets if none are provided
if [ "$#" -eq 0 ]; then
  DATASETS=("bios" "longfact" "wildhallu", "eli5", "sciqa")
else
  DATASETS=("$@")
fi

for DATASET in "${DATASETS[@]}"; do
  python vllm_generate_atomic_facts_samples.py --model_name "$MODEL_NAME" --dataset "$DATASET"
done

# ./run_gen_atomic_facts.sh mistral-7b-instruct bios longfact wildhallu

# ./run_gen_atomic_facts.sh falcon-7b-instruct longfact wildhallu