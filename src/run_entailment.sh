#!/bin/bash

# python generate_pairwise_entailment.py --dataset bios --model qwen2-7b-instruct
# python generate_pairwise_entailment.py --dataset longfact --model qwen2-7b-instruct
# python generate_pairwise_entailment.py --dataset wildhallu --model qwen2-7b-instruct

#!/bin/bash

# Usage: ./run_pairwise_entailment.sh <model> [<dataset1> <dataset2> ...]
# If no datasets are provided, the script defaults to bios, longfact, and wildhallu.

# Check that at least the model name is provided.
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model> [<dataset1> <dataset2> ...]"
  exit 1
fi

MODEL="$1"
shift 1  # Remove the model argument

# Set default datasets if none are provided.
if [ "$#" -eq 0 ]; then
  DATASETS=("bios" "longfact" "wildhallu" "eli5" "sciqa")
else
  DATASETS=("$@")
fi

# Loop through each dataset and run the Python script.
for DATASET in "${DATASETS[@]}"; do
  python generate_pairwise_entailment.py --dataset "$DATASET" --model "$MODEL"
done


# ./run_entailment.sh qwen2-7b-instruct bios longfact wildhallu
