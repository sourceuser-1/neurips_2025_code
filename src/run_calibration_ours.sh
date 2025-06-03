#!/bin/bash

# Usage: ./run_calibration.sh <model> [<dataset1> <dataset2> ...]
# If no datasets are provided, the script defaults to bios, longfact, and wildhallu.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model> [<dataset1> <dataset2> ...]"
  exit 1
fi

MODEL="$1"
shift 1  # Remove the model argument

# Set default datasets if none are provided
if [ "$#" -eq 0 ]; then
  DATASETS=("bios" "longfact" "wildhallu" "eli5" "sciqa")
else
  DATASETS=("$@")
fi

# Loop through each dataset and run the calibration script.
for DATASET in "${DATASETS[@]}"; do
  python calibration_ours.py --dataset "$DATASET" --model "$MODEL"
done

# ./run_calibration_ours.sh qwen2-7b-instruct bios longfact wildhallu