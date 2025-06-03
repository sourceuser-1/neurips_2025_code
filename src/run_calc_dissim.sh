#!/bin/bash

# python calculate_dissimilarities.py --dataset bios --model qwen2-7b-instruct --method "batch_alignment_cost_entailment"
# python calculate_dissimilarities.py --dataset longfact --model qwen2-7b-instruct --method "batch_alignment_cost_entailment"
# python calculate_dissimilarities.py --dataset wildhallu --model qwen2-7b-instruct --method "batch_alignment_cost_entailment"

#!/bin/bash

# Check for at least three arguments: model, method, and at least one dataset
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <model> <method> <dataset1> [<dataset2> ...]"
  exit 1
fi

MODEL="$1"
METHOD="$2"
shift 2  # Remove the first two arguments

for DATASET in "$@"; do
    python calculate_dissimilarities.py --dataset "$DATASET" --model "$MODEL" --method "$METHOD"
done

# ./run_calc_dissim.sh qwen2-7b-instruct "batch_alignment_cost_entailment" bios longfact wildhallu
