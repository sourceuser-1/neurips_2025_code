# Run the following steps sequentially to get the calibration results reported in the paper:

# Generate Responses
```./run_gen_responses.sh [model] [dataset]```

# Generate Atomic Facts
```./run_gen_atomic_facts.sh [model] [dataset]```

# Generate Semantic Graphs from Decomposed Atomic Facts
```./run_gen_graphs.sh [model] [dataset] ```

# Factchecking using FACTSCORE and SAFE for calibration
```python vllm_fact_check.py --dataset bios --model_name [model]```
``` ./run_SAFE.sh [model] [dataset] ```

# Run gating signal by calculating entailment 
``` ./run_entailment.sh [model] [dataset] ```

# Get the graph based uncertainty score
``` ./run_calc_dissim.sh [model] "batch_alignment_cost_entailment" [dataset] ```


# Calibration 
``` ./run_calibration_ours.sh [model] [dataset] ```

# Possible models and datasets
[model] can be :
qwen2-57b-instruct 
qwen2-7b-instruct
mistral-7b-instruct
falcon-7b-instruct
llama3-8b-instruct

[dataset] can be:
bios
longfact
wildhallu
eli5
sciqa
