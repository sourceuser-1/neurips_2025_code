

# src/process_entailments.py
import json
import numpy as np
from tqdm import tqdm  # For progress bar
import sys
import argparse  # For argument parsing
sys.path.append('../utils')
from entailment import get_entailment_probs_batch  # Use the new batch function
import os


def process_jsonl(input_path, output_path, dataset, model, batch_size=128):
    print(f"Processing dataset: {dataset}, model: {model}")

    # 1) Figure out which topics we've already done
    processed_topics = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f_out:
            for line in f_out:
                try:
                    entry = json.loads(line)
                    processed_topics.add(entry.get("topic"))
                except json.JSONDecodeError:
                    continue

    # 2) Open input for reading, output for appending
    with open(input_path, 'r') as infile, open(output_path, 'a') as outfile:
        # (Optional) total for progress bar
        total_lines = sum(1 for _ in open(input_path, 'r'))
        pbar = tqdm(infile, total=total_lines, desc="Lines", unit="line")

        for line in pbar:
            entry = json.loads(line)
            topic = entry["topic"]

            # 3) Skip if already processed
            if topic in processed_topics:
                pbar.set_postfix(skipped=topic)
                continue

            atomic_facts = entry.get("atomic_facts", [])
            if len(atomic_facts) < 2:
                new_entry = {"topic": topic, "entailment_matrices": []}
                outfile.write(json.dumps(new_entry) + "\n")
                processed_topics.add(topic)
                continue

            reference_claims = atomic_facts[0][1:]
            non_reference_claims_lists = [grp[1:] for grp in atomic_facts[1:]]
            entailment_matrices = []

            for non_ref_claims in non_reference_claims_lists:
                N, M = len(reference_claims), len(non_ref_claims)
                premises, hypotheses = [], []
                for r in reference_claims:
                    for n in non_ref_claims:
                        premises.append(r)
                        hypotheses.append(n)

                probs_list = get_entailment_probs_batch(
                    premises, hypotheses, batch_size=batch_size
                )
                matrix = np.array(probs_list).reshape((N, M, 3))
                entailment_matrices.append(matrix.tolist())

            new_entry = {
                "topic": topic,
                "entailment_matrices": entailment_matrices
            }
            outfile.write(json.dumps(new_entry) + "\n")
            processed_topics.add(topic)

    print(f"Done — processed {len(processed_topics)} topics total.")


def main():
    parser = argparse.ArgumentParser(description="Process atomic facts and generate entailment matrices.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., wildhallu, bios).")
    parser.add_argument('--model', type=str, required=True, help="Model name to use for processing (e.g., llama3-8b-instruct).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for model inference.")
    args = parser.parse_args()

    input_file = f"../results/{args.dataset}/{args.model}/{args.model}_atomic_facts_samples.jsonl"
    output_file = f"../results/{args.dataset}/{args.model}/{args.model}_entailment_samples.jsonl"
    
    process_jsonl(input_file, output_file, args.dataset, args.model, batch_size=args.batch_size)
    print(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
