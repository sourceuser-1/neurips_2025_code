# src/process_graphs.py
import json
import numpy as np
from tqdm import tqdm  # Optional for progress bar
import sys
sys.path.append('../utils')
from graph_creation import TextGraphGenerator
import argparse

def process_jsonl(input_path, output_path, method, threshold = 0.7):
    # Initialize graph generator
    graph_generator = TextGraphGenerator()
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in tqdm(infile):  # Remove tqdm if not installed
            entry = json.loads(line)
            new_entry = {
                "topic": entry["topic"],
                "adjacency_matrix": [],
                "node_embeddings": []
            }
            
            print(entry["topic"])
            # Process each atomic facts group
            for fact_group in entry["atomic_facts"]:
                if not fact_group or len(fact_group)==1:  # Skip empty groups
                    continue
                    
                # Generate graph data
                # adj_matrix, embeddings = graph_generator.create_graph(texts = fact_group[1:],  method='nli', batch_size=8)
                # adj_matrix, embeddings = graph_generator.create_graph(texts = fact_group[1:],  method='soft', batch_size=8)
                adj_matrix, embeddings = graph_generator.create_graph(texts = fact_group[1:],  method=method, batch_size=32, threshold = threshold)

                
                # Convert numpy arrays to lists for JSON serialization
                new_entry["adjacency_matrix"].append(adj_matrix.tolist())
                new_entry["node_embeddings"].append(embeddings.tolist())
            
            # Write modified entry
            outfile.write(json.dumps(new_entry) + '\n')

def main():
    parser = argparse.ArgumentParser(
        description="Create graph adjacency matrices from atomic facts using the specified method."
    )
    parser.add_argument("--dataset", type=str, default="bios", help="Dataset name (default: bios)")
    parser.add_argument("--model", type=str, default="llama3-8b-instruct", help="Model name (default: llama3-8b-instruct)")
    parser.add_argument("--method", type=str, default="soft", choices=["soft", "nli", "nli_hybrid"],
                        help="Graph creation method (default: soft)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for nli_hybrid method (default: 0.7)")

    args = parser.parse_args()

    # Build file paths based on dataset and model
    input_file = f"../results/{args.dataset}/{args.model}/{args.model}_atomic_facts_samples.jsonl"
    
    # Determine output file name based on method
    if args.method == "soft":
        output_file = f"../results/{args.dataset}/{args.model}/{args.model}_soft_graph_samples.jsonl"
        # output_file = f"../results/{args.dataset}/{args.model}/{args.model}_soft_roberta_graph_samples.jsonl"
    elif args.method == "nli":
        output_file = f"../results/{args.dataset}/{args.model}/{args.model}_nli_graph_samples.jsonl"
    elif args.method == "nli_hybrid":
        output_file = f"../results/{args.dataset}/{args.model}/{args.model}_nli_hybrid_graph_samples.jsonl"
    else:
        raise ValueError("Invalid method specified.")

    process_jsonl(input_file, output_file, method=args.method, threshold=args.threshold)
    print(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # input_file = "../results/bios/llama3-8b-instruct/short_atomic_facts_samples.jsonl"
#     # output_file = "../results/bios/llama3-8b-instruct/short_graph_samples.jsonl"

#     # input_file = "../results/bios/llama3-8b-instruct/llama3-8b-instruct_atomic_facts_samples.jsonl"
#     # input_file = "../results/longfact/llama3-8b-instruct/llama3-8b-instruct_atomic_facts_samples.jsonl"
#     input_file = "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_atomic_facts_samples.jsonl"
#     # output_file = "../results/bios/llama3-8b-instruct/llama3-8b-instruct_graph_samples.jsonl"
#     # output_file = "../results/bios/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl"
#     # output_file = "../results/longfact/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl"
#     output_file = "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl"
#     # output_file = "../results/bios/llama3-8b-instruct/llama3-8b-instruct_nli_graph_samples.jsonl"
    
#     process_jsonl(input_file, output_file)
#     print(f"Processing complete. Output saved to {output_file}")