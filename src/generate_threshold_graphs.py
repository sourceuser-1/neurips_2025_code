import json
import argparse
import os
import sys
sys.path.append('../utils')

def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def threshold_adjacency_matrix(matrix, threshold):
    """
    Convert a soft adjacency matrix (list of lists of floats) into a hard adjacency matrix.
    Each element greater than or equal to threshold becomes 1; otherwise 0.
    """
    # convert threshold in range [0,1] to [-1,1] because matrix is range [-1,1]
    # threshold = (threshold)*2 - 1
    print(matrix)
    return [[ [1.0 if elem >= threshold else 0.0 for elem in cell ] for cell in row] for row in matrix]
    # return [[ [1.0 if elem <= threshold else 0.0 for elem in cell ] for cell in row] for row in matrix]

def process_file(input_file, output_file, threshold):
    data = read_jsonl(input_file)
    updated_data = []
    for entry in data:
        # Assume the adjacency_matrix is stored as a nested list of numbers.
        soft_adj = entry.get("adjacency_matrix")
        if soft_adj is not None:
            hard_adj = threshold_adjacency_matrix(soft_adj, threshold)
            # Update the entry with the new hard adjacency matrix.
            entry["adjacency_matrix"] = hard_adj
        updated_data.append(entry)
    write_jsonl(output_file, updated_data)
    print(f"Processed {len(updated_data)} entries and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert soft adjacency matrices to hard matrices in a JSONL file.")
    # parser.add_argument("--input_file", type=str, default = "../results/bios/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl", required=True, help="Path to the input JSONL file")
    # parser.add_argument("--output_file", type=str, default = "../results/bios/llama3-8b-instruct/llama3-8b-instruct_threshold_graph_samples.jsonl", required=True, help="Path to the output JSONL file")
    # parser.add_argument("--input_file", type=str, default = "../results/longfact/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl", required=True, help="Path to the input JSONL file")
    # parser.add_argument("--output_file", type=str, default = "../results/longfact/llama3-8b-instruct/llama3-8b-instruct_threshold_graph_samples.jsonl", required=True, help="Path to the output JSONL file")

    parser.add_argument("--input_file", type=str, default = "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_nli_hybrid_graph_samples.jsonl", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, default = "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_threshold_nli_hybrid_graph_samples.jsonl", required=True, help="Path to the output JSONL file")


    parser.add_argument("--threshold", type=float, default = 0.2, required=True, help="Threshold value between 0 and 1")

    args = parser.parse_args()
    process_file(args.input_file, args.output_file, args.threshold)

    # python generate_threshold_graphs.py --threshold 0.2 --input_file "../results/bios/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl" --output_file "../results/bios/llama3-8b-instruct/llama3-8b-instruct_threshold_graph_samples.jsonl"
    # python generate_threshold_graphs.py --threshold 0.2 --input_file "../results/longfact/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl" --output_file "../results/longfact/llama3-8b-instruct/llama3-8b-instruct_threshold_graph_samples.jsonl"
    # python generate_threshold_graphs.py --threshold 0.2 --input_file "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_soft_graph_samples.jsonl" --output_file "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_threshold_graph_samples.jsonl"
    # python generate_threshold_graphs.py --threshold 0.2 --input_file "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_nli_hybrid_graph_samples.jsonl" --output_file "../results/wildhallu/llama3-8b-instruct/llama3-8b-instruct_threshold_nli_hybrid_graph_samples.jsonl"
