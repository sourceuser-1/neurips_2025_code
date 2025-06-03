import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from datasets import load_dataset
import argparse
from scipy.stats import kendalltau


def calculate_calibration_metrics(scores, veracity, inconsistency, num_bins=10, eps=1e-8):
    import numpy as np
    from scipy.stats import spearmanr, pearsonr

    # Ensure inputs are numpy arrays
    scores = np.array(scores)
    veracity = np.array(veracity)

    # perm = np.random.permutation(len(scores))
    # scores = scores[perm[:limited_num]]
    # veracity = veracity[perm[:limited_num]]
    
    # Compute correlation metrics
    sc, _ = spearmanr(scores, veracity)
    pc, _ = pearsonr(scores, veracity)
    tau, _ = kendalltau(scores, veracity)
    
    # UCCE: Uniform Continuous Calibration Error
    # Uniform binning based on scores (x-axis)
    bin_edges = np.linspace(np.min(scores), np.max(scores), num_bins + 1)
    bin_indices = np.digitize(scores, bin_edges, right=False)
    ucces = []
    for i in range(1, num_bins + 1):
        mask = (bin_indices == i)
        if np.sum(mask) == 0:
            continue
        bin_scores = scores[mask]
        bin_veracity = veracity[mask]
        
        # Normalize scores using the range of scores in this bin
        min_x = np.min(bin_scores)
        max_x = np.max(bin_scores)
        norm_bin_scores = (bin_scores - min_x) / (max_x - min_x + eps)
        
        # Normalize veracity using the range of veracity in this bin
        min_y = np.min(bin_veracity)
        max_y = np.max(bin_veracity)
        norm_bin_veracity = (bin_veracity - min_y) / (max_y - min_y + eps)
        
        # Compute error as the absolute difference between the means of the normalized values
        error = np.abs(np.mean(norm_bin_scores) - np.mean(norm_bin_veracity))
        ucces.append(error)
    ucc_e = np.mean(ucces) if ucces else 0.0

    # QCCE: Quantile Continuous Calibration Error
    # Here we form bins so that each bin has an equal number of points.
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_veracity = veracity[sorted_idx]
    qcce_total = 0.0
    bin_size = len(scores) // num_bins
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else len(scores)
        bin_scores = sorted_scores[start:end]
        bin_veracity = sorted_veracity[start:end]
        
        # Normalize separately
        min_x = np.min(bin_scores)
        max_x = np.max(bin_scores)
        norm_bin_scores = (bin_scores - min_x) / (max_x - min_x + eps)
        
        min_y = np.min(bin_veracity)
        max_y = np.max(bin_veracity)
        norm_bin_veracity = (bin_veracity - min_y) / (max_y - min_y + eps)
        
        error = np.abs(np.mean(norm_bin_scores) - np.mean(norm_bin_veracity))
        qcce_total += error
    qcc_e = qcce_total / num_bins

    return {
        'SC': round(sc, 4),
        'PC': round(pc, 4),
        # 'tau': round(tau, 4),
        'UCCE': round(ucc_e, 4),
        'QCCE': round(qcc_e, 4)
    }

def get_mean(your_list):
    if len(your_list) == 0:
        return 0.0
    return sum(1 if s == 'S' else 0 for s in your_list) / len(your_list)

# def create_scatter_plot(topics, scores, veracity, inconsistency,  output_path):
#     plt.figure(figsize=(10, 6))
#     veracity = inconsistency
#     plt.scatter(scores, veracity , alpha=0.6, edgecolors='w', linewidth=0.5)
#     # plt.scatter(scores, inconsistency, alpha=0.6, edgecolors='w', linewidth=0.5)
    
#     # Add labels and title
#     plt.xlabel('Dissimilarity Score (avg_cost)', fontsize=12)
#     plt.ylabel('Veracity', fontsize=12)
#     plt.title('Veracity vs Dissimilarity Score', fontsize=14)
    
#     # Add grid for better readability
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Save the plot
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()

def create_scatter_plot(topics, scores, veracity, inconsistency, output_path):
    plt.figure(figsize=(10, 6))
    # Note: In your original code, veracity is being overwritten by inconsistency.
    # Make sure to use the correct variable if they differ.
    # For this example, I'll assume you want to plot veracity vs scores.
    plt.scatter(scores, veracity, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    # Add text labels next to each point
    for topic, x, y in zip(topics, scores, veracity):
        plt.annotate(topic, (x, y),
                     textcoords="offset points",  # how to position the text
                     xytext=(5, 5),               # distance from text to points (x,y)
                     ha='left', fontsize=5)
    
    # Add labels and title
    plt.xlabel('Dissimilarity Score (avg_cost)', fontsize=12)
    plt.ylabel('Veracity', fontsize=12)
    plt.title('Veracity vs Dissimilarity Score', fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_jsonl(args, input_path_scores, input_path_veracity, input_path_inconsistency,  output_jsonl_path, output_plot_path):
    with open(input_path_scores, 'r') as f:
        topics = [json.loads(line)['topic'] for line in f]
    selected_topics = []
    if "wildhallu" in input_path_scores:
        data = load_dataset("wentingzhao/WildHallucinations", split="train")
        for item in data:
            if item['entity'] in topics: # and item['wiki'] == 0:
                selected_topics.append(item['entity'])

        # Read input JSONL   
        with open(input_path_scores, 'r') as f:
            entries_scores = [json.loads(line) for line in f if json.loads(line)['topic'] in selected_topics]

        with open(input_path_veracity, 'r') as f:
            entries_veracity = [json.loads(line) for line in f if json.loads(line)['topic'] in selected_topics]
    
    else:    
        # Read input JSONL   
        with open(input_path_scores, 'r') as f:
            entries_scores = [json.loads(line) for line in f]

        with open(input_path_veracity, 'r') as f:
            entries_veracity = [json.loads(line) for line in f]

    with open(input_path_inconsistency, 'r') as f:
        entries_inconsistency = [json.loads(line) for line in f]

    topics_safe = np.array([entry['topic'] for entry in entries_veracity if "Error" not in entry['atomic_facts_veracity']])
    topics_ours = np.array([entry['topic'] for entry in entries_scores ])

    topics_safe = {entry['topic']:get_mean(entry['atomic_facts_veracity']) for entry in entries_veracity if "Error" not in entry['atomic_facts_veracity']}
    topics_ours = {entry['topic']:entry['avg_cost'] for entry in entries_scores }

    veracity = []
    scores = []
    for tpc in topics_safe:
        if tpc in topics_ours:
            veracity.append(topics_safe[tpc])
            scores.append(topics_ours[tpc])
    veracity = np.array(veracity)
    scores = np.array(scores)

    finite_mask_veracity = np.isfinite(veracity)
    finite_mask_scores = np.isfinite(scores)
    veracity = veracity[finite_mask_scores]
    scores = scores[finite_mask_scores]


    # for i in range(len(topics_ours)):
    #     print(i, topics_ours[i], topics_safe[i])
    
    # # Extract data for plotting
    # topics = np.array([entry['topic'] for entry in entries_veracity if "Error" not in entry['atomic_facts_veracity']])
    # scores = np.array([entry['avg_cost'] for entry in entries_scores if entry['topic'] in topics])
    # veracity = np.array([get_mean(entry['atomic_facts_veracity']) for entry in entries_veracity if entry['topic'] in topics])
    inconsistency = np.array([entry['avg_inconsistency_score'] for entry in entries_inconsistency if entry['topic'] in topics] )

    # mean = sum(1 if s == 'S' else 0 for s in your_list) / len(your_list)
    min_ = min(scores.shape[0], veracity.shape[0])
    min_ = 499#500
    scores = scores[:min_]
    veracity = veracity[:min_]
    print(scores.shape, veracity.shape)
    
    # Create scatter plot
    create_scatter_plot(topics, scores, veracity, inconsistency, output_plot_path)
    
    # Calculate calibration metrics
    metrics = calculate_calibration_metrics(scores, veracity, inconsistency)
    
    # Create output entries (same metrics for all topics)
    output_entries = []
    for entry in entries_scores:
        output_entry = {
            'topic': entry['topic'],
            'SC': metrics['SC'],
            'PC' : metrics['PC'],
            #'tau': metrics['tau'], 
            'UCCE': metrics['UCCE'],
            'QCCE': metrics['QCCE']
        }
        output_entries.append(output_entry)
    
    # Write output JSONL
    # with open(output_jsonl_path, 'w') as f:
    #     for entry in output_entries:
    #         f.write(json.dumps(entry) + '\n')

    return metrics

def main(args):

    if args.dataset == 'bios':
        input_path_veracity = f'../results/{args.dataset}/{args.model}/{args.model}_atomic_facts_RAG_veracity.jsonl'
    else:
        input_path_veracity = f'../results/{args.dataset}/{args.model}/{args.model}_atomic_facts_SAFE_veracity.jsonl'
    
    metrics = process_jsonl(
        args, 
        #input_path_scores = f'../results/{args.dataset}/{args.model}/{args.model}_threshold_dissimilarity_samples.jsonl', 
        ####input_path_scores = f'../results/{args.dataset}/{args.model}/{args.model}_{args.method}_graph_dissimilarities.jsonl', 
        ####### input_path_scores = f'../results/{args.dataset}/{args.model}/{args.model}_{args.method}_graph_dissimilarities_1.jsonl',
        input_path_scores = f'../results/{args.dataset}/{args.model}/{args.model}_{args.method}_graph_dissimilarities_roberta.jsonl', 
        # input_path_scores = f'../results/{args.dataset}/{args.model}/{args.model}_{args.method}_graph_dissimilarities_{args.alpha}.jsonl', 
        input_path_veracity = input_path_veracity, 
        input_path_inconsistency = f'../results/bios/llama3-8b-instruct/llama3-8b-instruct_consistency_context_scores.jsonl',
        output_jsonl_path = f'../results/{args.dataset}/{args.model}/{args.model}_{args.method}_calibration.jsonl',
        output_plot_path = f'../results/{args.dataset}/{args.model}/{args.model}_{args.method}_plot.png',
    )

    return metrics

if __name__ == "__main__":
    # batch_alignment_cost_entailment
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--dataset", type=str, default="bios", help="Dataset")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha value in graph dissimilarity")
    parser.add_argument("--model", type=str, default="llama3-8b-instruct", help="Model ID for LLM")
    parser.add_argument("--confidence_type", type=str, default="", help="Generate answers")
    parser.add_argument("--confidence_method", type=str, default="binary", help="Generate answers")
    parser.add_argument("--method", type=str, default="batch_alignment_cost_entailment", help="Model ID for LLM")


    args = parser.parse_args()
    main(args)

    