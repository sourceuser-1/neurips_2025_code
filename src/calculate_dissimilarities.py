#!/usr/bin/env python3
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append('../utils')
from graph_dissimilarity import batch_alignment_cost, batch_alignment_cost_entailment

def _process_topic_job(job):
    """
    Worker function to compute dissimilarity for one topic.
    job is a tuple: (topic, threshold_entry, entailment_entry, method, alpha)
    """
    topic, thresh_entry, entail_entry, method, alpha, raw_scores_entry, dataset = job

    # Load adjacency mats & embeddings
    adj_mats = [np.array(adj) for adj in thresh_entry['adjacency_matrix']]
    node_embs = [np.array(emb) for emb in thresh_entry.get('node_embeddings', [])]

    # Prepare reference
    if dataset == "bios":
        ref_mat  = adj_mats[0]
        ref_embs = node_embs[0] if node_embs else None  
        # Build batch_graphs
        batch_graphs = [
            (adj_mats[i], node_embs[i] if i < len(node_embs) else None)
            for i in range(1, len(adj_mats))
        ]  
    else:
        ref_mat  = adj_mats[0]
        ref_embs = node_embs[0] if node_embs else None
        # Build batch_graphs
        batch_graphs = [
            (adj_mats[i], node_embs[i] if i < len(node_embs) else None)
            for i in range(1, len(adj_mats))
        ]

    out = {}
    if method == "batch_alignment_cost":
        # Pure structural+semantic batch
        if batch_graphs:
            costs, _, struct_costs, semantic_costs = batch_alignment_cost(
                (ref_mat, ref_embs),
                batch_graphs,
                alpha=alpha
            )
            out = {
                'costs':          costs.tolist(),
                'struct_costs':   struct_costs.tolist(),
                'semantic_costs': semantic_costs.tolist(),
                'avg_cost':       float(np.mean(costs))
            }

    else:
        # entailment-based
        if entail_entry is None:
            raise KeyError(f"No entailment data for topic '{topic}'")
        ent_mats = [np.array(m) for m in entail_entry['entailment_matrices']]
        
        # build parallel entailment list (one per non-ref graph)
        batch_ent = [ent_mats[i-1] for i in range(1, len(adj_mats))]

        if batch_graphs:
            costs, _, struct_costs, semantic_costs = batch_alignment_cost_entailment(
                (ref_mat, ref_embs),
                batch_graphs,
                raw_scores_entry,
                batch_ent,
                alpha=alpha,
                threshold=0.7
            )
            out = {
                'costs':          costs.tolist(),
                'struct_costs':   struct_costs.tolist(),
                'semantic_costs': semantic_costs.tolist(),
                'avg_cost':       float(np.mean(costs))
            }

    return topic, out


def process_topic_dissimilarities(args,
                                  threshold_input_path,
                                  output_path,
                                  method,
                                  alpha=0.5,
                                  dataset=None,
                                  model=None,
                                  entailment_input_path=None):
    # (Re-)construct default paths if the user hasn't overridden them
    input_path_encompass_scores = f"../results/{dataset}/{model}/{model}_generative_confidence_binary.jsonl"
    ### threshold_input_path_wo_nli = f"../results/{dataset}/{model}/{model}_soft_graph_samples.jsonl"
    threshold_input_path_wo_nli = f"../results/{dataset}/{model}/{model}_soft_roberta_graph_samples.jsonl"
    threshold_input_path       = f"../results/{dataset}/{model}/{model}_nli_hybrid_graph_samples.jsonl"
    entailment_input_path      = entailment_input_path or f"../results/{dataset}/{model}/{model}_entailment_samples.jsonl"
    # output_path                = f"../results/{dataset}/{model}/{model}_{method}_graph_dissimilarities.jsonl"
    ### output_path                = f"../results/{dataset}/{model}/{model}_{method}_graph_dissimilarities_1.jsonl"
    output_path                = f"../results/{dataset}/{model}/{model}_{method}_graph_dissimilarities_roberta.jsonl"
    # output_path   = f"../results/{dataset}/{model}/{model}_{method}_graph_dissimilarities_{alpha}.jsonl"


    # 0) Load the encompassing information for each claim
    raw_scores_groups = defaultdict(list)
    with open(input_path_encompass_scores, 'r') as f:
            # entries_scores = [json.loads(line) for line in f]
            for line in tqdm(f, desc='Loading encompassing scores'):
                data = json.loads(line)
                raw_scores_groups[data['topic']].append(np.array(data['raw_scores']))

    # 1) Load threshold data
    threshold_groups = defaultdict(list)
    with open(threshold_input_path_wo_nli, 'r') as f:
        for line in tqdm(f, desc='Loading threshold data'):
            data = json.loads(line)
            threshold_groups[data['topic']].append(data)

    # 2) (Optional) Load entailment data
    entailment_groups = {}
    if method == "batch_alignment_cost_entailment":
        local_ent = defaultdict(list)
        with open(entailment_input_path, 'r') as f:
            for line in tqdm(f, desc='Loading entailment data'):
                d = json.loads(line)
                local_ent[d['topic']].append(d)
        entailment_groups = {t: entries for t, entries in local_ent.items()}

    # 3) Build job list
    jobs = []
    for topic, group_data in threshold_groups.items():
        if not group_data:
            continue
        thresh_entry = group_data[0]
        entail_entry = None
        raw_scores_entry = raw_scores_groups[topic][0]
        if method == "batch_alignment_cost_entailment":
            if topic not in entailment_groups:
                raise KeyError(f"Topic '{topic}' missing from entailment file")
            entail_entry = entailment_groups[topic][0]
        jobs.append((topic, thresh_entry, entail_entry, method, alpha, raw_scores_entry, dataset))

    # 4) Parallel dispatch
    results = {}
    with ProcessPoolExecutor() as executor:
        # map() returns results in the same order as `jobs`
        for topic, out in tqdm(
            executor.map(_process_topic_job, jobs),
            total=len(jobs),
            desc='Processing topics'
        ):
            results[topic] = out

    # 5) Write out
    with open(output_path, 'w') as wf:
        for topic, data in results.items():
            obj = {'topic': topic}
            obj.update(data)
            wf.write(json.dumps(obj) + '\n')

    return results


def cross_dissimilarity(adj1, emb1, adj2, emb2):
    costs, _, _, _ = batch_alignment_cost(
        (adj1, emb1),
        [(adj2, emb2)],
        alpha=0.5
    )
    print('-'*80)
    print(f"Cross paragraph Dissimilarity is {costs}")
    print('-'*80)


def print_dissimilarities(results):
    """Pretty-print dissimilarity results."""
    for topic, d in results.items():
        print(f"\nTopic: {topic}")
        print("-" * 40)
        print(f"Average dissimilarity: {d.get('avg_cost', 'N/A')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute topic graph dissimilarities (parallelized)."
    )
    parser.add_argument("--dataset", type=str, default="bios")
    parser.add_argument("--model",   type=str, default="llama3-8b-instruct")
    parser.add_argument("--method",  type=str,
                        choices=["batch_alignment_cost",
                                 "batch_alignment_cost_entailment"],
                        default="batch_alignment_cost")
    parser.add_argument("--threshold_input_file", type=str,
                        help="(Optional) override threshold JSONL path")
    parser.add_argument("--entailment_file", type=str,
                        help="(Optional) override entailment JSONL path")
    parser.add_argument("--output_file",     type=str,
                        help="(Optional) override output JSONL path")
    parser.add_argument("--alpha",           type=float, default=0.5)
    args = parser.parse_args()

    results = process_topic_dissimilarities(
        args,
        threshold_input_path=args.threshold_input_file,
        output_path=args.output_file,
        method=args.method,
        alpha=args.alpha,
        dataset=args.dataset,
        model=args.model,
        entailment_input_path=args.entailment_file
    )

    print("\nDissimilarity results:")
    print_dissimilarities(results)
