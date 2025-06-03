import os
import json
import argparse
import sys
sys.path.append('../utils')
from luq_vllm import LUQ_vllm
from luq import LUQ
from dis_vllm import DIS_vllm
import numpy as np
from tqdm import tqdm

def setup_environment():
    os.environ["HF_TOKEN"] = "-----"
    os.environ['HF_HOME'] = 'huggingface'
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    #os.environ["HF_TOKEN"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #"6,7"


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
        return data
        
def main(args):
    setup_environment()

    if args.confidence_type == "generative":
        assert args.confidence_method in ["binary", "multiclass"]
    elif args.confidence_type == "luq":
        assert args.confidence_method in ["binary"]
    elif args.confidence_type == "discriminative":
        assert args.confidence_method in ["single", "context", "rating"]
    else:
        raise ValueError(f"Unsupported confidence type: {args.confidence_type}")

    atomic_file_path = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts_samples.jsonl"

    atomic_facts = read_jsonl_file(atomic_file_path)

    samples_file_path = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts_samples.jsonl"

    samples = read_jsonl_file(samples_file_path)

    # Sometimes we only use part of the data in fact checking (e.g. 500)
    samples = samples[:len(atomic_facts)]

    if args.debug:
        print("Debug mode.")
        atomic_facts = atomic_facts[:2]
        samples = samples[:2]

    
    # if args.confidence_type == 'luq':
    #     save_file = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_{args.confidence_type}_confidence_{args.confidence_method}.jsonl"
    save_file = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_{args.confidence_type}_confidence_{args.confidence_method}.jsonl"

    if os.path.exists(save_file) and not args.overwrite:
        print(f"File {save_file} already exists.")
        exit()
    
    # Initialize the uncertainty calculator
    if args.confidence_type == "generative":
        # qwen2-7b-Instruct
        # luq_vllm = LUQ_vllm(nli_model="meta-llama/Meta-Llama-3-8B-Instruct", method=args.confidence_method, gpu_memory_utilization=args.gpu_memory_utilization, cuda_devices=args.cuda_devices)
        luq_vllm = LUQ_vllm(nli_model="qwen2-7b-Instruct", method=args.confidence_method, gpu_memory_utilization=args.gpu_memory_utilization, cuda_devices=args.cuda_devices)

    elif args.confidence_type == "luq":
        luq_ = LUQ(nli_model="potsawee/deberta-v3-large-mnli", method="luq", gpu_memory_utilization=args.gpu_memory_utilization, cuda_devices=args.cuda_devices)

    if args.confidence_type == "discriminative":
        dis_vllm = DIS_vllm(model=args.model_name, method=args.confidence_method, gpu_memory_utilization=args.gpu_memory_utilization, cuda_devices=args.cuda_devices)

    results_to_save = []
    for item, item_samples in tqdm(zip(atomic_facts, samples), total=len(atomic_facts)):
        # First calculate the generative uncertainty
        try:
            assert item["prompt"] == item_samples['prompt']
        except AssertionError:
            print(f"Assertion failed: \n {item['prompt']} \n {item_samples['prompt']}")
        prompt = item["prompt"]
        # answer = item["answer"]
        # print(item, item_samples)
        samples = item_samples['responses'][1:] # Get all responses other than reference (first response)
        atomic_response = item["atomic_facts"][0] #item["atomic_facts"][1] # Get the reference atomic claims

        if args.confidence_type == "generative":
            confidence_scores, raw_scores = luq_vllm.predict(
                sentences=atomic_response,              
                sampled_passages=samples,
            )
            item["confidence_scores"] = confidence_scores.tolist()
            item["raw_scores"] = raw_scores.tolist()
            item["uncertainty_scores"] = 1 - np.mean(np.array(item["confidence_scores"]))

        elif args.confidence_type == "discriminative":
            confidence_scores, raw_scores = dis_vllm.predict(
                context=prompt,
                targets=atomic_response,
            )
            item["confidence_scores"] = confidence_scores.tolist()
            item["raw_scores"] = raw_scores.tolist()
            item["uncertainty_scores"] = 1 - np.mean(np.array(item["confidence_scores"]))

        else:
            uncertainty_scores = luq_.predict(
                sampled_passages=samples,
                sentences=atomic_response,
            )
            item["uncertainty_scores"] = uncertainty_scores

        
        results_to_save.append(item)
    
    # Save as jsonl
    with open(save_file, "w") as f:
        for item in results_to_save:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files and generate uncertainty charts.")
    parser.add_argument('--confidence_type')#, choices=['generative', 'discriminative', 'luq'], help='Type of confidence to calculate.')
    parser.add_argument('--confidence_method', help='Generative method to use.')
    parser.add_argument('--model_name', type=str, help='Model name.')
    parser.add_argument('--dataset', choices=['bios', 'longfact', 'wildhallu', 'motivation', 'convergence_analysis_dataset', 'eli5', 'sciqa'], help='Dataset to use.')
    # parser.add_argument('--cuda_devices', type=str, default="0", help='CUDA devices to use.')
    parser.add_argument('--cuda_devices', type=str, default="4,5", help='CUDA devices to use.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization.')
    parser.add_argument('--debug', action='store_true', help='Debug mode (no saving results).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    main(args)

    # python calculate_uncertainty_samples.py --confidence_type 'generative' --confidence_method 'binary' --model_name 'llama3-8b-instruct' --dataset 'bios' 
    # python calculate_uncertainty_samples.py --confidence_type 'luq' --confidence_method 'binary' --model_name 'llama3-8b-instruct' --dataset 'bios' 

    # python calculate_uncertainty_samples.py --confidence_type 'generative' --confidence_method 'binary' --model_name 'llama3-8b-instruct' --dataset 'longfact' 
    # python calculate_uncertainty_samples.py --confidence_type 'luq' --confidence_method 'binary' --model_name 'llama3-8b-instruct' --dataset 'longfact' 

    # python calculate_uncertainty_samples.py --confidence_type 'generative' --confidence_method 'binary' --model_name 'llama3-8b-instruct' --dataset 'wildhallu' 
    # python calculate_uncertainty_samples.py --confidence_type 'luq' --confidence_method 'binary' --model_name 'llama3-8b-instruct' --dataset 'wildhallu' 