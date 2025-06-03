import os
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
import pytz
import argparse
import random
import numpy as np
SEED = 927
random.seed(SEED)

longfact_selected_topics = ['international-law', 
'moral-disputes', 
'movies', 
'electrical-engineering', 
'architecture', 
'prehistory' ]


def read_jsonl_files(folder_path):
    print(folder_path)
    all_files = os.listdir(folder_path)
    print(all_files)
    # jsonl_files = [file for file in all_files if (file.endswith(".jsonl") and file in longfact_selected_topics)]
    # jsonl_files = [file for file in all_files if ('jsonl' in file and file in longfact_selected_topics)]
    jsonl_files = [
    file for file in all_files 
    # if 'jsonl' in file and any(topic in file for topic in longfact_selected_topics)
]
    print(jsonl_files)

    data = []
    for jsonl_file in jsonl_files:
        file_path = os.path.join(folder_path, jsonl_file)
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                data.append(item)
    print("Loaded {} items from {} files".format(len(data), len(jsonl_files)))
    return data


def get_prompt(args, tokenizer, original_prompt):
    if "mistral" in args.model_name:
        messages = [
            {"role": "user", "content": original_prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": original_prompt}
        ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt


def get_responses(args, usage):
    input_prompts = []
    original_prompts = []
    topics = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    output_dir = f'../results/{args.dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.dataset == "bios":
        with open("../data/dataset/bio_entities.txt") as f:
            entities = [line.strip() for line in f]
        
        if args.debug:
            entities = entities[:1]

        for entity in entities:
            original_prompt = f"Tell me a bio of {entity}."
            prompt = get_prompt(args, tokenizer, original_prompt)
            input_prompts.append(prompt)
            original_prompts.append(original_prompt)
            topics.append(entity)
    
    elif args.dataset == "longfact":
        data = read_jsonl_files("../data/dataset/longfact-objects_gpt4_01-12-2024_noduplicates")
        
        if args.debug:
            data = data[:1]

        for item in data:
            original_prompt = item['prompt']
            prompt = get_prompt(args, tokenizer, original_prompt)
            input_prompts.append(prompt)
            original_prompts.append(original_prompt)
            topics.append(None)

    elif args.dataset == "convergence_analysis_dataset":
        data = read_jsonl_files("../data/dataset/convergence_analysis_dataset")
        
        if args.debug:
            data = data[:1]

        for item in data:
            original_prompt = item['prompt']
            prompt = get_prompt(args, tokenizer, original_prompt)
            input_prompts.append(prompt)
            original_prompts.append(original_prompt)
            topics.append(None)

    elif args.dataset == "wildhallu":
        data = load_dataset("wentingzhao/WildHallucinations", split="train")
        
        # Randomly sample 1000 prompts with a fixed seed
        data = data.shuffle(seed=SEED)
        data = data.select(range(1000))
        # data = data.select(range(500))
        # Option 2: Use a quantile to set the threshold (e.g., top 10% perplexity)
        # perps = np.array(data["perplexity"])
        # quantile_threshold = np.percentile(perps, 90)
        # high_perp_data = data.filter(lambda x: x["perplexity"] >= quantile_threshold)
        # # Shuffle and select 100 data points
        # data = high_perp_data.shuffle(seed=SEED).select(range(100))

        if args.debug:
            data = data.select([0])
        
        for item in data:
            # original_prompt = "In a paragraph, could you tell me what you know about {}?".format(item['entity'])
            original_prompt = "Could you tell me all you know about {} in a paragraph?".format(item['entity'])
            prompt = get_prompt(args, tokenizer, original_prompt)
            input_prompts.append(prompt)
            original_prompts.append(original_prompt)
            topics.append(item['entity'])

    elif args.dataset == "eli5":
        data = load_dataset("Pavithree/eli5", split="train")
        
        # Randomly sample 1000 prompts with a fixed seed
        data = data.shuffle(seed=SEED)
        data = data.select(range(200))
        # data = data.select(range(500))
        # Option 2: Use a quantile to set the threshold (e.g., top 10% perplexity)
        # perps = np.array(data["perplexity"])
        # quantile_threshold = np.percentile(perps, 90)
        # high_perp_data = data.filter(lambda x: x["perplexity"] >= quantile_threshold)
        # # Shuffle and select 100 data points
        # data = high_perp_data.shuffle(seed=SEED).select(range(100))

        if args.debug:
            data = data.select([0])
        
        for item in data:
            # original_prompt = "In a paragraph, could you tell me what you know about {}?".format(item['entity'])
            original_prompt = "Could you answer in a paragraph about the following question. {}".format(item['title'])
            prompt = get_prompt(args, tokenizer, original_prompt)
            input_prompts.append(prompt)
            original_prompts.append(original_prompt)
            topics.append(item['title'])

    elif args.dataset == "sciqa":
        data = load_dataset("fangyuan/longform_sciqa", split="train")
        
        # Randomly sample 1000 prompts with a fixed seed
        data = data.shuffle(seed=SEED)
        # data = data.select(range(200))
        # data = data.select(range(500))
        # Option 2: Use a quantile to set the threshold (e.g., top 10% perplexity)
        # perps = np.array(data["perplexity"])
        # quantile_threshold = np.percentile(perps, 90)
        # high_perp_data = data.filter(lambda x: x["perplexity"] >= quantile_threshold)
        # # Shuffle and select 100 data points
        # data = high_perp_data.shuffle(seed=SEED).select(range(100))

        if args.debug:
            data = data.select([0])
        
        for item in data:
            # original_prompt = "In a paragraph, could you tell me what you know about {}?".format(item['entity'])
            original_prompt = "Could you answer in a paragraph about the following question. {}".format(item['question'])
            prompt = get_prompt(args, tokenizer, original_prompt)
            input_prompts.append(prompt)
            original_prompts.append(original_prompt)
            topics.append(item['question'])


    else:
        raise ValueError(f"Dataset {args.dataset} is not supported")
    
    if usage == "answers":
        # Define sampling parameters
        sampling_params = SamplingParams(
            n=1,
            temperature=1,
            top_p=0.95,
            max_tokens=512,
            stop_token_ids=[tokenizer.eos_token_id],
            skip_special_tokens=True,
        )
        
        llm = LLM(
            model=args.model_path, 
            tensor_parallel_size=len(args.cuda_devices.split(",")),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=1024, # For input prompts 
            )        

        # Generate outputs
        outputs = llm.generate(input_prompts, sampling_params)

        # Collect data
        answers = []
        for item_outputs in outputs:
            answers.append(item_outputs.outputs[0].text)

        # Save the results
        output_dir = f'{output_dir}/{args.model_name}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = f'{output_dir}/{args.model_name}_answers.jsonl'
        with open(output_file, 'w') as f:
            for original_prompt, answer, topic in zip(original_prompts, answers, topics):
                f.write(json.dumps({
                    "prompt": original_prompt, 
                    "answer": answer,
                    "topic": topic
                    }) + '\n')

    elif usage == "samples":
        # Define sampling parameters
        sampling_params = SamplingParams(
            n=20, #500,
            temperature=1,
            top_p=0.95,
            max_tokens=512,
            stop_token_ids=[tokenizer.eos_token_id],
            skip_special_tokens=True,
            seed=SEED
        )

        print(f"Cuda Devices: {args.cuda_devices}")
        llm = LLM(
            model=args.model_path, 
            tensor_parallel_size=len(args.cuda_devices.split(",")),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=1024, # For input prompts 
            seed=SEED
        )

        # Generate outputs
        outputs = llm.generate(input_prompts, sampling_params)

        # Collect data
        samples = []
        for item_outputs in outputs:
            responses = []
            for single_response in item_outputs.outputs:
                responses.append(single_response.text)
            samples.append(responses)

        # Save the results
        output_dir = f'{output_dir}/{args.model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f'{output_dir}/{args.model_name}_samples.jsonl'
        with open(output_file, 'w') as f:
            for original_prompt, responses, topic in zip(original_prompts, samples, topics):
                if args.dataset =='longfact':
                    f.write(json.dumps({
                    "prompt": original_prompt, 
                    "responses": responses,
                    "topic": original_prompt
                    }) + '\n')
                else:
                    f.write(json.dumps({
                        "prompt": original_prompt, 
                        "responses": responses,
                        "topic": topic
                        }) + '\n')
        
    else:
        raise ValueError(f"Usage {usage} is not supported")

def main(args):
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ["HF_TOKEN"] = "---------"

    if args.model_name == "llama3-8b-instruct":
        args.model_path = "meta-llama/Meta-Llama-3-8B-Instruct"#"huggingface/Meta-Llama-3-8B-Instruct"
    elif args.model_name == "llama3-70b-instruct":
        args.model_path = "huggingface/Meta-Llama-3-70B-Instruct"
    elif args.model_name == "mistral-7b-instruct":
        args.model_path = "mistralai/Mistral-7B-Instruct-v0.2" #"/apdcephfs_qy3/share_733425/timhuang/cindychung/Mistral-7B-Instruct-v0.2"
    elif args.model_name == "mistral-8-7b-instruct":
        args.model_path = "huggingface/Mixtral-8x7B-Instruct-v0.1"
    elif args.model_name == "qwen2-7b-instruct":
        args.model_path = "Qwen/Qwen2-7B-Instruct" #"huggingface/Qwen2-7B-Instruct"
    elif args.model_name == "qwen2-57b-instruct":
        args.model_path = "Qwen/Qwen2-57B-A14B-Instruct" #"huggingface/Qwen2-57B-A14B-Instruct"
    elif args.model_name == "qwen2-72b-instruct":
        args.model_path = "huggingface/Qwen2-72B-Instruct"
    elif args.model_name == "falcon-7b-instruct":
        args.model_path = "tiiuae/falcon-7b-instruct"
    elif args.model_name == "llama3.3-70b-instruct":
        args.model_path = "meta-llama/Llama-3.3-70B-Instruct"
    else:
        raise ValueError(f"Model {args.model_name} is not supported")
    
    if args.generate_answers:
        get_responses(args, usage="answers")

    if args.generate_samples:
        get_responses(args, usage="samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--cuda_devices", type=str, default="4,5", help="CUDA visible devices")
    parser.add_argument("--dataset", type=str, default="bios", help="Dataset for LLM")
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct", help="Model ID for LLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--generate_answers", action='store_true', help="Generate answers")
    parser.add_argument("--generate_samples", action='store_true', help="Generate samples")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")

    args = parser.parse_args()
    main(args)

'''
python generate_responses_vllm.py \
    --cuda_devices 1 \
    --dataset bios \
    --model_name qwen2-7b-instruct \
    --generate_samples \
    --debug
'''