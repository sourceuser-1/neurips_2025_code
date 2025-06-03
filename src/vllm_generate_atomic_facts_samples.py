import os
import json
from tqdm import tqdm
import argparse
import time
from transformers import GPT2Tokenizer, AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["HF_TOKEN"] = "---------"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"#"6,7"#"4,5"#"6,7"

# Initialize the tokenizer and model for Llama3-8B-Instruct
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") #("meta-llama/Meta-Llama-3-70B-Instruct") #("meta-llama/Meta-Llama-3-8B-Instruct")
# llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=2, gpu_memory_utilization=0.6)#("meta-llama/Meta-Llama-3-70B-Instruct")#(model="meta-llama/Meta-Llama-3-8B-Instruct")
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", gpu_memory_utilization=0.9)#("meta-llama/Meta-Llama-3-70B-Instruct")#(model="meta-llama/Meta-Llama-3-8B-Instruct")

def estimate_cost(input_tokens, output_tokens):
    # Cost per million tokens for gpt-4-1106-preview model
    INPUT_COST_PER_MILLION = 5.00  # USD
    OUTPUT_COST_PER_MILLION = 15.00  # USD
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    total_cost = input_cost + output_cost
    return total_cost

def estimate_overall_cost(data):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    total_input_tokens = sum([sum([len(tokenizer.encode(resp)) for resp in item["responses"]]) for item in data])
    total_output_tokens = total_input_tokens
    total_cost = estimate_cost(total_input_tokens, total_output_tokens)
    return total_cost

def get_completion(user_prompt, retries=100, delay=2):
    global tokenizer, llm
    # Format the prompt using the tokenizer's chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    
    for attempt in range(retries):
        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()
        except Exception as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return None

prompt_template = """
Please breakdown the following passage into independent fact pieces. 

Step 1: For each sentence, you should break it into several fact pieces. Each fact piece should only contain one single independent fact. Normally the format of a fact piece is "subject + verb + object". If the sentence does not contain a verb, you can use "be" as the verb.

Step 2: Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting. 

Step 3: Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.

Here are some examples:

Example 1:
Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
### Michael Collins was born on October 31, 1930.
### Michael Collins is retired.
### Michael Collins is an American.
### Michael Collins was an astronaut.
### Michael Collins was a test pilot.
### Michael Collins was the Command Module Pilot.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Example 2:
League of Legends (often abbreviated as LoL) is a multiplayer online battle arena (MOBA) video game developed and published by Riot Games. 
### League of Legends is a video game.
### League of Legends is often abbreviated as LoL.
### League of Legends is a multiplayer online battle arena.
### League of Legends is a MOBA video game.
### League of Legends is developed by Riot Games.
### League of Legends is published by Riot Games.

Example 3:
Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.
### Emory University has a strong athletics program.
### Emory University competes in the National Collegiate Athletic Association Division I.
### Emory University competes in the Atlantic Coast Conference.
### Emory University is part of the ACC.
### Emory University's mascot is the Eagle.

Now it's your turn. Here is the passage: 

{}

You should only return the final answer. Now your answer is:
"""

def get_atomic_facts(args, generation_file_name):
    print('IN')
    data = []
    with open(generation_file_name, "r") as f:
        for line in f:
            data.append(json.loads(line))

    if args.debug:
        data = data[:2]

    estimated_cost = estimate_overall_cost(data)
    print(f"Estimated overall cost: ${estimated_cost:.6f}")

    output_file_name = f"../results/{args.dataset}/{args.model_name}/{args.model_name}_atomic_facts_samples.jsonl"
    
    if os.path.exists(output_file_name):
        with open(output_file_name, "r") as f:
            processed_items = [json.loads(line) for line in f.readlines()]
        
        data = [item for item in data if item["prompt"] not in {processed_item["prompt"] for processed_item in processed_items}]
        print(data)
    
    print(f"Continue from {len(data)} items")

    new_data = []
    for item in tqdm(data, desc=args.model_name):
        answer = item["responses"]
        atomic_facts_array = []
        raw_atomic_facts_array = []
        #answer = answer[:1]
        for x in answer:
            try:
                raw_atomic_facts = get_completion(prompt_template.format(x))
                raw_atomic_facts_array.append(raw_atomic_facts)
                atomic_facts = [fact.strip() for fact in raw_atomic_facts.split("###") if fact.strip()]
                atomic_facts_array.append(atomic_facts)
                # print(raw_atomic_facts)
            except Exception as e:
                print(f"Error: {e}. Skipping...")
                atomic_facts = []
                atomic_facts_array.append(atomic_facts)
                raw_atomic_facts = []
                raw_atomic_facts_array.append(raw_atomic_facts)

        new_item = {
            "prompt": item["prompt"],
            "responses": item["responses"],
            'topic': item['topic'],
            "atomic_facts": atomic_facts_array,
            "raw_atomic_facts": raw_atomic_facts_array
        }
        new_data.append(new_item)

        with open(output_file_name, "a") as f:
            f.write(json.dumps(new_item) + "\n")

         # cutoff if number of items is more than 500
        if len(new_data) > 500:
            break
    
    

def main(args):
    if args.dataset == "bios":
        generation_file_name = f"../results/bios/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    elif args.dataset == "longfact":
        generation_file_name = f"../results/longfact/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    elif args.dataset == "wildhallu":
        generation_file_name = f"../results/wildhallu/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    elif args.dataset == "motivation":
        generation_file_name = f"../results/motivation/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    elif args.dataset == "convergence_analysis_dataset":
        generation_file_name = f"../results/convergence_analysis_dataset/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    if args.dataset == "eli5":
        generation_file_name = f"../results/eli5/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    if args.dataset == "sciqa":
        generation_file_name = f"../results/sciqa/{args.model_name}/{args.model_name}_samples.jsonl"
        get_atomic_facts(args, generation_file_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLM")
    parser.add_argument("--dataset", type=str, default="bios", help="Dataset for LLM")
    parser.add_argument("--sub_dataset", type=str, default="architecture", help="Dataset for LLM")
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct", help="Model name")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    args = parser.parse_args()
    main(args)

'''
export OPENAI_API_KEY=""

python generate_atomic_facts.py --model_name llama3-8b-instruct --dataset bios --debug
'''