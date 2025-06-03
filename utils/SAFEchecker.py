


import requests
import json
from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" #"2,3" #"4,5"

class SafeEvaluator:
    def __init__(self,
                 model_path="Qwen/Qwen2.5-32B-Instruct",
                 tensor_parallel_size=2,
                 gpu_memory_utilization=0.9,
                 device="cuda",
                 serper_api_key="------------",
                 serper_endpoint="https://google.serper.dev/search",
                 max_tokens=100,
                 temperature=0.7,
                 top_k=40,
                 top_p=0.9):
        """
        Initialize the SAFE evaluator with a Qwen model via vllm and Serper API configuration.
        """
        self.serper_api_key = serper_api_key
        self.serper_endpoint = serper_endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            device=device
        )

    def qwen_generate(self, prompt: str) -> str:
        """
        Query the Qwen model via vllm to generate text for a given prompt.
        """
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        # The generate method returns a list; we assume the structure is outputs[0].outputs[0].text
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_search_query(self, fact: str, topic: str) -> str:
        """
        Uses Qwen to generate a concise search query for the given fact and topic.
        """
        prompt = (
            f"Given the fact:\n\"{fact}\"\n"
            f"and the topic \"{topic}\", generate a concise search query that can be used "
            "to verify this fact with online sources. Please output only the query."
        )
        query = self.qwen_generate(prompt)
        return query

    def search_serper(self, query: str) -> list:
        """
        Uses the Serper API to retrieve search results for the given query.
        Returns a list of search result snippets.
        """
        headers = {
            "X-API-Key": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 3}  # Retrieve top 3 results
        response = requests.post(self.serper_endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            for result in data.get("organic", []):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                results.append(f"{title}. {snippet}")
            return results
        else:
            print(f"Serper API error: {response.status_code}, {response.text}")
            return []

    def analyze_search_results(self, fact: str, search_results: list) -> str:
        """
        Uses Qwen to analyze search results and decide whether the fact is supported.
        Returns 'supported' or 'not supported'.
        """
        results_summary = "\n".join(f"- {res}" for res in search_results)
        prompt = (
            f"Given the fact:\n\"{fact}\"\n"
            f"and the following search results:\n{results_summary}\n"
            "Determine if these search results provide clear evidence that supports the fact. "
            "Answer with only one word: 'supported' or 'not supported'."
        )
        analysis = self.qwen_generate(prompt).strip().lower()
        return "supported" if analysis.startswith("supported") else "not supported"

    def is_fact_supported(self, fact: str, topic: str) -> bool:
        """
        For a given atomic fact and topic, generate a search query, retrieve search results,
        and analyze the results to determine if the fact is supported.
        """
        query = self.generate_search_query(fact, topic)
        search_results = self.search_serper(query)
        analysis = self.analyze_search_results(fact, search_results)
        return 'S' if analysis.startswith("supported") else 'NS'#analysis == "supported"

    def are_atomic_facts_supported(self, atomic_facts: list, topic: str) -> list:
        """
        Given a list of atomic facts and a topic, returns a list of booleans indicating
        whether each fact is supported.
        """
        support_list = []
        for fact in atomic_facts:
            support_list.append(self.is_fact_supported(fact, topic))
        return support_list


# --- Example usage ---
if __name__ == "__main__":
    topic = "economics"
    atomic_facts = [
        "The Federal Reserve was established in 1813 to regulate the US economy.",
        "Inflation is a key indicator of economic performance.",
        "The Pacific Ocean is the largest ocean on Earth."
    ]
    
    evaluator = SafeEvaluator()
    support_list = evaluator.are_atomic_facts_supported(atomic_facts, topic)
    print("Support for atomic facts:", support_list)
