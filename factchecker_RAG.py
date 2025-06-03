


import os
import time
import wikipedia
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import GPT2Tokenizer
from vllm import LLM, SamplingParams
from rank_bm25 import BM25Okapi
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

INSTUCT_FACTCHECK_RAG_SINGLE = """Your task is to fact-check the following statement.
This statement is extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

Assign a veracity label: 'S' means the statement is factually correct; 'NS' means the statement is factually wrong.
Pay close attention to numbers, dates, and other details. You can refer to the provided knowledge source to verify the statement.

To assess the statement, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Statement: Lebron James is a basketball player.
Evidence: Lebron James is an American professional basketball player for the Los Angeles Lakers of the NBA.
Analysis: Lebron James is an American professional basketball player, so this is correct. $S$

### Statement: Obama was the 46th president of the United States.
Evidence: Barack Obama served as the 44th president of the United States.
Analysis: Barack Obama served as the 44th president, so this is incorrect. $NS$

### Statement: {atomic_fact}
Evidence: {retrieved_evidence}
"""

def split_into_passages(text, max_tokens=300, tokenizer=None):
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    lines = text.split("\n")
    passages = []
    current_passage = ""
    for line in lines:
        tokens = tokenizer.encode(current_passage + " " + line)
        if len(tokens) > max_tokens and current_passage:
            passages.append(current_passage.strip())
            current_passage = line
        else:
            current_passage += " " + line
    if current_passage.strip():
        passages.append(current_passage.strip())
    return passages

def retrieve_relevant_passages(passages, query, top_k=3):
    tokenized_passages = [p.split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = [passages[i] for i in top_indices]
    return "\n".join(retrieved)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Assign GPUs 4 and 5


class FactChecker(object):
    def __init__(self, max_evidence_length):
        self.llm_token_count = 0
        self.openai_cost = 0
        self.max_evidence_length = max_evidence_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.llm = LLM(
            model="Qwen/Qwen2.5-32B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization = 0.9,#0.9, 
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=["\n"]
        )

    def truncate_text(self, text):
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_evidence_length:
            tokens = tokens[:self.max_evidence_length]
            text = self.tokenizer.decode(tokens)
        return text

    def build_wikipedia_evidence(self, topic):
        wikipedia.set_lang("en")
        try:
            results = wikipedia.search(topic)
            # page = wikipedia.page(topic)
            page = wikipedia.page(results[0])
            return page.content
        except Exception as e:
            print(f"Error retrieving Wikipedia page for topic '{topic}': {e}")
            return ""

    # @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(20))
    @retry(wait=wait_random_exponential(min=0.1, max=0.5), stop=stop_after_attempt(20))
    def get_completion(self, user_prompt: str = ""):
        outputs = self.llm.generate([user_prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response

    def get_prompt_with_retrieved_evidence(self, topic, atomic_fact):
        full_evidence = self.build_wikipedia_evidence(topic)
        passages = split_into_passages(full_evidence, max_tokens=300, tokenizer=self.tokenizer)
        retrieved_evidence = retrieve_relevant_passages(passages, atomic_fact, top_k=3)
        retrieved_evidence = self.truncate_text(retrieved_evidence)
        instruct = INSTUCT_FACTCHECK_RAG_SINGLE.format(
            atomic_fact=atomic_fact,
            retrieved_evidence=retrieved_evidence
        )
        return instruct

    def get_veracity_labels(self, topic="", atomic_facts=[], knowledge_source=None, 
                                                 evidence_type=None):
        veracity_labels = []
        for fact in atomic_facts:
            for attempt in range(1):
                try:
                    user_prompt = self.get_prompt_with_retrieved_evidence(topic, fact)
                    completion = self.get_completion(user_prompt)
                    if "$S$" in completion:
                        veracity_labels.append("S")
                    else:# "$NS$" in completion:
                        veracity_labels.append("NS")
                    # else:
                    #     veracity_labels.append("Unknown")
                    break
                except Exception as e:
                    print(e)
                    print(f"Retrying ({attempt + 1}/3), wait for {2 ** (attempt + 1)} sec...")
                    time.sleep(2 ** (attempt + 1))
            else:
                veracity_labels.append("Error")
        return veracity_labels, []

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "-------"
    topic = "Kang Ji-hwan"
    fact_list = ["Kang Ji-hwan is a South Korean actor.", "Kang Ji-hwan is a South Korean singer.", "Kang Ji-hwan was born on August 6, 1977.", "Kang Ji-hwan was born in Seoul, South Korea.", "Kang Ji-hwan began his career as a member of the boy band \"Sechs Kies\".", "Kang Ji-hwan began his career in the late 1990s.", "The boy band \"Sechs Kies\" disbanded in 2000.", "Kang Ji-hwan transitioned to acting after his music career.", "Kang Ji-hwan made his acting debut in the 2001 film \"Break Out\".", "Kang Ji-hwan gained recognition for his role in \"Lovers in Paris\" in 2004.", "Kang Ji-hwan gained recognition for his role in \"Be Strong Geum-soon\" in 2005.", "Kang Ji-hwan has appeared in \"The World That They Live In\" in 2008.", "Kang Ji-hwan has appeared in \"The Fugitive Plan B\" in 2010.", "Kang Ji-hwan has appeared in \"Myung-wol the Spy\" in 2013.", "Kang Ji-hwan has appeared in \"My ID is Gangnam Beauty\" in 2018.", "Kang Ji-hwan received the Best Actor award at the 2005 SBS Drama Awards.", "Kang Ji-hwan received the award for his role in \"Be Strong Geum-soon\".", "Kang Ji-hwan is known for his versatility as an actor.", "Kang Ji-hwan has played a wide range of characters.", "Kang Ji-hwan's characters span various genres.", "Kang Ji-hwan is also a talented singer.", "Kang Ji-hwan has released several solo albums.", "Kang Ji-hwan is married to actress Park Ha-sun.", "Kang Ji-hwan has two children with Park Ha-sun."]
    
    fc = FactChecker(max_evidence_length=100000)
    veracity_labels, _ = fc.get_veracity_labels(topic=topic, atomic_facts=fact_list)
    print("Veracity Labels:", veracity_labels)
