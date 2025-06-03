import os
import time
import wikipedia
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import GPT2Tokenizer
# from openai import OpenAI   # your OpenAI client code

from openai import OpenAI
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import os
import time
import wikipedia
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import GPT2Tokenizer
from vllm import LLM, SamplingParams

# Define the instruction template.
INSTUCT_FACTCHECK_RAG_ALL = """Your task is to fact-check the following statements starting with ###.
These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is factually correct; 'NS' means the statement is factually wrong.
Pay close attention to numbers, dates, and other details. You can refer to the provided knowledge source to verify the statements.

To assess a statement, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Lebron James is a basketball player. Analysis: Lebron James is an American professional basketball player, so this is correct. $S$
### Obama was the 46th president of the United States. Analysis: Barack Obama served as the 44th president, so this is incorrect. $NS$
### Jackie Chan was born on April 7, 1955. Analysis: Jackie Chan was born on April 7, 1954, so this is incorrect. $NS$

The statements to evaluate are as follows:
{atomic_facts_string}

The knowledge source for this fact-check is as follows:
{retrieved_evidence}
"""

class FactChecker(object):
    def __init__(self, max_evidence_length):
        super().__init__()
        self.llm_token_count = 0
        self.openai_cost = 0
        self.max_evidence_length = max_evidence_length
        # Use GPT2Tokenizer for truncation purposes.
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Initialize vLLM with the Qwen2.5-32B-Instruct-GGUF model.
        self.llm = LLM(
            model="Qwen/Qwen2.5-32B-Instruct",
            tensor_parallel_size=2,         # Adjust based on your GPU setup.
            trust_remote_code=True
        )
        # Set sampling parameters (adjust max_tokens if needed)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,  # Adjust this if your model supports a longer context.
            stop=["\n"]
        )

    def truncate_text(self, text):
        """Truncate the text to a maximum token length."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_evidence_length:
            tokens = tokens[:self.max_evidence_length]
            text = self.tokenizer.decode(tokens)
        return text

    def build_wikipedia_evidence(self, topic):
        """
        Fetch the entire Wikipedia page for the given topic.
        """
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(topic)
            return page.content
        except Exception as e:
            print(f"Error retrieving Wikipedia page for topic '{topic}': {e}")
            return ""

    @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(20))
    def get_completion(self, user_prompt: str = ""):
        """
        Get a completion from the LLM using vLLM.
        """
        outputs = self.llm.generate([user_prompt], self.sampling_params)
        print(outputs)
        # Assume the first result is our answer.
        response = outputs[0].outputs[0].text.strip()
        return response

    def get_prompt_with_wiki(self, topic, fact_list):
        """
        Build a prompt that contains all atomic facts and the full Wikipedia article as evidence.
        """
        atomic_facts_string = "### " + "\n### ".join(fact_list) + "\n\n"
        evidence = self.build_wikipedia_evidence(topic)
        # Truncate only if necessary; here we use the full text as much as possible.
        retrieved_evidence = self.truncate_text(evidence)
        instruct = INSTUCT_FACTCHECK_RAG_ALL.format(
            atomic_facts_string=atomic_facts_string,
            retrieved_evidence=retrieved_evidence
        )
        return instruct

    def get_veracity_labels(self, topic="", atomic_facts=[], evidence_type="wiki"):
        """
        Get veracity labels for the given atomic facts using the Wikipedia page as evidence.
        """
        for attempt in range(3):
            try:
                user_prompt = self.get_prompt_with_wiki(topic, atomic_facts)
                completion = self.get_completion(user_prompt)
                print(completion)
                # Split the output on the "### " marker.
                atomic_responses = completion.split("\n### ")
                atomic_responses = [x for x in atomic_responses if x]
                # Extract the label enclosed in dollar signs, e.g. $S$ or $NS$
                gpt_labels = [response.split("$")[1] for response in atomic_responses]
                assert len(atomic_facts) == len(gpt_labels)
                return gpt_labels, completion
            except Exception as e:
                print(e)
                print(f"Retrying ({attempt + 1}/3), wait for {2 ** (attempt + 1)} sec...")
                time.sleep(2 ** (attempt + 1))
        return [], ""

if __name__ == "__main__":
    # You can set your OPENAI_API_KEY here if needed; not used in this version.
    os.environ["OPENAI_API_KEY"] = "dummy"
    
    # Example topic and atomic facts
    topic = "Kang Ji-hwan"
    fact_list = ["Kang Ji-hwan is a South Korean actor.", "Kang Ji-hwan is a South Korean singer.", "Kang Ji-hwan was born on August 6, 1977.", "Kang Ji-hwan was born in Seoul, South Korea.", "Kang Ji-hwan began his career as a member of the boy band \"Sechs Kies\".", "Kang Ji-hwan began his career in the late 1990s.", "The boy band \"Sechs Kies\" disbanded in 2000.", "Kang Ji-hwan transitioned to acting after his music career.", "Kang Ji-hwan made his acting debut in the 2001 film \"Break Out\".", "Kang Ji-hwan gained recognition for his role in \"Lovers in Paris\" in 2004.", "Kang Ji-hwan gained recognition for his role in \"Be Strong Geum-soon\" in 2005.", "Kang Ji-hwan has appeared in \"The World That They Live In\" in 2008.", "Kang Ji-hwan has appeared in \"The Fugitive Plan B\" in 2010.", "Kang Ji-hwan has appeared in \"Myung-wol the Spy\" in 2013.", "Kang Ji-hwan has appeared in \"My ID is Gangnam Beauty\" in 2018.", "Kang Ji-hwan received the Best Actor award at the 2005 SBS Drama Awards.", "Kang Ji-hwan received the award for his role in \"Be Strong Geum-soon\".", "Kang Ji-hwan is known for his versatility as an actor.", "Kang Ji-hwan has played a wide range of characters.", "Kang Ji-hwan's characters span various genres.", "Kang Ji-hwan is also a talented singer.", "Kang Ji-hwan has released several solo albums.", "Kang Ji-hwan is married to actress Park Ha-sun.", "Kang Ji-hwan has two children with Park Ha-sun."]
    
    fc = FactChecker(max_evidence_length=100000)
    veracity_labels, complete_output = fc.get_veracity_labels(
        topic=topic, 
        atomic_facts=fact_list, 
        evidence_type="wiki"
    )
    
    print("Veracity Labels:", veracity_labels)
    print("Complete LLM Output:", complete_output)
