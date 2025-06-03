from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import os
import time
import json
from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_TOKEN"] = "----------"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class LUQ:
    def __init__(
        self,
        nli_model: str,
        method: str,
        cuda_devices: str = "0",
        gpu_memory_utilization: float = 0.9,
    ):
        """
        nli_model: str - the identifier of the model to do NLI (e.g., "microsoft/deberta-v3-large")
        method: str - the method to use for the task, can be "binary", "multiclass", or "luq"
        """
        print(f"Loading NLI model: {nli_model}")
        if method not in ["binary", "multiclass", "luq"]:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

        # Set device (we simply use the first device provided)
        device_ids = cuda_devices.split(",")
        device_str = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Load the tokenizer and model using Transformers
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.model.to(self.device)
        self.model.eval()

        # Define label-to-score mappings for binary and multiclass modes
        if self.method == "binary":
            self.text_mapping = {'yes': 1, 'no': 0, 'n/a': 0.5}
        elif self.method == "multiclass":
            self.text_mapping = {'supported': 1, 'refuted': 0, 'not mentioned': -1, 'n/a': 0.5}

        # For LUQ, we will use the model's entailment probability (index 2 in MNLI order)
        # The prompt_template is kept here for clarity/documentation.
        if self.method == "luq":
            self.prompt_template = "Premise: {premise}\nHypothesis: {hypothesis}"
        else:
            self.prompt_template = "Premise: {premise}\nHypothesis: {hypothesis}"

        self.not_defined_text: Set[str] = set()

    def predict(self, sentences: List[str], sampled_passages: List[str]):
        """
        For 'binary' or 'multiclass':
          For each sentence and passage pair, run the NLI model and convert the label to a score.
        For 'luq':
          For each claim (sentence) and paragraph pair, compute S(claim, paragraph) = P(entailment).
          Then, for each paragraph, average over all claims and finally compute:
               uncertainty = 1 - (mean over all paragraphs of these averages).
        """
        if self.method in ["binary", "multiclass"]:
            return self._predict_binary_multiclass(sentences, sampled_passages)
        elif self.method == "luq":
            return self._predict_luq(sentences, sampled_passages)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _predict_binary_multiclass(self, sentences: List[str], sampled_passages: List[str]):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        raw_scores = np.zeros((num_sentences, num_samples))
        for i, sentence in enumerate(sentences):
            for j, passage in enumerate(sampled_passages):
                # We treat the passage as the premise and the sentence as the hypothesis.
                label_str = self._nli_inference_label(passage, sentence)
                score = self.text_postprocessing(label_str)
                raw_scores[i, j] = score
        scores_filtered = np.ma.masked_equal(raw_scores, -1)
        scores_per_sentence = scores_filtered.mean(axis=-1)
        scores_per_sentence = np.where(scores_per_sentence.mask, 0, scores_per_sentence)
        return scores_per_sentence, raw_scores

    def _predict_luq(self, claims: List[str], paragraphs: List[str]):
        num_claims = len(claims)
        num_paragraphs = len(paragraphs)
        S_matrix = np.zeros((num_claims, num_paragraphs))
        for i, claim in enumerate(claims):
            for j, paragraph in enumerate(paragraphs):
                # For LUQ, S(claim, paragraph) is the entailment probability.
                entail_prob = self._nli_inference_entail_prob(paragraph, claim)
                S_matrix[i, j] = entail_prob
        # For each paragraph, average the entailment probabilities across all claims.
        paragraph_means = np.mean(S_matrix, axis=0)
        overall_mean = np.mean(paragraph_means)
        uncertainty = 1.0 - overall_mean
        uncty_dict =  {
            "S_matrix": S_matrix,
            "paragraph_means": paragraph_means,
            "overall_mean": overall_mean,
            "uncertainty": uncertainty
        }
        return uncty_dict["uncertainty"]

    def _nli_inference_label(self, premise: str, hypothesis: str) -> str:
        """
        Run the NLI model on a (premise, hypothesis) pair.
        The model's output logits follow the MNLI ordering:
          0: contradiction, 1: neutral, 2: entailment.
        For binary: return "yes" if P(entailment) >= P(contradiction), else "no".
        For multiclass: return 'supported' for entailment, 'refuted' for contradiction, 'not mentioned' for neutral.
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512, padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: [1, 3]
        probs = F.softmax(logits, dim=-1)[0]
        contr_prob = probs[0].item()
        neut_prob = probs[1].item()
        ent_prob = probs[2].item()
        if self.method == "binary":
            return "yes" if ent_prob >= contr_prob else "no"
        elif self.method == "multiclass":
            label_id = torch.argmax(probs).item()
            if label_id == 0:
                return "refuted"
            elif label_id == 1:
                return "not mentioned"
            elif label_id == 2:
                return "supported"
            else:
                return "n/a"
        return "n/a"

    def _nli_inference_entail_prob(self, premise: str, hypothesis: str) -> float:
        """
        For LUQ mode: return the entailment probability (index 2) for the (premise, hypothesis) pair.
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512, padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # shape: [1, 3]
        probs = F.softmax(logits, dim=-1)[0]
        # print(logits, probs)
        return probs[0].item()

    def text_postprocessing(self, text: str) -> float:
        """
        Convert a label string into a numeric score.
        For binary: 'yes' -> 1, 'no' -> 0, 'n/a' -> 0.5.
        For multiclass: 'supported' -> 1, 'refuted' -> 0, 'not mentioned' -> -1, 'n/a' -> 0.5.
        """
        text = text.lower().strip()
        if self.method == "binary":
            if text.startswith("yes"):
                return self.text_mapping["yes"]
            elif text.startswith("no"):
                return self.text_mapping["no"]
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined for binary")
                    self.not_defined_text.add(text)
                return self.text_mapping["n/a"]
        elif self.method == "multiclass":
            if text.startswith("support"):
                return self.text_mapping["supported"]
            elif text.startswith("refut"):
                return self.text_mapping["refuted"]
            elif text.startswith("not"):
                return self.text_mapping["not mentioned"]
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined for multiclass")
                    self.not_defined_text.add(text)
                return self.text_mapping["n/a"]
        else:
            return 0.5

if __name__ == "__main__":
    os.environ["HF_HOME"] = "huggingface"
    os.environ["HF_HUB_CACHE"] = "huggingface"
    os.environ["HF_TOKEN"] = "------"
    # Ensure online mode is enabled so that the model can be downloaded
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # Example usage for LUQ method using a DeBERTa-based NLI model.
    # You can use a fine-tuned version (e.g., "potsawee/deberta-v3-large-mnli") or another appropriate identifier.
    nli_model_name = "potsawee/deberta-v3-large-mnli" # "microsoft/deberta-v3-large"  # update this if you have a specific fine-tuned NLI model
    luq_obj = LUQ(
        nli_model=nli_model_name,
        method="luq",
        cuda_devices="0",
        gpu_memory_utilization=0.5
    )

    # Example claims and paragraphs:
    claims = [
        "Michael Alan Weiner was born on March 31, 1942.",
        "He is an American radio host.",
        "He has a PhD from MIT."
    ]
    paragraphs = [
        "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is known as Michael Savage.",
        "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times.",
        "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from UC Berkeley."
    ]

    results = luq_obj.predict(claims, paragraphs)
    print("LUQ results:", results)
