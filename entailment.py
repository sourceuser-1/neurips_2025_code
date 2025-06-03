

# utils/entailment.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"#"6,7"

MODEL_NAME = "roberta-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Optionally, move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_entailment_probs(claim_x, claim_y):
    """Process a single pair of claims."""
    inputs = tokenizer.encode_plus(claim_x, claim_y, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)[0]
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    # Reorder [contradiction, neutral, entailment] to [entailment, neutral, contradiction]
    return [probs[2], probs[1], probs[0]]

def get_entailment_probs_batch(premises, hypotheses, batch_size=32):
    """
    Process batches of (premise, hypothesis) pairs.
    Returns a list of probability vectors.
    """
    all_probs = []
    num_pairs = len(premises)
    for i in range(0, num_pairs, batch_size):
        batch_premises = premises[i:i+batch_size]
        batch_hypotheses = hypotheses[i:i+batch_size]
        inputs = tokenizer(batch_premises, batch_hypotheses, return_tensors="pt", 
                           truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs)[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Reorder probabilities for each pair: [contradiction, neutral, entailment] -> [entailment, neutral, contradiction]
        for prob in probs:
            all_probs.append([prob[2], prob[1], prob[0]])
    return all_probs

if __name__ == "__main__":
    premise = "A soccer game with multiple males playing."
    hypothesis = "Some men are playing a sport."
    print("Single pair:", get_entailment_probs(premise, hypothesis))
    
    # Test batch function
    premises = [premise] * 5
    hypotheses = [hypothesis] * 5
    print("Batch processing:", get_entailment_probs_batch(premises, hypotheses))
