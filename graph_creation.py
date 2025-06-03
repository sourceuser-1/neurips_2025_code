

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from entailment import get_entailment_probs, get_entailment_probs_batch

class TextGraphGenerator:
    def __init__(self, embedding_model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the TextGraphGenerator.
        
        Args:
            embedding_model_name: Name of the Sentence Transformer model for embeddings.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        # self.embedding_model = SentenceTransformer("sentence-transformers/stsb-roberta-base-v2")
        self._nli_tokenizer = None
        self._nli_model = None

    def _load_nli_model(self):
        """Lazy-load the MNLI model and tokenizer using transformers."""
        if self._nli_tokenizer is None or self._nli_model is None:
            self._nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            self._nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    
    def _nli_predict_transformers(self, premises: List[str], hypotheses: List[str]) -> List[str]:
        """
        Use transformers to predict the MNLI label for each (premise, hypothesis) pair.
        
        Args:
            premises: List of premise texts.
            hypotheses: List of hypothesis texts.
        
        Returns:
            List of predicted labels (in lowercase), e.g. "contradiction", "neutral", or "entailment".
        """
        self._load_nli_model()
        inputs = self._nli_tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self._nli_model(**inputs)
        logits = outputs.logits  # shape: (batch_size, 3)
        preds = torch.argmax(logits, dim=1)
        labels = [self._nli_model.config.id2label[p.item()].lower() for p in preds]
        return labels

    def create_graph(self, texts: List[str], method: str = 'soft', batch_size: int = 32, threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a graph from a list of texts using the specified method.
        
        Args:
            texts: List of text strings.
            method: Either 'soft' (using cosine similarity), 'nli' (using NLI-based contradiction),
                    or 'nli_hybrid' (using entailment probability threshold and cosine distance).
            batch_size: Batch size for processing (used in 'nli' and 'nli_hybrid' methods).
            threshold: Threshold for entailment probability in 'nli_hybrid' method.
        
        Returns:
            A tuple (adjacency_matrix, node_embeddings)
        """
        N = len(texts)
        embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
        
        if method == 'soft':
            # Using cosine distances between embeddings
            adjacency_matrix = cosine_distances(embeddings).astype(np.float32)
            # normalize cost matrix to [-1,1] range from [0,1]
            adjacency_matrix = 2*adjacency_matrix - 1
        elif method == 'nli':
            adjacency_matrix = np.zeros((N, N), dtype=np.int8)
            pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start:start+batch_size]
                premises = [texts[i] for i, j in batch_pairs]
                hypotheses = [texts[j] for i, j in batch_pairs]
                preds = self._nli_predict_transformers(premises, hypotheses)
                for idx, (i, j) in enumerate(batch_pairs):
                    if preds[idx].startswith("contradiction"):
                        adjacency_matrix[i, j] = 1
                        adjacency_matrix[j, i] = 1  # ensure symmetry
        elif method == 'nli_hybrid':
            adjacency_matrix = np.zeros((N, N), dtype=np.float32)
            pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
            # Process pairs in batches using get_entailment_probs_batch
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start:start+batch_size]
                premises = [texts[i] for i, j in batch_pairs]
                hypotheses = [texts[j] for i, j in batch_pairs]
                # Call the batched entailment function
                probs_batch = get_entailment_probs_batch(premises, hypotheses, batch_size=len(premises))
                for idx, (i, j) in enumerate(batch_pairs):
                    # Check if the entailment probability (first element) exceeds threshold
                    cosine_sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0] #(cosine_distances(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0] + 1)/2
                    if (probs_batch[idx][0]  > probs_batch[idx][1] and probs_batch[idx][0]  > probs_batch[idx][2]) or cosine_sim > threshold:
                        d = max(cosine_sim, probs_batch[idx][0]) #cosine_distances(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0]
                        adjacency_matrix[i, j] = d
                        adjacency_matrix[j, i] = d
            adjacency_matrix = (1 - adjacency_matrix) # converting to cost matrix and normalize to (-1, 1) range
            # normalize cost matrix to [-1,1] range from [0,1]
            adjacency_matrix = 2*adjacency_matrix - 1
        else:
            raise ValueError("Invalid method. Use 'soft', 'nli', or 'nli_hybrid'.")
        
        
        
        return adjacency_matrix, embeddings

    @staticmethod
    def print_graph_info(texts: List[str], adjacency_matrix: np.ndarray):
        """Helper function to print graph information."""
        print(f"Generated graph with {len(texts)} nodes")
        print("\nNode texts:")
        for i, text in enumerate(texts):
            print(f"{i}: {text[:50]}...")
        print("\nAdjacency matrix:")
        print(adjacency_matrix)

# Example usage
if __name__ == "__main__":
    texts = [
        "Kang Ji-hwan is a South Korean actor.",
        "Kang Ji-hwan is a singer.",
        "Kang Ji-hwan was born on January 6, 1977.",
        "Kang Ji-hwan was born in Seoul, South Korea.",
        "Kang Ji-hwan began his career in the entertainment industry in the late 1990s.",
    ]
    generator = TextGraphGenerator()
    
    soft_adj, soft_emb = generator.create_graph(texts, method='soft')
    print("Soft Graph:")
    generator.print_graph_info(texts, soft_adj)
    print("\nSoft Embedding shape:", soft_emb.shape)
    
    nli_adj, nli_emb = generator.create_graph(texts, method='nli', batch_size=8)
    print("\nNLI Graph:")
    generator.print_graph_info(texts, nli_adj)
    print("\nNLI Embedding shape:", nli_emb.shape)
    
    nli_hybrid_adj, nli_hybrid_emb = generator.create_graph(texts, method='nli_hybrid', batch_size=8, threshold=0.7)
    print("\nNLI Hybrid Graph:")
    generator.print_graph_info(texts, nli_hybrid_adj)
    print("\nNLI Hybrid Embedding shape:", nli_hybrid_emb.shape)
