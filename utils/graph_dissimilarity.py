# utils/compute_cost.py
import numpy as np
from ot import fused_gromov_wasserstein
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import eigsh


def gw_alignment_cost_entailment(A_ref, A, NE_ref, NE, raw_scores_matrix_slice, prob_ent_matrix, alpha=0.5, threshold=0.7):
    """
    Compute fused Gromov-Wasserstein alignment cost between two graphs with entailment information.
    
    Args:
        A_ref: Reference adjacency matrix (n_ref x n_ref)
        A: Target adjacency matrix (n x n)
        NE_ref: Reference node embeddings (n_ref x d)
        NE: Target node embeddings (n x d)
        prob_ent_matrix: A J x K x 3 matrix containing entailment probabilities,
                         where J is the number of claims in the reference paragraph and 
                         K is the number of claims in the target paragraph.
        alpha: Weight between structural/semantic costs [0,1]
        threshold: Decision threshold for entailment probability and cosine similarity
        
    Returns:
        P: Optimal transport plan (n_ref x n)
        total_cost: Combined alignment cost (fused GW cost)
        struct_cost: Pure structural cost component
        semantic_cost: Pure semantic cost component
    """

    raw_scores_matrix_slice = raw_scores_matrix_slice[1:].reshape(-1, 1)
    # raw_scores_matrix_slice = raw_scores_matrix_slice.mean(1)[1:].reshape(-1, 1)

    # 0. Convert adjacency matrices to the [0, 1] range.
    A_ref = (A_ref + 1) / 2
    A = (A + 1) / 2

    # 1. Convert adjacency matrices to "distance" matrices.
    # Here, we are directly using the scaled adjacency matrices.
    C1 = A_ref
    C2 = A

    C_threshold = 0.0
    #### C_threshold = 0.0#0.7 # 0.4
    condition_1 = (C1 >= C_threshold)
    condition_2 = (C2 >= C_threshold)
    C1 = np.where(condition_1, C1, 1.0)
    C2 = np.where(condition_2, C2, 1.0)

    # C1 = (C1 >= 0.4).astype(float) # this works better for qwen -bios, wh; 
    # C2 = (C2 >= 0.4).astype(float)
    ## C1 = (C1 >= 0.1).astype(float) # this works better for qwen -bios, wh; 
    ## C2 = (C2 >= 0.1).astype(float)


    # 2. Compute semantic cost matrix.
    # First, compute the cosine similarity matrix between node embeddings.
    cosine_sim_matrix = cosine_similarity(NE_ref, NE)  
    M = np.where(1, 1 - cosine_sim_matrix*raw_scores_matrix_slice, 1.0)
    M = (M >= 0.6).astype(float)

    # 3. Define uniform node distributions.
    n_ref, n = A_ref.shape[0], A.shape[0]
    p = np.ones(n_ref) / n_ref  # Reference distribution
    q = np.ones(n) / n          # Target distribution

    # 4. Compute fused Gromov-Wasserstein alignment.
    P, log = fused_gromov_wasserstein(
        M, C1, C2, p, q, 
        loss_fun='square_loss',
        alpha=alpha,
        verbose=False,
        log=True
    )

    # 5. Decompose costs.
    # Compute the structural cost component.
    struct_cost = 0.0
    n_rows, n_cols = P.shape
    for i in range(n_rows):
        for j in range(n_rows):
            for k in range(n_cols):
                for l in range(n_cols):
                    struct_cost += ((C1[i, j] - C2[k, l]) ** 2) * P[i, k] * P[j, l]
    
    # Compute the semantic cost component.
    semantic_cost = np.sum(P * M)
    # semantic_cost = np.sum(P * (1 - cosine_sim_matrix*raw_scores_matrix_slice))

    return P, log['fgw_dist'], struct_cost, semantic_cost

def batch_alignment_cost_entailment(G_ref, Gs, raw_scores_matrix, batch_entailment_matrices, alpha=0.5, threshold=0.7):
    """
    Batch compute costs against a reference graph using entailment-based alignment.
    
    Args:
        G_ref: Tuple of (A_ref, NE_ref)
        Gs: List of tuples (A, NE) representing target graphs.
        batch_entailment_matrices: List of entailment matrices (prob_ent_matrix) for each target graph.
        alpha: Cost weighting parameter.
        threshold: Decision threshold for entailment probability and cosine similarity.
        
    Returns:
        costs: Array of total costs for each graph.
        plans: List of optimal transport matrices.
        struct_costs: Array of structural cost components.
        semantic_costs: Array of semantic cost components.
    """
    (A_ref, NE_ref) = G_ref
    costs, plans, struct_costs, semantic_costs = [], [], [], []
    
    reference_index = 0
    for (A, NE), prob_ent_matrix in zip(Gs, batch_entailment_matrices):
        if prob_ent_matrix.ndim == 3:
            #print(raw_scores_matrix.shape, NE_ref.shape, NE.shape)
            raw_scores_slice = raw_scores_matrix[:,reference_index].reshape(-1, 1)
            if prob_ent_matrix.shape[1] == NE.shape[0]:
                P, total_cost, struct_cost, semantic_cost = gw_alignment_cost_entailment(
                    A_ref, A, NE_ref, NE, raw_scores_matrix[:,reference_index], prob_ent_matrix, alpha=alpha, threshold=threshold
                    # A_ref, A, NE_ref, NE, raw_scores_matrix, prob_ent_matrix, alpha=alpha, threshold=threshold
                )
                costs.append(total_cost)
                plans.append(P)
                struct_costs.append(struct_cost)
                semantic_costs.append(semantic_cost)

        reference_index += 1
        # if reference_index >= 1:
        #     break
        
    return np.array(costs), plans, np.array(struct_costs), np.array(semantic_costs)

def batch_alignment_cost(G_ref, Gs, alpha=0.5):
    pass


def compute_heat_kernel(A, t=0.9):
    """
    Compute the heat kernel matrix H(t) = exp(-tL) from a 1-adjacency matrix A.
    
    Parameters:
    A : numpy.ndarray
        1-adjacency matrix of the graph with shape (N, N)
    t : float
        Diffusion time (t > 0)
        
    Returns:
    H : numpy.ndarray
        Heat kernel matrix with shape (N, N)
    """
    # A: (N x N) adjacency matrix, where N = number of nodes.
    N = A.shape[0]  # N is a scalar (number of nodes)
    
    # Compute the degree matrix D.
    # Sum over each row gives a vector of shape (N,).
    # Then, create a diagonal matrix from this vector.
    D = np.diag(np.sum(A, axis=1))  # D: (N x N)
    
    # Compute the unnormalized graph Laplacian: L = D - A.
    L = D - A  # L: (N x N)
    
    # Compute the eigen-decomposition of L: L = U * Lambda * U^T.
    # Since L is symmetric, np.linalg.eigh returns:
    # eigenvalues: (N,) vector and U: (N x N) matrix.
    eigenvalues, U = np.linalg.eigh(L)  # U: (N x N), eigenvalues: (N,)
    
    # Form a diagonal matrix of the eigenvalues.
    Lambda = np.diag(eigenvalues)  # Lambda: (N x N)
    
    # Compute the exponential of the diagonal matrix: exp(-t*Lambda).
    # This is done elementwise on the diagonal.
    exp_neg_t_Lambda = np.diag(np.exp(-t * eigenvalues))  # exp_neg_t_Lambda: (N x N)
    
    # Reconstruct the heat kernel matrix using the spectral decomposition:
    # H(t) = U * exp(-t*Lambda) * U^T.
    H = U @ exp_neg_t_Lambda @ U.T  # H: (N x N)
    
    return H

def spectral_embedding(adjacency, n_components=2):
    """
    Computes the spectral embedding matrix of a graph given its adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray
        The (n x n) adjacency matrix of the graph.
    n_components : int, optional
        The number of dimensions for the embedding (default is 2).
    
    Returns
    -------
    embedding : np.ndarray
        A matrix of shape (n, n_components) containing the spectral embedding
        coordinates for each node.
    
    Notes
    -----
    This function computes the normalized graph Laplacian:
    
        L = I - D^(-1/2) A D^(-1/2)
    
    where D is the degree matrix. The eigenvectors corresponding to the smallest 
    non-zero eigenvalues (ignoring the trivial eigenvector) are returned as the 
    spectral embedding.
    """
    n_nodes = adjacency.shape[0]
    # Compute the degree vector and the inverse square root of degrees
    degrees = np.sum(adjacency, axis=1)
    # Avoid division by zero by adding a small constant
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    
    # Compute the normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    I = np.eye(n_nodes)
    L = I - D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    # Compute the smallest (n_components+1) eigenvectors.
    # For a connected graph, the smallest eigenvalue is 0 and corresponds to the trivial eigenvector.
    # eigenvalues, eigenvectors = eigsh(L, k=n_components + 1, which='SM')
    eigenvalues, eigenvectors = eigsh(L, k=n_components+1, which='SM', ncv=max(2*(n_components+1),20))

    
    # Sort eigenvalues (and corresponding eigenvectors) in ascending order.
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Drop the first eigenvector (trivial constant vector) and return the next n_components.
    embedding = eigenvectors[:, 1:n_components+1]
    return embedding

