"""
GraphWave embedding implementation
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import networkx as nx
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from embeddings.base_embedder import BaseEmbedder
from embeddings.graphwave.characteristic_functions import charac_function
from embeddings.graphwave.utils.graph_tools import laplacian

# Constants
TAUS = [1, 10, 25, 50]
ORDER = 30
PROC = 'approximate'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 2


def compute_cheb_coeff_basis(scale, order):
    """Vectorized Chebyshev coefficient computation."""
    # Compute Chebyshev nodes
    xx = np.cos((2 * np.arange(1, order + 1) - 1) * np.pi / (2 * order))
    
    # Compute basis functions using recurrence (vectorized)
    basis = np.zeros((order + 1, order))
    basis[0] = 1.0  # T_0
    basis[1] = xx   # T_1
    
    for k in range(2, order + 1):
        basis[k] = 2 * xx * basis[k-1] - basis[k-2]
    
    # Compute coefficients
    f = np.exp(-scale * (xx + 1))
    coeffs = 2.0 / order * np.sum(f[np.newaxis, :] * basis, axis=1)
    coeffs[0] = coeffs[0] / 2
    
    return coeffs.astype(np.float32)


def _process_tau_worker(args):
    """Worker function for parallel tau processing."""
    tau_idx, tau, L_hat, n_nodes, order, time_pnts = args
    
    # Compute Chebyshev coefficients
    coeffs = compute_cheb_coeff_basis(tau, order)
    
    # Batched Chebyshev recurrence (extract diagonal only)
    T0 = sp.eye(n_nodes, format="csr", dtype=np.float32)
    T1 = L_hat @ T0
    
    phi_diag = coeffs[0] * np.ones(n_nodes, dtype=np.float32)
    phi_diag = phi_diag + coeffs[1] * np.asarray(T1.diagonal(), dtype=np.float32)
    
    for k in range(2, order + 1):
        T2 = 2.0 * (L_hat @ T1) - T0
        phi_diag = phi_diag + coeffs[k] * np.asarray(T2.diagonal(), dtype=np.float32)
        T0, T1 = T1, T2
    
    # Characteristic function
    temp = phi_diag[:, None]
    chi_tau = charac_function(time_pnts, temp)
    
    return tau_idx, chi_tau.T


def graphwave_alg_fast(
    graph,
    time_pnts,
    taus="auto",
    verbose=False,
    approximate_lambda=True,
    order=30,
    nb_filters=2,
):
    """
    High-memory / high-speed GraphWave implementation.

    Returns
    -------
    chi : np.ndarray, shape (n_nodes, num_time_points * 2 * n_taus)
    taus : list
    """
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    num_time_points = len(time_pnts)

    # ------------------------------------------------------------------
    # Automatic tau selection (unchanged logic)
    # ------------------------------------------------------------------
    if taus == "auto":
        if approximate_lambda:
            l1 = 1.0 / n_nodes
        else:
            lap_mat = laplacian(nx.adjacency_matrix(graph))
            try:
                l1 = np.sort(
                    sp.linalg.eigsh(
                        lap_mat, 2, which="SM", return_eigenvectors=False
                    )
                )[1]
            except Exception:
                l1 = np.sort(
                    sp.linalg.eigsh(
                        lap_mat, 5, which="SM", return_eigenvectors=False
                    )
                )[1]

        smax = -np.log(0.80) * np.sqrt(0.5 / l1)
        smin = -np.log(0.95) * np.sqrt(0.5 / l1)
        taus = np.linspace(smin, smax, nb_filters)

    taus = list(taus)
    n_taus = len(taus)

    # ------------------------------------------------------------------
    # Allocate output
    # ------------------------------------------------------------------
    chi = np.zeros(
        (n_nodes, num_time_points * 2 * n_taus),
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Precompute normalized Laplacian (float32, CSR)
    # ------------------------------------------------------------------
    lap_mat = laplacian(nx.adjacency_matrix(graph))
    L_hat = (lap_mat - sp.eye(n_nodes)).astype(np.float32).tocsr()

    # ------------------------------------------------------------------
    # Main loop over taus (parallel processing)
    # ------------------------------------------------------------------
    # Prepare worker arguments
    worker_args = [
        (tau_idx, tau, L_hat, n_nodes, order, time_pnts)
        for tau_idx, tau in enumerate(taus)
    ]
    
    # Use multiprocessing for parallel tau computation
    n_workers = min(len(taus), cpu_count())
    if n_workers > 1 and len(taus) > 1:
        if verbose:
            print(f"[GraphWave] Using {n_workers} processes for parallel tau computation")
        
        with Pool(n_workers) as pool:
            results = pool.map(_process_tau_worker, worker_args)
    else:
        # Fallback to sequential processing for small cases
        results = [_process_tau_worker(args) for args in worker_args]
    
    # Fill chi matrix from results (maintain correct order)
    for tau_idx, chi_tau_T in results:
        idx_start = tau_idx * 2 * num_time_points
        idx_end = idx_start + 2 * num_time_points
        chi[:, idx_start:idx_end] = chi_tau_T
        if verbose and len(taus) <= 4:
            print(f"[GraphWave] Completed tau {tau_idx + 1}/{n_taus}")

    if verbose:
        print("GraphWave embedding completed (parallel version).")

    return chi, taus


class GraphWaveEmbedder(BaseEmbedder):
    """
    GraphWave implementation for graph embedding using spectral methods.
    """
    
    def __init__(self, embedding_dim=32, order=ORDER, proc=PROC, nb_filters=NB_FILTERS, random_state=42):
        """
        Initialize GraphWave embedder.
        
        Args:
            embedding_dim (int): Dimension of embedding vectors
            order (int): Order of the polynomial approximation
            proc (str): Procedure to compute signatures ('approximate' or 'exact')
            nb_filters (int): Number of filters (taus) when using automatic scale selection
            random_state (int): Random seed
        """
        super().__init__(embedding_dim, random_state)
        self.order = order
        self.proc = proc
        self.nb_filters = nb_filters
        self.scaler = None
        self.pca = None
    
    def generate_embeddings(self, graph):
        """
        Generate GraphWave embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        # Calculate time points to match embedding_dim exactly
        # chi dimensions: (time_points × 2 × nb_filters) = embedding_dim
        # Therefore: time_points = embedding_dim / (2 * nb_filters)
        num_time_points = max(1, int(np.ceil(self.embedding_dim / (2 * self.nb_filters))))
        time_points = np.linspace(-math.pi, math.pi, num_time_points)
        
        chi, taus = graphwave_alg_fast(
            graph,
            time_points,
            taus="auto",
            verbose=True,
            order=self.order,
            nb_filters=self.nb_filters,
        )

        print(f"GraphWave: Generated chi with shape {chi.shape}")
        # Standardize features (chi shape: [features, nodes])
        self.scaler = StandardScaler()
        chi_scaled = self.scaler.fit_transform(chi)  # Transpose to [nodes, features]
        
        # Adjust dimensionality if needed
        actual_dim = chi_scaled.shape[1]
        if actual_dim > self.embedding_dim:
            chi_final = chi_scaled[:, :self.embedding_dim]
        elif actual_dim < self.embedding_dim:
            padding = np.zeros((chi_scaled.shape[0], self.embedding_dim - actual_dim))
            chi_final = np.hstack([chi_scaled, padding])
        else:
            chi_final = chi_scaled
        
        # Store embeddings in a dictionary
        nodes_list = list(graph.nodes())
        self.embeddings = {}
        for i, node in enumerate(nodes_list):
            self.embeddings[node] = chi_final[i]
            if (i + 1) % 10000 == 0:
                print(f"[GraphWave] Embedded {i + 1} nodes")
        
        print(f"[GraphWave] Finished embedding all {len(nodes_list)} nodes")
        
        return self.embeddings