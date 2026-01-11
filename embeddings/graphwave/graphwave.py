"""
GraphWave embedding implementation
"""
import copy
import time
import math
import numpy as np
import scipy as sc
import scipy.sparse as sp
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from embeddings.base_embedder import BaseEmbedder
from embeddings.graphwave.characteristic_functions import charac_function, charac_function_multiscale
from embeddings.graphwave.utils.graph_tools import laplacian

# Constants
TAUS = [1, 10, 25, 50]
ORDER = 30
PROC = 'approximate'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 2


def compute_cheb_coeff(scale, order):
    coeffs = [(-scale)**k * 1.0 / math.factorial(k) for k in range(order + 1)]
    return coeffs


def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1-2):
        basis.append(2* np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def chebyshev_phi(L_hat, coeffs, node_idx, order):
    """
    Compute Chebyshev approximation for one node using CSR matrix dot product.
    
    Args:
        L_hat: CSR Laplacian matrix (n_nodes x n_nodes)
        coeffs: Chebyshev coefficients (list of length order+1)
        node_idx: index of the node to compute
        order: polynomial order
    
    Returns:
        phi_u: np.ndarray of shape (n_nodes,)
    """
    n_nodes = L_hat.shape[0]

    T_km2 = np.zeros(n_nodes)
    T_km2[node_idx] = 1.0
    T_km1 = L_hat.dot(T_km2)

    phi_u = coeffs[0] * T_km2 + coeffs[1] * T_km1

    for k in range(2, order + 1):
        T_k = 2 * L_hat.dot(T_km1) - T_km2
        phi_u += coeffs[k] * T_k
        T_km2, T_km1 = T_km1, T_k

    return phi_u

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
    # Main loop over taus (no node loop!)
    # ------------------------------------------------------------------
    for tau_idx, tau in enumerate(taus):
        if verbose:
            print(f"[GraphWave] Processing tau {tau_idx + 1}/{n_taus}: {tau:.5f}")

        # Chebyshev coefficients (float32)
        coeffs = np.asarray(
            compute_cheb_coeff_basis(tau, order),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Batched Chebyshev recurrence
        # --------------------------------------------------------------
        T0 = sp.eye(n_nodes, format="csr", dtype=np.float32)
        T1 = L_hat @ T0

        # We only need the diagonal
        phi_diag = coeffs[0] * np.ones(n_nodes, dtype=np.float32)
        phi_diag += coeffs[1] * T1.diagonal()

        for k in range(2, order + 1):
            T2 = 2.0 * (L_hat @ T1) - T0
            phi_diag += coeffs[k] * T2.diagonal()
            T0, T1 = T1, T2

        # --------------------------------------------------------------
        # Characteristic function (fully vectorized)
        # --------------------------------------------------------------
        # Shape: (n_nodes, 1)
        temp = phi_diag[:, None]

        # Output shape: (2 * num_time_points, n_nodes)
        chi_tau = charac_function(time_pnts, temp)

        # Write into final tensor
        idx_start = tau_idx * 2 * num_time_points
        idx_end = idx_start + 2 * num_time_points
        chi[:, idx_start:idx_end] = chi_tau.T

    if verbose:
        print("GraphWave embedding completed (fast version).")

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
        self.embeddings = {node: chi_final[i] for i, node in enumerate(graph.nodes())}
        
        return self.embeddings