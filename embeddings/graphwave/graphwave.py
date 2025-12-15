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


def heat_diffusion_ind(graph, taus=TAUS, order=ORDER, proc=PROC, lap=None, eta_min=0.8, eta_max=0.95):
    """
    Memory-efficient heat diffusion for GraphWave.
    Computes node embeddings without forming full n x n matrices.

    Args:
        graph : networkx.Graph
        taus  : list of scales
        order : Chebyshev polynomial order
        proc  : 'exact' or 'approximate'
        lap   : optional precomputed Laplacian
        eta_min, eta_max: used if taus='auto' to select scales

    Returns:
        heat_dict: dict of node -> list of wavelet vectors per scale
        taus: scales used
    """

    # Compute Laplacian if not provided
    if lap is None:
        a = nx.adjacency_matrix(graph)
        lap = laplacian(a)

    n_nodes = lap.shape[0]

    # Exact computation
    if proc == 'exact':
        lamb, U = np.linalg.eigh(lap.todense())
        heat_dict = {}
        for u_idx, node in enumerate(graph.nodes()):
            heat_dict[node] = [U[u_idx, :] @ np.diag(np.exp(-tau*lamb)) @ U.T for tau in taus]
        return heat_dict, taus

    # Approximate computation
    heat_dict = {node: [] for node in graph.nodes()}
    L_hat = lap - sp.eye(n_nodes)
    i = 0
    # Row-wise Chebyshev recursion per node and per scale
    for tau in taus:
        coeffs = compute_cheb_coeff_basis(tau, order)
        for u_idx, node in enumerate(graph.nodes()):
            i += 1
            if i % 1000 == 0:
                print(f"Processing node {i}/{n_nodes} for tau={tau}")
            # Initialize T_0 and T_1 for this node
            T_km2 = np.zeros(n_nodes)
            T_km2[u_idx] = 1.0          # e_u
            T_km1 = L_hat @ T_km2       # sparse matvec

            # Initialize node wavelet
            phi_u = coeffs[0] * T_km2 + coeffs[1] * T_km1

            # Recurrence
            for k in range(2, order + 1):
                T_k = 2 * (L_hat @ T_km1) - T_km2
                phi_u += coeffs[k] * T_k

                # Optional pruning for sparsity
                T_k[np.abs(T_k) < 1e-12] = 0

                T_km2, T_km1 = T_km1, T_k

            heat_dict[node].append(phi_u)

    return heat_dict, taus



def graphwave_alg(graph, time_pnts, taus='auto', 
                verbose=False, approximate_lambda=True,
                order=ORDER, proc=PROC, nb_filters=NB_FILTERS,
                **kwargs):
    ''' wrapper function for computing the structural signatures using GraphWave
    INPUT
    --------------------------------------------------------------------------------------
    graph             :   nx Graph
    time_pnts         :   time points at which to evaluate the characteristic function
    taus              :   list of scales that we are interested in. Alternatively,
                        'auto' for the automatic version of GraphWave
    verbose           :   the algorithm prints some of the hidden parameters
                        as it goes along
    approximate_lambda:   (boolean) should the range oflambda be approximated or
                        computed?
    proc              :   which procedure to compute the signatures (approximate == that
                        is, with Chebychev approx -- or exact)
    nb_filters        :   nuber of taus that we require if  taus=='auto'
    OUTPUT
    --------------------------------------------------------------------------------------
    chi               :  embedding of the function in Euclidean space
    heat_print        :  returns the actual embeddings of the nodes
    taus              :  returns the list of scales used.
    '''

    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    # Compute Laplacian if needed
    lap = None
    if taus == 'auto' and approximate_lambda is not True:
        start = time.time()
        a = nx.adjacency_matrix(graph)
        print("Computed adjacency matrix in {:.4f} seconds".format(time.time() - start))
        start = time.time()
        lap = laplacian(a)
        print("Computed laplacian in {:.4f} seconds".format(time.time() - start))
        try:
            l1 = np.sort(sc.sparse.linalg.eigsh(lap, 2,  which='SM',return_eigenvectors=False))[1]
        except:
            l1 = np.sort(sc.sparse.linalg.eigsh(lap, 5,  which='SM',return_eigenvectors=False))[1]
    elif taus == 'auto':
        l1 = 1.0 / n_nodes

    if taus == 'auto':
        smax = -np.log(ETA_MIN) * np.sqrt(0.5 / l1)
        smin = -np.log(ETA_MAX) * np.sqrt(0.5 / l1)
        taus = np.linspace(smin, smax, nb_filters)

    n_taus = len(taus)
    num_time_points = len(time_pnts)

    # Precompute L_hat if approximate
    if proc == 'approximate':
        if lap is None:
            lap = laplacian(nx.adjacency_matrix(graph))
        L_hat = lap - sp.eye(n_nodes)

    # Initialize heat_print_dict to store row-wise wavelets
    heat_print_dict = {node: [] for node in nodes}

    start_total = time.time()
    i = 0
    for tau_idx, tau in enumerate(taus):
        start_tau = time.time()
        coeffs = compute_cheb_coeff_basis(tau, order)

        for u_idx, node in enumerate(nodes):
            i += 1
            if i % 1000 == 0:
                print(f"Processing node {i}/{n_nodes} for tau={tau:.4f}")

            # Row-wise Chebyshev recursion
            T_km2 = np.zeros(n_nodes)
            T_km2[u_idx] = 1.0
            T_km1 = L_hat @ T_km2
            phi_u = coeffs[0] * T_km2 + coeffs[1] * T_km1

            for k in range(2, order + 1):
                T_k = 2 * (L_hat @ T_km1) - T_km2
                phi_u += coeffs[k] * T_k
                T_k[np.abs(T_k) < 1e-12] = 0
                T_km2, T_km1 = T_km1, T_k

            heat_print_dict[node].append(phi_u)

        print(f"Finished tau {tau_idx+1}/{n_taus} in {time.time() - start_tau:.4f} seconds")

    print(f"Finished all taus in {time.time() - start_total:.4f} seconds")

   # Restructure heat_print_dict into truly sparse row-by-row matrix
    start = time.time()
    heat_print = {}
    for tau_idx in range(n_taus):
        # Initialize empty sparse matrix in LIL format for efficient row assignment
        sparse_matrix = sp.lil_matrix((n_nodes, n_nodes))
        for row_idx, node in enumerate(nodes):
            # Assign the row directly
            sparse_matrix[row_idx, :] = heat_print_dict[node][tau_idx]
        # Convert to CSR for faster subsequent computations
        heat_print[tau_idx] = sparse_matrix.tocsr()
    print("Restructured heat diffusion data (fully sparse) in {:.4f} seconds".format(time.time() - start))
    
    # Compute characteristic function
    start = time.time()
    chi = charac_function_multiscale(heat_print, time_pnts)
    print("Computed characteristic function in {:.4f} seconds".format(time.time() - start))

    return chi, heat_print, taus

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
        num_time_points = max(1, self.embedding_dim // (2 * self.nb_filters))
        time_points = np.linspace(-math.pi, math.pi, num_time_points)
        
        chi, _, _ = graphwave_alg(graph, time_points, taus='auto',
                                    approximate_lambda=True,
                                    order=self.order, proc=self.proc,
                                    nb_filters=self.nb_filters)
        print(f"GraphWave: Generated chi with shape {chi.shape}")
        # Standardize features (chi shape: [features, nodes])
        self.scaler = StandardScaler()
        chi_scaled = self.scaler.fit_transform(chi.T)  # Transpose to [nodes, features]
        
        # Adjust dimensionality if needed
        actual_dim = chi_scaled.shape[1]
        if actual_dim > self.embedding_dim:
            # Truncate if we have more features than needed
            chi_final = chi_scaled[:, :self.embedding_dim]
        elif actual_dim < self.embedding_dim:
            # Pad with zeros if we have fewer features
            padding = np.zeros((chi_scaled.shape[0], self.embedding_dim - actual_dim))
            chi_final = np.hstack([chi_scaled, padding])
        else:
            chi_final = chi_scaled
        
        # Store embeddings in a dictionary
        self.embeddings = {node: chi_final[i] for i, node in enumerate(graph.nodes())}
        
        return self.embeddings