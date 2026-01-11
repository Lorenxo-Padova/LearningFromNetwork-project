"""
Optimized characteristic_functions.py for GraphWave
Vectorized, sparse-friendly, float32
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sb
import math

def charac_function(time_points, temp):
    """
    Sparse-friendly and vectorized computation of the characteristic function
    for a set of node embeddings.

    Args:
        time_points: 1D array of time points to evaluate
        temp: (n_nodes x n_features) sparse matrix (CSR) or dense array

    Returns:
        final_sig: 2D array of shape (2 * len(time_points), n_nodes)
                   rows: [Re(t0), Im(t0), Re(t1), Im(t1), ...] per node
    """
    n_nodes = temp.shape[0]
    n_timepnts = len(time_points)
    final_sig = np.zeros((2 * n_timepnts, n_nodes), dtype=np.float32)

    if sp.issparse(temp):
        temp_csr = temp.tocsr().astype(np.float32)
        data = temp_csr.data
        indptr = temp_csr.indptr
        n_features = temp_csr.shape[1]
        nnz_per_row = np.diff(indptr)

        for it, t in enumerate(time_points):
            # Compute cos and sin for all non-zero entries at once
            cos_vals = np.cos(t * data)
            sin_vals = np.sin(t * data)

            # Sum per row using np.add.reduceat
            cos_sum = np.add.reduceat(cos_vals, indptr[:-1])
            sin_sum = np.add.reduceat(sin_vals, indptr[:-1])

            # Add contribution from zeros (cos(0)=1, sin(0)=0)
            cos_sum += n_features - nnz_per_row

            final_sig[it * 2, :] = cos_sum / n_features
            final_sig[it * 2 + 1, :] = sin_sum / n_features
    else:
        # Dense fallback (vectorized)
        temp = temp.astype(np.float32)
        n_features = temp.shape[1]
        for it, t in enumerate(time_points):
            final_sig[it * 2, :] = np.cos(t * temp).mean(axis=1)
            final_sig[it * 2 + 1, :] = np.sin(t * temp).mean(axis=1)

    return final_sig


def charac_function_multiscale(heat, time_points):
    """
    Compute multiscale characteristic function in a sparse-friendly way.

    Args:
        heat: dict of {tau_idx: sparse/dense matrix (n_nodes, n_features)}
        time_points: array of time points to evaluate

    Returns:
        final_sig: 2D array, rows = features, cols = nodes
    """
    final_sig_list = []
    for tau_idx in sorted(heat.keys()):
        final_sig_list.append(charac_function(time_points, heat[tau_idx]))
    return np.vstack(final_sig_list).T


def plot_characteristic_function(phi_s, bunch, time_points, ind_tau):
    """Simple function for plotting characteristic function for selected nodes"""
    sb.set_style('white')
    plt.figure()
    n_time_pnts = len(time_points)
    cmap = plt.cm.get_cmap('RdYlBu')
    for n in bunch:
        x = [phi_s[n, ind_tau * n_time_pnts + 2 * j] for j in range(n_time_pnts)]
        y = [phi_s[n, ind_tau * n_time_pnts + 2 * j + 1] for j in range(n_time_pnts)]
        plt.scatter(x, y, c=cmap(n), label="node "+str(n))
    plt.legend(loc='upper left')
    plt.title('Characteristic function of distribution for selected nodes')
    plt.show()


def plot_angle_chi(f, t=[]):
    """Compute evolution of the angle of a 2D parametric curve"""
    if len(t) == 0:
        t = range(f.shape[0])
    theta = np.zeros(f.shape[0])
    for tt in t:
        theta[tt] = math.atan2(f[tt, 1], f[tt, 0])
    return theta
