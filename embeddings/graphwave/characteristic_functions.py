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
    Fully vectorized characteristic function computation using NumPy broadcasting.
    Computes all time points at once instead of looping.

    Args:
        time_points: 1D array of time points to evaluate
        temp: (n_nodes x n_features) sparse matrix (CSR) or dense array

    Returns:
        final_sig: 2D array of shape (2 * len(time_points), n_nodes)
                   rows: [Re(t0), Im(t0), Re(t1), Im(t1), ...] per node
    """
    n_nodes = temp.shape[0]
    n_features = temp.shape[1]
    n_timepnts = len(time_points)
    
    # Convert to dense if sparse (small size, better for vectorization)
    if sp.issparse(temp):
        temp = temp.toarray().astype(np.float32)
    else:
        temp = temp.astype(np.float32)
    
    time_points = np.asarray(time_points, dtype=np.float32)
    
    # Vectorized computation: (n_nodes, n_features) * (n_timepnts,) -> (n_timepnts, n_nodes, n_features)
    # Using einsum for efficient broadcasting: temp[n,f] * time[t] -> result[t,n,f]
    temp_scaled = np.einsum('nf,t->tnf', temp, time_points, optimize=True)
    
    # Compute cos and sin for all time points at once
    cos_vals = np.cos(temp_scaled)  # (n_timepnts, n_nodes, n_features)
    sin_vals = np.sin(temp_scaled)  # (n_timepnts, n_nodes, n_features)
    
    # Mean over features axis
    cos_mean = cos_vals.mean(axis=2)  # (n_timepnts, n_nodes)
    sin_mean = sin_vals.mean(axis=2)  # (n_timepnts, n_nodes)
    
    # Interleave Re and Im: [Re(t0), Im(t0), Re(t1), Im(t1), ...]
    final_sig = np.zeros((2 * n_timepnts, n_nodes), dtype=np.float32)
    final_sig[0::2, :] = cos_mean  # Even rows: cosine (real part)
    final_sig[1::2, :] = sin_mean  # Odd rows: sine (imaginary part)
    
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


def plot_angle_chi(f, t=[]):
    """Compute evolution of the angle of a 2D parametric curve"""
    if len(t) == 0:
        t = range(f.shape[0])
    theta = np.zeros(f.shape[0])
    for tt in t:
        theta[tt] = math.atan2(f[tt, 1], f[tt, 0])
    return theta
