# -*- coding: utf-8 -*-
"""
This file contains the script for defining characteristic functions
and using them as a way to embed distributional information
in Euclidean space
"""
import cmath
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import scipy.sparse as sp


def plot_characteristic_function(phi_s, bunch, time_pnts, ind_tau):
    ''' simple function for plotting the variation that is induced
        INPUT:
        ===========================================================================
        phi_s   :    array: each node is a row,
                     and the entries are the concatenated Re/Im values of
                     the characteristic function for the different
                     values in taus (output of chi_vary_scale)
        bunch   :    list of nodes for which to visualize the corresponding
                     characteristic curves
        taus    :    list of scale values corresponding to phi_s
                     (corresponding input of chi_vary_scale)
        OUTPUT:
        ===========================================================================
        None
    '''
    sb.set_style('white')
    plt.figure()
    n_time_pnts = len(time_pnts)
    cmap = plt.cm.get_cmap('RdYlBu')
    for n in bunch:
            x = [phi_s[n, ind_tau * n_time_pnts + 2 * j]
                 for j in range(n_time_pnts)]
            y = [phi_s[n, ind_tau * n_time_pnts + 2 * j + 1]
                 for j in range(n_time_pnts)]
            plt.scatter(x, y, c=cmap(n), label="node "+str(n), cmap=cmap)
    plt.legend(loc='upper left')
    plt.title('characteristic function of the distribution as s varies')
    plt.show()
    return


def plot_angle_chi(f, t=[], savefig=False, filefig='plots/angle_chi.png'):
    '''Plots the evolution of the angle of a 2D paramteric curve with time
    Parameters
    ----------
    f : 2D paramteric curve (columns corresponds to  X and Y)
    t: (optional) values where the curve is evaluated
    Returns
    -------
    theta: time series of the associated angle (array)
    '''
    if len(t) == 0:
        t = range(f.shape[0])
    theta = np.zeros(f.shape[0])
    for tt in t:
        theta[tt] = math.atan(f[tt, 1] * 1.0 / f[tt, 0])
    return theta

def charac_function(time_points, temp):
    """
    Sparse-safe computation of the characteristic function for a set of node embeddings.
    
    Args:
        time_points: 1D array of time points to evaluate the characteristic function
        temp: (n_nodes x n_features) sparse matrix of wavelet embeddings

    Returns:
        final_sig: 2D array of shape (2 * len(time_points), n_nodes)
                   rows: [Re(t0), Im(t0), Re(t1), Im(t1), ...] per node
    """
    n_nodes = temp.shape[0] if sp.issparse(temp) else temp.shape[0]
    n_timepnts = len(time_points)
    final_sig = np.zeros((2 * n_timepnts, n_nodes))

    if sp.issparse(temp):
        temp_csr = temp.tocsr()
        for it, t in enumerate(time_points):
            cos_sum = np.zeros(n_nodes)
            sin_sum = np.zeros(n_nodes)
            for row_idx in range(n_nodes):
                row_data = temp_csr.data[temp_csr.indptr[row_idx]:temp_csr.indptr[row_idx+1]]
                if len(row_data) > 0:
                    cos_sum[row_idx] = np.cos(t * row_data).sum()
                    sin_sum[row_idx] = np.sin(t * row_data).sum()
                # Add contribution from zeros in the sparse vector
                n_zeros = temp_csr.shape[1] - len(row_data)
                cos_sum[row_idx] += n_zeros  # cos(0) = 1
                # sin(0) = 0, so sin_sum is fine
            final_sig[it * 2, :] = cos_sum / temp_csr.shape[1]  # Re part
            final_sig[it * 2 + 1, :] = sin_sum / temp_csr.shape[1]  # Im part
    else:
        # Dense fallback (same as original)
        for it, t in enumerate(time_points):
            final_sig[it * 2, :] = np.cos(t * temp).mean(axis=1)
            final_sig[it * 2 + 1, :] = np.sin(t * temp).mean(axis=1)

    return final_sig


def charac_function_multiscale(heat, time_points):
    """
    Compute multiscale characteristic function in a sparse-safe way.

    Args:
        heat: dict of {tau_idx: sparse matrix of shape (n_nodes, n_features)}
        time_points: array of time points to evaluate

    Returns:
        final_sig: 2D array, rows = features, cols = nodes
    """
    final_sig_list = []
    for tau_idx in sorted(heat.keys()):
        final_sig_list.append(charac_function(time_points, heat[tau_idx]))
    return np.vstack(final_sig_list).T
