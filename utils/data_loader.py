"""
Data loading and preprocessing utilities
"""
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import KFold
import random

def load_graph_data(filepath):
    """
    Load homogeneous or heterogeneous edge list.

    Required columns:
      - Source, Target
    Optional columns:
      - SourceType, TargetType
    """
    df = pd.read_csv(filepath)

    required = {"Source", "Target"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain Source and Target columns")

    df["Source"] = df["Source"].astype(str)
    df["Target"] = df["Target"].astype(str)

    if "SourceType" in df.columns and "TargetType" in df.columns:
        df["SourceType"] = df["SourceType"].astype(str)
        df["TargetType"] = df["TargetType"].astype(str)
        heterogeneous = True
    else:
        heterogeneous = False

    df = df[df["Source"] != df["Target"]].drop_duplicates()

    return df, heterogeneous

def create_graph_from_edges(edge_df):
    """
    Create a NetworkX graph from edge DataFrame.
    Supports heterogeneous graphs if SourceType/TargetType are present.
    """
    G = nx.Graph()

    heterogeneous = "SourceType" in edge_df.columns

    for _, row in edge_df.iterrows():
        u, v = row["Source"], row["Target"]
        G.add_edge(u, v)

        if heterogeneous:
            G.nodes[u]["ntype"] = row["SourceType"]
            G.nodes[v]["ntype"] = row["TargetType"]
        else:
            G.nodes[u]["ntype"] = "protein"
            G.nodes[v]["ntype"] = "protein"

    return G


def generate_negative_samples(graph, num_samples, random_state=42):
    """
    Generate negative samples (non-existent edges) from the graph.
    
    Args:
        graph (networkx.Graph): Input graph
        num_samples (int): Number of negative samples to generate
        random_state (int): Random seed
        
    Returns:
        pandas.DataFrame: DataFrame with negative edges
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    nodes = list(graph.nodes())
    negative_samples = []
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(negative_samples) < num_samples and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        
        # Ensure the edge doesn't exist and nodes are different
        if not graph.has_edge(u, v) and u != v:
            negative_samples.append({'Source': u, 'Target': v})
        
        attempts += 1
    
    if len(negative_samples) < num_samples:
        print(f"Warning: Only generated {len(negative_samples)} negative samples out of {num_samples} requested")
    
    return pd.DataFrame(negative_samples)

def create_cv_folds(edges_df, n_folds=5, random_state=42):
    """
    Create cross-validation folds for edge prediction.
    
    Args:
        edges_df (pandas.DataFrame): DataFrame with edges
        n_folds (int): Number of folds
        random_state (int): Random seed
        
    Returns:
        list: List of (train_indices, test_indices) tuples
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = list(kf.split(edges_df))
    
    return folds

def prepare_fold_data(edges_df, train_idx, test_idx, negative_ratio=1.0, random_state=42):
    """
    Prepare training and test data for a single fold.
    
    Args:
        edges_df (pandas.DataFrame): All edges
        train_idx (array): Indices for training
        test_idx (array): Indices for testing
        negative_ratio (float): Ratio of negative to positive samples
        random_state (int): Random seed
        
    Returns:
        tuple: (train_edges, test_positive_edges, test_negative_edges, train_graph)
    """
    # Split edges
    train_edges = edges_df.iloc[train_idx].copy()
    test_edges = edges_df.iloc[test_idx].copy()
    
    # Create training graph
    print("Creating training graph...")
    train_graph = create_graph_from_edges(train_edges)
    
    print("Generating negative samples for testing...")
    # Generate negative samples for testing
    num_negative = int(len(test_edges) * negative_ratio)
    test_negative = generate_negative_samples(train_graph, num_negative, random_state)
    
    return train_edges, test_edges, test_negative, train_graph