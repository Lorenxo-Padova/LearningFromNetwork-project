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
    
    If the first line contains "heterogeneous", infers node types from node ID prefixes
    (e.g., 'gene:123', 'compound:456', 'disease:789')
    """
    # Check first line for heterogeneous indicator
    with open(filepath, 'r') as f:
        first_line = f.readline().strip().lower()
    
    #TODO: CHANGE THE FIRST LINE OF THE CSV FILE TO CHECK EVERY TIME
    # heterogeneous_flag = 'heterogeneous' in first_line
    heterogeneous_flag = True

    with open(filepath, "rb") as f:
        print(f.readline())
        print(f.readline())
    df = pd.read_csv(filepath,sep=",", engine="python")

    required = {"Source", "Target"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain Source and Target columns")

    heterogeneous = False
    
    if heterogeneous_flag:
        # Infer node types from prefixes
        def extract_node_type(node_id):
            """Extract node type from prefix (e.g., 'Gene::...', 'Compound::...', 'Disease::...')"""
            node_str = str(node_id).lower()
            if node_str.startswith('gene::'):
                return 'gene'
            elif node_str.startswith('compound::'):
                return 'compound'
            elif node_str.startswith('disease::'):
                return 'disease'
            return None
        
        # Apply extraction to all nodes
        df["SourceType"] = df["Source"].apply(extract_node_type)
        df["TargetType"] = df["Target"].apply(extract_node_type)
        heterogeneous = True
        print("Heterogeneous graph detected from first line; extracting node types from prefixes")
    elif "SourceType" in df.columns and "TargetType" in df.columns:
        df["SourceType"] = df["SourceType"].astype(str)
        df["TargetType"] = df["TargetType"].astype(str)
        heterogeneous = True
        print("Heterogeneous graph detected from SourceType/TargetType columns")
    else:
        print("Homogeneous graph detected; assigning default node type 'protein'")
        heterogeneous = False

    df = df[df["Source"] != df["Target"]].drop_duplicates()

    return df, heterogeneous

def create_graph_from_edges(edge_df):
    """
    Create a NetworkX graph from edge DataFrame.
    Supports heterogeneous graphs if SourceType/TargetType are present.
    Sets node attribute 'type' for heterogeneous graphs.
    """
    G = nx.Graph()

    heterogeneous = "SourceType" in edge_df.columns

    for _, row in edge_df.iterrows():
        u, v = row["Source"], row["Target"]
        G.add_edge(u, v)

        if heterogeneous:
            # Use 'type' attribute for metapath2vec compatibility
            G.nodes[u]["type"] = row["SourceType"]
            G.nodes[v]["type"] = row["TargetType"]
        else:
            G.nodes[u]["type"] = "protein"
            G.nodes[v]["type"] = "protein"

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
        tuple: (train_positive_edges, train_negative_edges,
                test_positive_edges, test_negative_edges, train_graph)
    """
    # Split edges
    train_edges = edges_df.iloc[train_idx].copy()
    test_edges = edges_df.iloc[test_idx].copy()
    
    # Create training graph
    print("Creating training graph...")
    train_graph = create_graph_from_edges(train_edges)
    
    print("Generating negative samples for training...")
    num_train_neg = int(len(train_edges) * negative_ratio)
    train_negative = generate_negative_samples(train_graph, num_train_neg, random_state)

    print("Generating negative samples for testing...")
    num_test_neg = int(len(test_edges) * negative_ratio)
    test_negative = generate_negative_samples(train_graph, num_test_neg, random_state + 1)

    return train_edges, train_negative, test_edges, test_negative, train_graph