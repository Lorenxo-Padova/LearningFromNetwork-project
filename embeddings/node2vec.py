"""
Node2Vec embedding implementation
"""
import random
import numpy as np
from gensim.models import Word2Vec
from embeddings.base_embedder import BaseEmbedder

class Node2VecEmbedder(BaseEmbedder):
    """
    Node2Vec implementation for graph embedding.
    Extends DeepWalk with biased random walks controlled by p and q parameters.
    """
    
    def __init__(self, embedding_dim=32, walk_length=10, num_walks=20,
                 p=1.0, q=1.0, window_size=10, workers=1, random_state=42):
        """
        Initialize Node2Vec embedder.
        
        Args:
            embedding_dim (int): Dimension of embedding vectors
            walk_length (int): Length of each random walk
            num_walks (int): Number of random walks per node
            p (float): Return parameter (controls likelihood of revisiting a node)
            q (float): In-out parameter (controls BFS vs DFS behavior)
            window_size (int): Window size for Word2Vec
            workers (int): Number of parallel workers
            random_state (int): Random seed
        """
        super().__init__(embedding_dim, random_state)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.workers = workers
        random.seed(random_state)
        np.random.seed(random_state)
        
    def _get_alias_edge(self, src, dst, graph):
        """
        Get the alias setup for the edge from src to dst.
        
        Args:
            src: Source node
            dst: Destination node
            graph: NetworkX graph
            
        Returns:
            tuple: Normalized probabilities and alias tables
        """
        unnormalized_probs = []
        for dst_nbr in sorted(graph.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(graph[dst][dst_nbr].get('weight', 1) / self.p)
            elif graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(graph[dst][dst_nbr].get('weight', 1))
            else:
                unnormalized_probs.append(graph[dst][dst_nbr].get('weight', 1) / self.q)
        
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        
        return normalized_probs
    
    def _node2vec_walk(self, graph, start_node):
        """
        Simulate a biased random walk starting from start_node.
        
        Args:
            graph: NetworkX graph
            start_node: Starting node for the walk
            
        Returns:
            list: Random walk sequence
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(graph.neighbors(cur))
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    probs = self._get_alias_edge(prev, cur, graph)
                    walk.append(np.random.choice(cur_nbrs, p=probs))
            else:
                break
                
        return walk
    
    def _generate_walks(self, graph):
        """
        Generate Node2Vec random walks.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            list: List of random walks
        """
        walks = []
        nodes = list(graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(graph, node))
                
        return walks
    
    def generate_embeddings(self, graph):
        """
        Generate Node2Vec embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        # Generate biased random walks
        walks = self._generate_walks(graph)
        
        # Train Word2Vec model
        model = Word2Vec(
            walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,
            workers=self.workers,
            seed=self.random_state
        )
        
        # Extract embeddings
        self.embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
        
        return self.embeddings
    
    def __str__(self):
        return (f"Node2Vec(dim={self.embedding_dim}, "
                f"walk_length={self.walk_length}, "
                f"num_walks={self.num_walks}, "
                f"p={self.p}, q={self.q})")