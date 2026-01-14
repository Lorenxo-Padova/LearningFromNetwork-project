"""
Node2Vec embedding implementation - Optimized for Speed
"""
import random
import numpy as np
from gensim.models import Word2Vec
from embeddings.base_embedder import BaseEmbedder

class Node2VecEmbedder(BaseEmbedder):
    """
    Node2Vec implementation for graph embedding.
    Extends DeepWalk with biased random walks controlled by p and q parameters.
    Optimized for speed with precomputed transition probabilities.
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
        self.alias_edges = {}
    
    def _precompute_transition_probs(self, graph):
        """
        Precompute all transition probabilities for faster walk generation.
        Trades RAM for speed.
        
        Args:
            graph: NetworkX graph
        """
        print("[Node2Vec] Precomputing transition probabilities...")
        edges = graph.edges()
        
        for src, dst in edges:
            src_nbrs = list(graph.neighbors(src))
            unnormalized_probs = []
            
            for dst_nbr in src_nbrs:
                if dst_nbr == src:
                    unnormalized_probs.append(graph[src][dst_nbr].get('weight', 1) / self.p)
                elif graph.has_edge(dst_nbr, src):
                    unnormalized_probs.append(graph[src][dst_nbr].get('weight', 1))
                else:
                    unnormalized_probs.append(graph[src][dst_nbr].get('weight', 1) / self.q)
            
            norm_const = sum(unnormalized_probs)
            if norm_const > 0:
                normalized_probs = np.array(unnormalized_probs, dtype=np.float32) / norm_const
                self.alias_edges[(src, dst)] = (np.array(src_nbrs), normalized_probs)
    
    def _get_alias_edge(self, prev, cur, graph):
        """
        Get precomputed or compute transition probabilities.
        
        Args:
            prev: Previous node
            cur: Current node
            graph: NetworkX graph
            
        Returns:
            tuple: (neighbor_list, probabilities)
        """
        if (prev, cur) in self.alias_edges:
            return self.alias_edges[(prev, cur)]
        
        # Fallback: compute on the fly
        cur_nbrs = np.array(list(graph.neighbors(cur)))
        unnormalized_probs = []
        
        for dst_nbr in cur_nbrs:
            if dst_nbr == prev:
                unnormalized_probs.append(graph[cur][dst_nbr].get('weight', 1) / self.p)
            elif graph.has_edge(dst_nbr, prev):
                unnormalized_probs.append(graph[cur][dst_nbr].get('weight', 1))
            else:
                unnormalized_probs.append(graph[cur][dst_nbr].get('weight', 1) / self.q)
        
        norm_const = sum(unnormalized_probs)
        normalized_probs = np.array(unnormalized_probs, dtype=np.float32) / norm_const
        return cur_nbrs, normalized_probs
    
    def _node2vec_walk(self, graph, start_node):
        """
        Simulate a biased random walk starting from start_node (vectorized).
        
        Args:
            graph: NetworkX graph
            start_node: Starting node for the walk
            
        Returns:
            list: Random walk sequence
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = list(graph.neighbors(cur))
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(np.random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    nbr_array, probs = self._get_alias_edge(prev, cur, graph)
                    next_node = np.random.choice(nbr_array, p=probs)
                    walk.append(next_node)
            else:
                break
                
        return [str(n) for n in walk]  # Convert to strings for Word2Vec
    
    def _generate_walks(self, graph):
        """
        Generate Node2Vec random walks with vectorized approach.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            list: List of random walks
        """
        print("[Node2Vec] Generating random walks...")
        walks = []
        nodes = list(graph.nodes())
        
        for walk_iter in range(self.num_walks):
            if (walk_iter + 1) % 5 == 0:
                print(f"[Node2Vec] Completed {walk_iter + 1}/{self.num_walks} walk iterations")
            
            # Shuffle nodes once per iteration
            np.random.shuffle(nodes)
            for idx, node in enumerate(nodes):
                walks.append(self._node2vec_walk(graph, node))
                if (idx + 1) % 10000 == 0:
                    print(f"[Node2Vec] Generated walks for {idx + 1} nodes in iteration {walk_iter + 1}")
                
        return walks
    
    def generate_embeddings(self, graph):
        """
        Generate Node2Vec embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        # Precompute transition probabilities (trades RAM for speed)
        self._precompute_transition_probs(graph)
        
        # Generate biased random walks
        walks = self._generate_walks(graph)
        print(f"[Node2Vec] Generated {len(walks)} walks")
        
        # Train Word2Vec model with optimized parameters
        print("[Node2Vec] Training Word2Vec model...")
        model = Word2Vec(
            walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,  # Skip-gram model (faster than CBOW)
            workers=self.workers,
            seed=self.random_state,
            negative=15,  # Negative sampling (faster than hierarchical softmax)
            ns_exponent=0.75
        )
        
        # Extract embeddings
        self.embeddings = {}
        nodes = list(graph.nodes())
        for i, node in enumerate(nodes):
            node_str = str(node)
            if node_str in model.wv:
                self.embeddings[node] = model.wv[node_str]
            if (i + 1) % 10000 == 0:
                print(f"[Node2Vec] Extracted {i + 1} embeddings")
        
        print(f"[Node2Vec] Finished embedding all {len(nodes)} nodes")
        
        return self.embeddings
    
    def __str__(self):
        return (f"Node2Vec(dim={self.embedding_dim}, "
                f"walk_length={self.walk_length}, "
                f"num_walks={self.num_walks}, "
                f"p={self.p}, q={self.q})")