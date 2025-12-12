"""
DeepWalk embedding implementation
"""
import random
from gensim.models import Word2Vec
from embeddings.base_embedder import BaseEmbedder

class DeepWalkEmbedder(BaseEmbedder):
    """
    DeepWalk implementation for graph embedding.
    Uses random walks and Word2Vec to learn node representations.
    """
    
    def __init__(self, embedding_dim=32, walk_length=10, num_walks=20, 
                 window_size=10, workers=1, random_state=42):
        """
        Initialize DeepWalk embedder.
        
        Args:
            embedding_dim (int): Dimension of embedding vectors
            walk_length (int): Length of each random walk
            num_walks (int): Number of random walks per node
            window_size (int): Window size for Word2Vec
            workers (int): Number of parallel workers
            random_state (int): Random seed
        """
        super().__init__(embedding_dim, random_state)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        random.seed(random_state)
        
    def _generate_random_walks(self, graph):
        """
        Generate random walks starting from each node.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            list: List of random walks (each walk is a list of nodes)
        """
        nodes = list(graph.nodes())
        walks = []
        
        for start_node in nodes:
            for _ in range(self.num_walks):
                walk = [start_node]
                current_node = start_node
                
                for _ in range(self.walk_length - 1):
                    neighbors = list(graph.neighbors(current_node))
                    if not neighbors:
                        break
                    
                    current_node = random.choice(neighbors)
                    walk.append(current_node)
                    
                walks.append(walk)
                
        return walks
    
    def generate_embeddings(self, graph):
        """
        Generate DeepWalk embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        # Generate random walks
        walks = self._generate_random_walks(graph)
        
        # Train Word2Vec model on walks
        model = Word2Vec(
            walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,  # Skip-gram
            workers=self.workers,
            seed=self.random_state
        )
        
        # Extract embeddings
        self.embeddings = {node: model.wv[node] for node in model.wv.index_to_key}
        
        return self.embeddings
    
    def __str__(self):
        return (f"DeepWalk(dim={self.embedding_dim}, "
                f"walk_length={self.walk_length}, "
                f"num_walks={self.num_walks})")