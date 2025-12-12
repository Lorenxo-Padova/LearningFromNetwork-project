"""
Base class for all embedding methods
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """
    Abstract base class for graph embedding methods.
    All embedding methods should inherit from this class.
    """
    
    def __init__(self, embedding_dim=32, random_state=42):
        """
        Initialize the embedder.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            random_state (int): Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.embeddings = None
        
    @abstractmethod
    def generate_embeddings(self, graph):
        """
        Generate embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        pass
    
    def get_embedding(self, node):
        """
        Get embedding for a specific node.
        
        Args:
            node: Node ID
            
        Returns:
            numpy.ndarray: Embedding vector for the node
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not generated yet. Call generate_embeddings() first.")
        
        if node not in self.embeddings:
            raise KeyError(f"Node {node} not found in embeddings.")
        
        return self.embeddings[node]
    
    def save_embeddings(self, filepath):
        """
        Save embeddings to a file.
        
        Args:
            filepath (str): Path to save the embeddings
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Generate embeddings first.")
        
        np.save(filepath, self.embeddings)
        
    def load_embeddings(self, filepath):
        """
        Load embeddings from a file.
        
        Args:
            filepath (str): Path to load the embeddings from
        """
        self.embeddings = np.load(filepath, allow_pickle=True).item()
        
    def __str__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim})"