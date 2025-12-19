"""
MetaPath2Vec embedding implementation.
"""
import random
import numpy as np
from gensim.models import Word2Vec
from embeddings.base_embedder import BaseEmbedder


class MetaPath2VecEmbedder(BaseEmbedder):
    """
    MetaPath2Vec implementation for heterogeneous graphs.
    Uses metapath-guided random walks and Word2Vec to learn node representations.
    """

    def __init__(
        self,
        embedding_dim=32,
        metapaths=None,
        walk_length=40,
        walks_per_node=10,
        window_size=5,
        workers=4,
        random_state=42,
    ):
        """
        Initialize MetaPath2Vec embedder.

        Args:
            embedding_dim (int): Dimension of embedding vectors.
            metapaths (list[list[str]]): List of metapath schemas (node types) to follow.
            walk_length (int): Length of each metapath-guided walk.
            walks_per_node (int): Number of walks to start for each node and metapath.
            window_size (int): Window size for Word2Vec.
            workers (int): Number of parallel workers for Word2Vec.
            random_state (int): Random seed for reproducibility.
        """
        super().__init__(embedding_dim, random_state)
        self.metapaths = metapaths or []
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.window_size = window_size
        self.workers = workers
        random.seed(random_state)
        np.random.seed(random_state)

    def _metapath_walk(self, graph, start_node, metapath):
        """
        Generate a single metapath-guided random walk starting from start_node.

        Args:
            graph (networkx.Graph): Input heterogeneous graph with node attribute "ntype".
            start_node: Node to start the walk from.
            metapath (list[str]): Sequence of node types defining the walk pattern.

        Returns:
            list: Sequence of nodes visited in the walk.
        """
        walk = [start_node]
        current = start_node

        for step in range(1, self.walk_length):
            expected_type = metapath[step % len(metapath)]
            neighbors = [
                n
                for n in graph.neighbors(current)
                if graph.nodes[n].get("ntype") == expected_type
            ]

            if not neighbors:
                break

            current = random.choice(neighbors)
            walk.append(current)

        return walk

    def generate_embeddings(self, graph):
        """
        Generate MetaPath2Vec embeddings for all nodes in the graph.

        Args:
            graph (networkx.Graph): Input heterogeneous graph with node attribute "ntype".

        Returns:
            dict: Dictionary mapping node IDs to embedding vectors.
        """
        if not self.metapaths:
            raise ValueError("metapaths must be provided to generate MetaPath2Vec embeddings")
        

        walks = []

        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get("ntype")
            for metapath in self.metapaths:
                if node_type != metapath[0]:
                    continue
                for _ in range(self.walks_per_node):
                    walks.append(self._metapath_walk(graph, node, metapath))
        
        if not walks:
            raise ValueError(
                "No meta-path walks were generated. "
                "Check that node types match the configured metapaths."
            )
        
        model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,  # Skip-gram
            workers=self.workers,
            seed=self.random_state,
        )

        self.embeddings = {
            node: model.wv[node] for node in graph.nodes() if node in model.wv
        }

        return self.embeddings

    def __str__(self):
        return (
            f"MetaPath2Vec(dim={self.embedding_dim}, "
            f"walk_length={self.walk_length}, "
            f"walks_per_node={self.walks_per_node})"
        )
