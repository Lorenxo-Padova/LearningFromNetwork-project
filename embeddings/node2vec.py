"""
Node2Vec embedding implementation (DeepWalk-style, no alias tables)
"""
import random
import numpy as np
from gensim.models import Word2Vec
from embeddings.base_embedder import BaseEmbedder


class Node2VecEmbedder(BaseEmbedder):
    """
    Node2Vec implementation without alias precomputation.
    Memory profile similar to DeepWalk, but slower per step.
    """

    def __init__(
        self,
        embedding_dim=32,
        walk_length=10,
        num_walks=20,
        p=1.0,
        q=1.0,
        window_size=10,
        workers=1,
        random_state=42
    ):
        super().__init__(embedding_dim, random_state)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.workers = workers
        random.seed(random_state)
        np.random.seed(random_state)

    def _biased_choice(self, prev, cur, graph):
        """
        Compute Node2Vec transition probabilities on the fly.
        NO caching, NO alias tables.
        """
        neighbors = list(graph.neighbors(cur))
        if not neighbors:
            return None

        probs = np.empty(len(neighbors), dtype=np.float32)

        for i, dst in enumerate(neighbors):
            weight = graph[cur][dst].get("weight", 1.0)

            if dst == prev:
                probs[i] = weight / self.p
            elif graph.has_edge(dst, prev):
                probs[i] = weight
            else:
                probs[i] = weight / self.q

        probs /= probs.sum()
        return np.random.choice(neighbors, p=probs)

    def _node2vec_walk(self, graph, start_node):
        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(graph.neighbors(cur))
            if not neighbors:
                break

            if len(walk) == 1:
                walk.append(random.choice(neighbors))
            else:
                nxt = self._biased_choice(walk[-2], cur, graph)
                if nxt is None:
                    break
                walk.append(nxt)

        return walk

    def _generate_walks(self, graph):
        nodes = list(graph.nodes())
        walks = []

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(graph, node))

        return walks

    def generate_embeddings(self, graph):
        print("[Node2Vec] Generating walks (DeepWalk-style)...")
        walks = self._generate_walks(graph)
        print(f"[Node2Vec] Generated {len(walks)} walks")

        print("[Node2Vec] Training Word2Vec...")
        model = Word2Vec(
            walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,
            workers=self.workers,
            seed=self.random_state,
            negative=5,
            sample=1e-4
        )

        print("[Node2Vec] Extracting embeddings...")
        self.embeddings = {
            node: model.wv[node]
            for node in model.wv.index_to_key
        }

        return self.embeddings

    def __str__(self):
        return (
            f"Node2Vec(no-alias, dim={self.embedding_dim}, "
            f"walk_length={self.walk_length}, num_walks={self.num_walks}, "
            f"p={self.p}, q={self.q})"
        )
