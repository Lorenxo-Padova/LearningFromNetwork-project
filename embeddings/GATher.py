"""
GATher (Graph Attention Network) embedding implementation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from embeddings.base_embedder import BaseEmbedder


class GATModel(nn.Module):
    """
    Graph Attention Network model for node embedding.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, 
                            concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATherEmbedder(BaseEmbedder):
    """
    GATher implementation for graph embedding using Graph Attention Networks.
    """
    
    def __init__(self, embedding_dim=32, hidden_dim=64, heads=4, 
                 epochs=200, lr=0.005, weight_decay=5e-4, dropout=0.6,
                 random_state=42):
        """
        Initialize GATher embedder.
        
        Args:
            embedding_dim (int): Dimension of output embedding vectors
            hidden_dim (int): Dimension of hidden layer
            heads (int): Number of attention heads in first layer
            epochs (int): Number of training epochs
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            dropout (float): Dropout rate
            random_state (int): Random seed
        """
        super().__init__(embedding_dim, random_state)
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def _prepare_data(self, graph):
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Data: PyTorch Geometric data object
        """
        # Convert to PyTorch Geometric format
        data = from_networkx(graph)
        
        # Use simple degree features (fast to compute)
        num_nodes = data.num_nodes
        
        # Use degree as a simple feature
        degrees = torch.tensor([graph.degree(node) for node in graph.nodes()], dtype=torch.float32)
        degrees = degrees.view(-1, 1)
        
        # Normalize degree
        max_degree = degrees.max()
        if max_degree > 0:
            degrees = degrees / max_degree
        
        # Use degree features or combine with existing features
        if hasattr(data, 'x') and data.x is not None:
            data.x = torch.cat([data.x, degrees], dim=1)
        else:
            data.x = degrees
        
        # Preprocess features with StandardScaler
        if data.x is not None and data.x.numel() > 0:
            self.scaler = StandardScaler()
            data.x = torch.FloatTensor(self.scaler.fit_transform(data.x.cpu().numpy()))
        
        return data.to(self.device)
    
    def generate_embeddings(self, graph):
        """
        Generate GATher embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        # Prepare data
        data = self._prepare_data(graph)
        in_channels = data.x.size(1)
        
        # Initialize model
        model = GATModel(
            in_channels=in_channels,
            hidden_channels=self.hidden_dim,
            out_channels=self.embedding_dim,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)
        

        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Self-supervised training with reconstruction loss
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(data.x, data.edge_index)
            
            # Reconstruction loss: try to predict edges from embeddings
            # Sample positive edges
            edge_index = data.edge_index
            num_edges = edge_index.size(1)
            
            # Compute edge predictions using dot product
            row, col = edge_index
            pos_score = (embeddings[row] * embeddings[col]).sum(dim=1)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            
            # Sample negative edges
            neg_row = torch.randint(0, data.num_nodes, (num_edges,), device=self.device)
            neg_col = torch.randint(0, data.num_nodes, (num_edges,), device=self.device)
            neg_score = (embeddings[neg_row] * embeddings[neg_col]).sum(dim=1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
        
        # Extract final embeddings
        model.eval()
        with torch.no_grad():
            final_embeddings = model(data.x, data.edge_index)
            final_embeddings = final_embeddings.cpu().numpy()
        
        # Preprocess final embeddings with StandardScaler
        embedding_scaler = StandardScaler()
        final_embeddings = embedding_scaler.fit_transform(final_embeddings)
        
        # Create mapping from node IDs to embeddings
        nodes = list(graph.nodes())
        self.embeddings = {}
        for i, node in enumerate(nodes):
            self.embeddings[node] = final_embeddings[i]
            if (i + 1) % 10000 == 0:
                print(f"[GATher] Embedded {i + 1} nodes")
        
        print(f"[GATher] Finished embedding all {len(nodes)} nodes")
        
        return self.embeddings
    
    def __str__(self):
        return (f"GATher(dim={self.embedding_dim}, "
                f"hidden={self.hidden_dim}, "
                f"heads={self.heads}, "
                f"epochs={self.epochs})")
