"""
GraphSAGE (Graph Sample and Aggregate) embedding implementation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
from networkx.algorithms import centrality
from sklearn.preprocessing import StandardScaler
from embeddings.base_embedder import BaseEmbedder


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for node embedding.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer without activation
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGEEmbedder(BaseEmbedder):
    """
    GraphSAGE implementation for graph embedding using sampling and aggregation.
    """
    
    def __init__(self, embedding_dim=32, hidden_dim=64, num_layers=2,
                 epochs=200, lr=0.01, weight_decay=5e-4, dropout=0.5,
                 random_state=42):
        """
        Initialize GraphSAGE embedder.
        
        Args:
            embedding_dim (int): Dimension of output embedding vectors
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of GraphSAGE layers
            epochs (int): Number of training epochs
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            dropout (float): Dropout rate
            random_state (int): Random seed
        """
        super().__init__(embedding_dim, random_state)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        
        # Always compute and add centrality measures as features
        num_nodes = data.num_nodes
        
        # Compute centrality measures
        degree_dict = {node: graph.degree(node) for node in graph.nodes()}
        betweenness_dict = centrality.betweenness_centrality(graph)
        closeness_dict = centrality.closeness_centrality(graph)
        
        # Create centrality feature matrix
        centrality_features = torch.zeros(num_nodes, 3)
        nodes_list = list(graph.nodes())
        for i, node in enumerate(nodes_list):
            centrality_features[i, 0] = degree_dict[node]
            centrality_features[i, 1] = betweenness_dict[node]
            centrality_features[i, 2] = closeness_dict[node]
        
        # Normalize each centrality feature independently
        for j in range(3):
            max_val = centrality_features[:, j].max()
            if max_val > 0:
                centrality_features[:, j] = centrality_features[:, j] / max_val
        
        # Combine with existing features or use centrality features alone
        if hasattr(data, 'x') and data.x is not None:
            data.x = torch.cat([data.x, centrality_features], dim=1)
        else:
            data.x = centrality_features
        
        # Preprocess features with StandardScaler
        if data.x is not None and data.x.numel() > 0:
            self.scaler = StandardScaler()
            data.x = torch.FloatTensor(self.scaler.fit_transform(data.x.cpu().numpy()))
        
        return data.to(self.device)
    
    def generate_embeddings(self, graph):
        """
        Generate GraphSAGE embeddings for all nodes in the graph.
        
        Args:
            graph (networkx.Graph): Input graph
            
        Returns:
            dict: Dictionary mapping node IDs to embedding vectors
        """
        # Prepare data
        data = self._prepare_data(graph)
        in_channels = data.x.size(1)
        
        # Initialize model
        model = GraphSAGEModel(
            in_channels=in_channels,
            hidden_channels=self.hidden_dim,
            out_channels=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Self-supervised training with link prediction loss
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(data.x, data.edge_index)
            
            # Link prediction loss: predict edges from embeddings
            edge_index = data.edge_index
            num_edges = edge_index.size(1)
            
            # Positive edges (existing edges)
            row, col = edge_index
            pos_score = (embeddings[row] * embeddings[col]).sum(dim=1)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            
            # Negative edges (random non-edges)
            neg_row = torch.randint(0, data.num_nodes, (num_edges,), device=self.device)
            neg_col = torch.randint(0, data.num_nodes, (num_edges,), device=self.device)
            neg_score = (embeddings[neg_row] * embeddings[neg_col]).sum(dim=1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            
            # Total loss
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
        self.embeddings = {node: final_embeddings[i] for i, node in enumerate(nodes)}
        
        return self.embeddings
    
    def __str__(self):
        return (f"GraphSAGE(dim={self.embedding_dim}, "
                f"hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, "
                f"epochs={self.epochs})")