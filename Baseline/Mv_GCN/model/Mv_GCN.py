import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    High-performance Graph Convolutional Layer.
    Flow: Linear Transformation -> Graph Propagation -> BatchNorm -> Activation -> Dropout.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        """
        Performs graph convolution propagation:
        $$H = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W)$$

        Args:
            x (torch.Tensor): Node feature matrix.
            adj (torch.Tensor): Sparse or dense normalized adjacency matrix.
        """
        # Linear transformation (W)
        support = self.linear(x)

        # Feature propagation using sparse matrix multiplication (spmm) for efficiency
        if adj.is_sparse:
            out = torch.spmm(adj, support)
        else:
            out = torch.matmul(adj, support)

        return self.dropout(self.activation(self.bn(out)))


class MultiViewGCN(nn.Module):
    """
    Multi-view Graph Convolutional Network.
    Extracts structural features from multiple views via view-specific GCN encoders
    and fuses them using various integration strategies.
    """

    def __init__(self, input_dims, hidden_dim, num_classes, fusion_strategy, dropout_rate):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.num_views = len(input_dims)

        # 1. Feature Projection: Maps heterogeneous raw features to a unified latent space
        self.input_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])

        # 2. Intra-view GCN Encoders: Deep structural feature extraction for each view
        self.view_encoders = nn.ModuleList([
            nn.ModuleList([
                GCNLayer(hidden_dim, hidden_dim, dropout_rate),
                GCNLayer(hidden_dim, hidden_dim, dropout_rate)
            ]) for _ in range(self.num_views)
        ])

        # 3. Adaptive Weighting Parameters (Enabled only for 'adaptive' strategy)
        if self.fusion_strategy == 'adaptive':
            self.view_weights = nn.Parameter(torch.ones(self.num_views))

        # 4. Classification Head: Final prediction via non-linear mapping
        classifier_in_dim = hidden_dim * self.num_views if fusion_strategy == 'concat' else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, views, adjs):
        """
        Args:
            views (list[torch.Tensor]): List of feature tensors for each view.
            adjs (list or torch.Tensor): Adjacency matrices. Supports view-specific
                                         list or a single shared graph.
        """
        encoded_feats = []

        for i in range(self.num_views):
            # Map input to common hidden dimension
            x = self.input_projections[i](views[i])
            curr_adj = adjs[i] if isinstance(adjs, list) else adjs

            # Sequential Message Passing through GCN layers
            for layer in self.view_encoders[i]:
                x = layer(x, curr_adj)

            encoded_feats.append(x)

        # 5. Fusion Module
        if self.fusion_strategy == 'concat':
            combined = torch.cat(encoded_feats, dim=1)
        elif self.fusion_strategy == 'sum':
            combined = torch.stack(encoded_feats, dim=0).sum(dim=0)
        elif self.fusion_strategy == 'mean':
            combined = torch.stack(encoded_feats, dim=0).mean(dim=0)
        elif self.fusion_strategy == 'adaptive':
            weights = F.softmax(self.view_weights, dim=0)
            combined = sum(weights[i] * encoded_feats[i] for i in range(self.num_views))
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        return self.classifier(combined)

    def get_view_weights(self):
        """
        Extracts view-wise importance scores for model interpretability.
        """
        if self.fusion_strategy == 'adaptive':
            return F.softmax(self.view_weights, dim=0).detach().cpu().numpy()
        return None
