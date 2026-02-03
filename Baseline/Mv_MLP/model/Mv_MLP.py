import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMLPBlock(nn.Module):
    """
    Standard building block for MLP: Linear -> BatchNorm -> GELU -> Dropout.
    Utilizes Kaiming initialization for optimal signal propagation.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Initialization: Kaiming Normal is recommended for ReLU/GELU activations
        nn.init.kaiming_normal_(self.block[0].weight, nonlinearity='relu')

    def forward(self, x):
        return self.block(x)


class MultiViewMLP(nn.Module):
    """
    Multi-view Classification Baseline Model.
    Supports heterogeneous input dimensions and various fusion strategies:

    - 'concat': Concatenates features (preserves information, high dimensionality).
    - 'sum': Element-wise summation (compact representation, assumes complementarity).
    - 'mean': Element-wise averaging (robust against view-specific noise).
    - 'adaptive': Learnable weighted fusion (automatically estimates view importance).
    """

    def __init__(self, input_dims, hidden_dim, num_classes, fusion_strategy, dropout_rate):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.num_views = len(input_dims)

        # 1. View-specific Encoders
        # Projects each view into a shared latent space (hidden_dim)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                StandardMLPBlock(dim, hidden_dim * 2, dropout_rate),
                StandardMLPBlock(hidden_dim * 2, hidden_dim, dropout_rate)
            ) for dim in input_dims
        ])

        # 2. Fusion Parameters
        if self.fusion_strategy == 'adaptive':
            # Initialize weights; normalized via Softmax during forward pass
            self.view_weights = nn.Parameter(torch.ones(self.num_views))

        # 3. Classifier Input Dimension Calculation
        if self.fusion_strategy == 'concat':
            classifier_input_dim = hidden_dim * self.num_views
        else:
            classifier_input_dim = hidden_dim

        # 4. Global Classifier Head
        self.classifier = nn.Sequential(
            StandardMLPBlock(classifier_input_dim, hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, views):
        """
        Forward pass for Multi-view features.

        Args:
            views (list[torch.Tensor]): List of Tensors with shape [batch_size, dim_i]
        Returns:
            torch.Tensor: Logits with shape [batch_size, num_classes]
        """
        # Feature extraction and alignment to common space
        extracted_feats = [encoder(v) for encoder, v in zip(self.encoders, views)]

        # --- Fusion Logic ---
        if self.fusion_strategy == 'concat':
            combined = torch.cat(extracted_feats, dim=1)

        elif self.fusion_strategy == 'sum':
            combined = torch.stack(extracted_feats, dim=0).sum(dim=0)

        elif self.fusion_strategy == 'mean':
            combined = torch.stack(extracted_feats, dim=0).mean(dim=0)

        elif self.fusion_strategy == 'adaptive':
            weights = F.softmax(self.view_weights, dim=0)
            # Weighted summation across views
            combined = sum(weights[i] * extracted_feats[i] for i in range(self.num_views))

        else:
            raise ValueError(f"Unsupported fusion_strategy: {self.fusion_strategy}")

        return self.classifier(combined)

    def get_view_weights(self):
        """
        Retrieves the normalized learned weights for each view (Adaptive strategy only).
        Useful for interpretability analysis.
        """
        if self.fusion_strategy == 'adaptive':
            with torch.no_grad():
                return F.softmax(self.view_weights, dim=0).cpu().numpy()
        return None
