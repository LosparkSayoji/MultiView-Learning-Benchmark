import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewTransformer(nn.Module):
    """
    Multi-view Transformer Architecture.

    Supports two main fusion paradigms:
    1. 'CLS Token': Deep cross-view interaction via a unified global transformer.
    2. Late Fusion: Independent view encoding followed by Concat/Sum/Mean/Adaptive fusion.
    """

    def __init__(self, input_dims, d_model, nhead, num_layers,
                 dim_feedforward, dropout_rate, num_classes, fusion_strategy):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.num_views = len(input_dims)
        self.d_model = d_model

        # 1. Input Projections: Map heterogeneous features to a common latent dimension
        self.input_projections = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in input_dims
        ])

        # 2. Transformer Backbone Configuration
        # norm_first=True is used for improved training stability (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        if self.fusion_strategy == 'CLS Token':
            # Paradigm A: Deep Cross-View Fusion
            # A single powerful encoder processes the sequence of all view features
            self.global_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.global_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        else:
            # Paradigm B: Independent View Encoding
            # Each view has its own encoder and local CLS token
            self.view_encoders = nn.ModuleList([
                nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                for _ in range(self.num_views)
            ])
            self.view_cls_tokens = nn.ParameterList([
                nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(self.num_views)
            ])

        # 3. Fusion Parameters (Only for 'adaptive' strategy)
        if self.fusion_strategy == 'adaptive':
            self.view_weights = nn.Parameter(torch.ones(self.num_views))

        # 4. Classification Head
        # Adjust input dimension based on fusion strategy
        classifier_in_dim = d_model * self.num_views if self.fusion_strategy == 'concat' else d_model
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, views):
        batch_size = views[0].size(0)

        # --- Strategy 1: Global CLS Token (Deep Cross-View Interaction) ---
        if self.fusion_strategy == 'CLS Token':
            # Project views and concatenate into a sequence: [Batch, num_views, d_model]
            view_features = [self.input_projections[i](views[i]).unsqueeze(1) for i in range(self.num_views)]
            all_views_seq = torch.cat(view_features, dim=1)

            # Prepend the Global CLS Token: [Batch, num_views + 1, d_model]
            global_cls = self.global_cls_token.expand(batch_size, -1, -1)
            full_seq = torch.cat((global_cls, all_views_seq), dim=1)

            # Global self-attention across all view tokens
            output_seq = self.global_encoder(full_seq)

            # Extract the refined global representation from the CLS position
            combined = output_seq[:, 0, :]

        # --- Strategy 2: Late Fusion (Post-Encoding Integration) ---
        else:
            encoded_feats = []
            for i in range(self.num_views):
                # Local encoding per view
                x = self.input_projections[i](views[i]).unsqueeze(1)
                cls = self.view_cls_tokens[i].expand(batch_size, -1, -1)

                # Concatenate local CLS token with the view feature
                x = torch.cat((cls, x), dim=1)
                x = self.view_encoders[i](x)

                # Extract local representation
                encoded_feats.append(x[:, 0, :])

            # Apply integration logic
            if self.fusion_strategy == 'concat':
                combined = torch.cat(encoded_feats, dim=1)
            elif self.fusion_strategy == 'sum':
                combined = torch.stack(encoded_feats, dim=0).sum(dim=0)
            elif self.fusion_strategy == 'mean':
                combined = torch.stack(encoded_feats, dim=0).mean(dim=0)
            elif self.fusion_strategy == 'adaptive':
                weights = F.softmax(self.view_weights, dim=0)
                combined = sum(w * f for w, f in zip(weights, encoded_feats))

        return self.classifier(combined)

    def get_view_weights(self):
        """Returns softmax-normalized view weights for the adaptive strategy."""
        if self.fusion_strategy == 'adaptive':
            return F.softmax(self.view_weights, dim=0).detach().cpu().numpy()
        return None