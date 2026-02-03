import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Custom modules
from model.Mv_Transformer import MultiViewTransformer


def calculate_metrics(logits, labels):
    """
    Computes standard classification metrics for multi-class tasks.

    Args:
        logits (torch.Tensor): Raw model outputs.
        labels (torch.Tensor): Target labels.

    Returns:
        tuple: (Accuracy, Macro-F1 Score)
    """
    if logits.numel() == 0:
        return 0.0, 0.0

    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = labels.cpu().numpy()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return acc, f1


def train(features, num_class, feature_dims, labels, train_idx, val_idx, args, device):
    """
    Standardized training pipeline for Multi-view Transformer.
    Implements strict feature scaling and validation-based model checkpointing.
    """

    # --- 1. Robust Preprocessing (Prevention of Data Leakage) ---
    processed_features = []
    # Convert index to numpy for sklearn compatibility
    train_idx_np = train_idx.cpu().numpy() if torch.is_tensor(train_idx) else train_idx

    for f in features:
        scaler = StandardScaler()
        # Fit scaler ONLY on training data to maintain experimental integrity
        scaler.fit(f[train_idx_np])
        # Transform the entire dataset using training statistics
        f_scaled = scaler.transform(f)
        processed_features.append(torch.FloatTensor(f_scaled).to(device))

    labels = labels.to(device)

    # --- 2. Reproducibility Configuration ---
    if args.rep_num > 1:
        args.seed = np.random.randint(100000)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # --- 3. Transformer Model Initialization ---
    model = MultiViewTransformer(
        input_dims=feature_dims,
        d_model=args.hid_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout_rate=args.dropout_rate,
        num_classes=num_class,
        fusion_strategy=args.fusion_strategy
    ).to(device)

    # Log model complexity (important for Transformer-based architectures)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized | Total Parameters: {total_params / 1e6:.4f}M")

    # --- 4. Optimization Suite ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_model_wts = None

    # --- 5. Training Loop ---
    pbar = tqdm(range(args.num_epochs), desc='Transformer Training')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass on full-batch features
        logits = model(processed_features)
        loss_train = criterion(logits[train_idx], labels[train_idx])

        loss_train.backward()
        optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            # Re-calculating logits to ensure correct Dropout/BatchNorm behavior in eval mode
            eval_logits = model(processed_features)

            acc_train, _ = calculate_metrics(eval_logits[train_idx], labels[train_idx])
            acc_val, f1_val = calculate_metrics(eval_logits[val_idx], labels[val_idx])

            # Checkpoint the best model based on validation accuracy
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # Update progress bar with performance metrics
        pbar.set_postfix({
            'Tr_Loss': f"{loss_train.item():.3f}",
            'Tr_Acc': f"{acc_train:.2f}",
            'Val_Acc': f"{acc_val:.2f}",
            'Best_Val': f"{best_val_acc:.2f}"
        })

    # Restore optimized parameters
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, processed_features, labels, best_epoch


def test(model, features, labels, test_idx):
    """
    Final performance evaluation on the held-out test set.
    """
    model.eval()
    with torch.no_grad():
        logits = model(features)
        acc, f1 = calculate_metrics(logits[test_idx], labels[test_idx])

    # Visual Reporting
    print(f"\n" + "-" * 80)
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    print(f"Final Test Macro-F1: {f1 * 100:.2f}%")

    # Interpretability: Extract view importance if the adaptive strategy is used
    if hasattr(model, 'get_view_weights'):
        weights = model.get_view_weights()
        if weights is not None:
            formatted_weights = "  |  ".join([f"View {i}: {w:.3f}" for i, w in enumerate(weights)])
            print(f"Learned View Weights: {formatted_weights}")
    print("-" * 80 + "\n")

    return acc, f1
