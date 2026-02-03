import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Custom modules
from model.Mv_GCN import MultiViewGCN


def calculate_metrics(logits, labels):
    """
    Computes standard classification metrics: Accuracy and Macro-F1.

    Args:
        logits (torch.Tensor): Model output logits.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        tuple: (accuracy, macro_f1)
    """
    if logits.numel() == 0:
        return 0.0, 0.0

    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = labels.cpu().numpy()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return acc, f1


def train(processed_features, adjs, num_class, feature_dims, labels, train_idx, val_idx, args, device):
    """
    Standardized training pipeline for Multi-view GCN.
    Includes reproducibility setup, full-batch GNN training, and model persistence.
    """
    labels = labels.to(device)

    # --- Reproducibility Setup ---
    # Reseed if multiple repetitions are required to ensure statistical variance
    if args.rep_num > 1:
        args.seed = np.random.randint(100000)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # --- Model Initialization ---
    model = MultiViewGCN(
        input_dims=feature_dims,
        hidden_dim=args.hid_dim,
        num_classes=num_class,
        fusion_strategy=args.fusion_strategy,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Log model complexity
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized | Total Parameters: {total_params / 1e6:.4f}M")

    # --- Optimizer and Loss Configuration ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_model_wts = None

    # --- Training Loop ---
    pbar = tqdm(range(args.num_epochs), desc='GNN Training')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Full-batch forward pass for graph-structured data
        logits = model(processed_features, adjs)
        loss_train = criterion(logits[train_idx], labels[train_idx])

        loss_train.backward()
        optimizer.step()

        # --- Evaluation Protocol ---
        model.eval()
        with torch.no_grad():
            # Re-calculating logits for consistent inference behavior
            eval_logits = model(processed_features, adjs)

            acc_train, _ = calculate_metrics(eval_logits[train_idx], labels[train_idx])
            acc_val, f1_val = calculate_metrics(eval_logits[val_idx], labels[val_idx])

            # Validation-based model selection (Best Weight Persistence)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # Update progress bar with current batch statistics
        pbar.set_postfix({
            'Loss': f"{loss_train.item():.3f}",
            'Tr_Acc': f"{acc_train:.2f}",
            'Val_Acc': f"{acc_val:.2f}",
            'Best_Va': f"{best_val_acc:.2f}"
        })

    # Restore optimized weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, labels, best_epoch


def test(model, processed_features, adjs, labels, test_idx, args, device):
    """
    Final performance evaluation on the test set split.
    """
    model.eval()
    with torch.no_grad():
        logits = model(processed_features, adjs)
        acc, f1 = calculate_metrics(logits[test_idx], labels[test_idx])

    # Visual Reporting Header
    print(f"\n" + "-" * 80)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Test Macro-F1: {f1 * 100:.2f}%")

    # Interpretability: Display view-wise contribution weights if available
    if hasattr(model, 'get_view_weights'):
        weights = model.get_view_weights()
        if weights is not None:
            formatted_weights = "  |  ".join([f"View {i}: {w:.3f}" for i, w in enumerate(weights)])
            print(f"View Contribution Weights: {formatted_weights}")
    print("-" * 80 + "\n")

    return acc, f1
