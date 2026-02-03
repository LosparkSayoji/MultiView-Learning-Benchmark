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
from model.Mv_MLP import MultiViewMLP


def calculate_metrics(logits, labels):
    """
    Computes standard classification metrics: Accuracy and Macro-F1.
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
    Standardized training pipeline with feature normalization and model persistence.

    This function handles:
    1. Feature scaling (preventing data leakage).
    2. Model initialization and seed synchronization.
    3. Optimization loop with validation-based model selection.
    """

    # 1. Feature Preprocessing (Strictly fit on training set only to prevent leakage)
    processed_features = []
    train_idx_np = train_idx.cpu().numpy() if torch.is_tensor(train_idx) else train_idx

    for f in features:
        scaler = StandardScaler()
        # Fit on training data only, then transform the entire feature set
        scaler.fit(f[train_idx_np])
        f_scaled = scaler.transform(f)
        processed_features.append(torch.FloatTensor(f_scaled).to(device))

    labels = labels.to(device)

    # Synchronize Random Seeds for Reproducibility
    if args.rep_num > 1:
        args.seed = np.random.randint(100000)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 2. Model Initialization
    model = MultiViewMLP(
        input_dims=feature_dims,
        hidden_dim=args.hid_dim,
        num_classes=num_class,
        fusion_strategy=args.fusion_strategy,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Parameter Statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized: Total Parameters = {total_params / 1e6:.4f}M")

    # 3. Optimization Components
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
    pbar = tqdm(range(args.num_epochs), desc='Training Pipeline')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Full-batch forward pass
        logits = model(processed_features)
        loss_train = criterion(logits[train_idx], labels[train_idx])

        loss_train.backward()
        optimizer.step()

        # Evaluation Protocol
        model.eval()
        with torch.no_grad():
            # Re-compute logits for consistent Dropout/BN behavior during inference
            eval_logits = model(processed_features)

            acc_train, _ = calculate_metrics(eval_logits[train_idx], labels[train_idx])
            acc_val, f1_val = calculate_metrics(eval_logits[val_idx], labels[val_idx])

            # Persistence Logic: Save the best weights based on validation accuracy
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # Update Progress Bar with key metrics
        pbar.set_postfix({
            'Loss': f"{loss_train.item():.3f}",
            'Tr_Acc': f"{acc_train:.2f}",
            'Val_Acc': f"{acc_val:.2f}",
            'Best_Va': f"{best_val_acc:.2f}"
        })

    # Restore the best model state
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, processed_features, labels, best_epoch


def test(model, features, labels, test_idx):
    """
    Final evaluation on the held-out test set.
    """
    model.eval()
    with torch.no_grad():
        logits = model(features)
        acc, f1 = calculate_metrics(logits[test_idx], labels[test_idx])

    # Visual Reporting
    print(f"\n" + "-" * 40)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Test Macro-F1: {f1 * 100:.2f}%")

    # Optional: Display View Weights if using Attention-based fusion
    if hasattr(model, 'get_view_weights'):
        weights = model.get_view_weights()
        if weights is not None:
            formatted_weights = " | ".join([f"V{i}: {w:.3f}" for i, w in enumerate(weights)])
            print(f"View Weights  : {formatted_weights}")
    print("-" * 40 + "\n")

    return acc, f1
