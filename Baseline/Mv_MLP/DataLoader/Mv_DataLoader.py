import os
import logging
import scipy.io as scio
import numpy as np
import torch
from typing import List, Tuple, Union

# Configure logging for better visibility during data preprocessing
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def mv_load_data(dataset_name: str, data_dir: str) -> Tuple[List[np.ndarray], torch.Tensor, List[int], int, int]:
    """
    Standardized multi-view data loader for MATLAB (.mat) files.

    Expected file structure:
    - 'X': A (1, num_views) cell array where each cell is (num_samples, num_features).
    - 'Y': A (num_samples, 1) or (num_samples,) array of ground truth labels.
    """
    logging.info(f"==> Loading dataset: {dataset_name}")

    file_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Load MATLAB dictionary
    data = scio.loadmat(file_path)

    # 1. Feature Extraction (X)
    # Flatten the cell array to handle various MATLAB nesting styles
    x_raw = data["X"].flatten()

    features = []
    feature_dims = []

    for i, view_data in enumerate(x_raw):
        # Convert to float32 for memory efficiency and PyTorch compatibility
        view_data = view_data.astype('float32')
        features.append(view_data)
        feature_dims.append(view_data.shape[1])
        logging.info(f"    View {i}: Shape {view_data.shape}")

    # 2. Label Processing (Y)
    # Standardize labels to [0, num_class - 1] to prevent index errors
    labels_raw = data["Y"].flatten().astype('int64')
    labels_min = labels_raw.min()
    labels_standardized = labels_raw - labels_min

    num_class = len(np.unique(labels_standardized))
    labels = torch.from_numpy(labels_standardized)
    num_views = len(features)

    logging.info(f"==> Load Complete: Views={num_views}, Dims={feature_dims}, Classes={num_class}")

    return features, labels, feature_dims, num_views, num_class


def mv_generate_permutation(labels: Union[torch.Tensor, np.ndarray], args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates stratified train/val/test indices to ensure class balance across splits.

    Args:
        labels: Ground truth labels (Tensor or ndarray).
        args: Argument parser containing 'seed', 'train_ratio', and 'val_ratio'.

    Returns:
        Indices for train, validation, and test sets as LongTensors.
    """
    rng = np.random.RandomState(args.seed)

    # Convert to numpy for indexing consistency
    y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    unique_labels = np.unique(y)

    # Retrieve split ratios from args
    r_train = args.train_ratio
    r_val = args.val_ratio
    # Default test ratio is the remainder
    r_test = getattr(args, 'test_ratio', 1.0 - r_train - r_val)

    train_idx, val_idx, test_idx = [], [], []

    # ====== Stratified Sampling per Class ======
    for label in unique_labels:
        # Get and shuffle indices for the current class
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        n_total = len(label_indices)

        # Calculate split sizes (Ensure at least 1 sample per split if total > 2)
        n_train = max(1, round(n_total * r_train))
        n_val = max(1, round(n_total * r_val))

        # Boundary check: Ensure we don't exceed total class samples
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1

        # Slice indices
        train_idx.extend(label_indices[:n_train])
        val_idx.extend(label_indices[n_train: n_train + n_val])
        test_idx.extend(label_indices[n_train + n_val:])

    # ====== Final Global Shuffle and Tensor Conversion ======
    # Shuffle the combined indices to avoid class-sequential ordering
    train_idx = torch.LongTensor(rng.permutation(np.array(train_idx)))
    val_idx = torch.LongTensor(rng.permutation(np.array(val_idx)))
    test_idx = torch.LongTensor(rng.permutation(np.array(test_idx)))

    return train_idx, val_idx, test_idx
