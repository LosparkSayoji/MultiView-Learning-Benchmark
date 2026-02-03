import os
import logging
import scipy.io as scio
import numpy as np
import torch

# Configure Logging for better visibility in experiment tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def mv_load_data(dataset_name: str, data_dir: str):
    """
    Standardized Multi-view data loader for .mat files.

    Loads feature matrices (X) and labels (Y) from a MATLAB file and
    standardizes them for PyTorch compatibility.

    Returns:
        tuple: (features_list, labels_tensor, feature_dims, num_views, num_class)
    """
    logging.info(f"Loading dataset: {dataset_name}")

    # Cross-platform path handling
    file_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    data = scio.loadmat(file_path)

    # 1. Feature Extraction (X)
    # Typical multi-view .mat format stores X as a (1, num_views) cell array
    x_raw = data["X"].flatten()

    features = []
    feature_dims = []

    for i, view_data in enumerate(x_raw):
        # Convert to float32 to optimize memory footprint
        view_data = view_data.astype('float32')

        # Handle specific dataset anomalies (e.g., orientation issues in 100leaves)
        if dataset_name.lower() == '100leaves' and view_data.shape[0] < view_data.shape[1]:
            view_data = view_data.T

        features.append(view_data)
        feature_dims.append(view_data.shape[1])
        logging.info(f"    - View {i}: Feature Dimension = {view_data.shape[1]}")

    # 2. Label Standardization (Y)
    # Ensure labels are 1D, zero-indexed [0, num_class - 1], and typed as Long
    labels_raw = data["Y"].flatten().astype('int64')
    labels_min = labels_raw.min()
    labels_standardized = labels_raw - labels_min

    unique_labels = np.unique(labels_standardized)
    num_class = len(unique_labels)
    labels = torch.from_numpy(labels_standardized)

    num_views = len(features)
    logging.info(f"Summary: Views={num_views}, Classes={num_class}, Samples={len(labels)}")

    return features, labels, feature_dims, num_views, num_class


def mv_generate_permutation(gnd, args):
    """
    Generates stratified split indices for Training, Validation, and Testing.
    Ensures each class is represented according to the defined ratios.
    """
    # Use seed for reproducibility
    rng = np.random.RandomState(args.seed)

    # Ensure ground truth is a numpy array for slicing
    gnd = gnd.cpu().numpy().astype(int) if torch.is_tensor(gnd) else gnd.astype(int)
    unique_labels = np.unique(gnd)

    # Fetch split ratios from configuration
    r_train = args.train_ratio
    r_val = args.val_ratio
    r_test = getattr(args, 'test_ratio', 1.0 - r_train - r_val)

    train_idx, val_idx, test_idx = [], [], []

    # ====== Stratified Per-Class Sampling ======
    for label in unique_labels:
        # Extract indices belonging to the current class and shuffle
        label_indices = np.where(gnd == label)[0]
        rng.shuffle(label_indices)

        n_class = len(label_indices)

        # Calculate split sizes
        n_train = max(1, round(n_class * r_train))
        n_val = max(1, round(n_class * r_val))

        # Defensive Check: Ensure ratios do not exceed class total
        if n_train + n_val >= n_class:
            n_train = max(1, n_class - 2)
            n_val = 1

        # Partition indices
        train_idx.extend(label_indices[:n_train])
        val_idx.extend(label_indices[n_train: n_train + n_val])
        test_idx.extend(label_indices[n_train + n_val:])

    # ====== Conversion to Tensor & Final Global Shuffling ======
    # Global shuffling prevents any unintended bias from label-sequential addition
    train_idx = torch.from_numpy(rng.permutation(np.array(train_idx, dtype=np.int64)))
    val_idx = torch.from_numpy(rng.permutation(np.array(val_idx, dtype=np.int64)))
    test_idx = torch.from_numpy(rng.permutation(np.array(test_idx, dtype=np.int64)))

    return train_idx, val_idx, test_idx
