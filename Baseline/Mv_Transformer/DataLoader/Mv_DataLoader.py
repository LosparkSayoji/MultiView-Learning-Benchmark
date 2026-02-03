import os
import logging
import scipy.io as scio
import numpy as np
import torch

# Configure standardized logging format
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def mv_load_data(dataset_name: str, data_dir: str):
    """
    Standardized multi-view data loader for .mat files.

    Parses feature matrices (X) and ground truth labels (Y), handles
    view-specific orientation issues, and standardizes class labels.

    Args:
        dataset_name (str): The name of the dataset to load.
        data_dir (str): Directory where the .mat file is stored.

    Returns:
        tuple: (features_list, labels_tensor, feature_dims, num_views, num_class)
    """
    logging.info(f"Initiating data loading: {dataset_name}")

    # Cross-platform path construction
    file_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset file at: {file_path}")

    data = scio.loadmat(file_path)

    # 1. Feature Extraction (X)
    # Extracts multi-view data usually stored as a (1, num_views) cell array
    x_raw = data["X"].flatten()

    features = []
    feature_dims = []

    for i, view_data in enumerate(x_raw):
        # Convert to float32 to reduce memory overhead during training
        view_data = view_data.astype('float32')

        # Dataset-specific preprocessing: Ensure samples are in rows, features in columns
        if dataset_name.lower() == '100leaves' and view_data.shape[0] < view_data.shape[1]:
            view_data = view_data.T

        features.append(view_data)
        feature_dims.append(view_data.shape[1])
        logging.info(f"    - View {i} identified: Shape {view_data.shape}")

    # 2. Label Processing (Y)
    # Standardize labels to a zero-indexed 1D LongTensor [0, num_class - 1]
    labels_raw = data["Y"].flatten().astype('int64')
    labels_min = labels_raw.min()
    labels_standardized = labels_raw - labels_min

    unique_labels = np.unique(labels_standardized)
    num_class = len(unique_labels)
    labels = torch.from_numpy(labels_standardized)

    num_views = len(features)

    logging.info(f"Loading complete | Views: {num_views} | Samples: {len(labels)} | Classes: {num_class}")

    return features, labels, feature_dims, num_views, num_class


def mv_generate_permutation(gnd, args):
    """
    Generates per-class stratified split indices for Train/Val/Test sets.
    Ensures that the class distribution is preserved across all partitions.
    """
    # Initialize Random State for reproducibility
    rng = np.random.RandomState(args.seed)

    # Standardize ground truth to numpy for indexing operations
    gnd = gnd.cpu().numpy().astype(int) if torch.is_tensor(gnd) else gnd.astype(int)
    unique_labels = np.unique(gnd)

    # Retrieve split ratios from arguments
    r_train = args.train_ratio
    r_val = args.val_ratio
    r_test = getattr(args, 'test_ratio', 1.0 - r_train - r_val)

    train_idx, val_idx, test_idx = [], [], []

    # ====== Stratified Per-Class Sampling ======
    # This prevents class imbalance in small training sets
    for label in unique_labels:
        # Retrieve and shuffle indices belonging to the current class
        label_indices = np.where(gnd == label)[0]
        rng.shuffle(label_indices)

        n_class = len(label_indices)

        # Calculate absolute split counts
        n_train = max(1, round(n_class * r_train))
        n_val = max(1, round(n_class * r_val))

        # Boundary logic: Ensure the split does not exceed the total class sample size
        if n_train + n_val >= n_class:
            n_train = max(1, n_class - 2)
            n_val = 1

        # Distribute indices to respective sets
        train_idx.extend(label_indices[:n_train])
        val_idx.extend(label_indices[n_train: n_train + n_val])
        test_idx.extend(label_indices[n_train + n_val:])

    # ====== Final Global Shuffle and Tensor Conversion ======
    # Shuffling after extension ensures that samples from the same class are not adjacent
    train_idx = torch.from_numpy(rng.permutation(np.array(train_idx, dtype=np.int64)))
    val_idx = torch.from_numpy(rng.permutation(np.array(val_idx, dtype=np.int64)))
    test_idx = torch.from_numpy(rng.permutation(np.array(test_idx, dtype=np.int64)))

    return train_idx, val_idx, test_idx
