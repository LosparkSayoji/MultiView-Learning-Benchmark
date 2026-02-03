import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def normalize_laplacian(adj):
    """
    Performs standard symmetric normalization: D^-1/2 * A * D^-1/2.
    This normalization is essential for GCNs to prevent exploding/vanishing gradients.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    # Calculate D^-1/2
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Compute D^-1/2 * A * D^-1/2
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Converts a Scipy sparse matrix into a torch.sparse_coo_tensor.
    Utilizing sparse tensors significantly reduces GPU memory consumption for large graphs.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def get_normalized_adj(features, k, device):
    """
    Constructs k-Nearest Neighbor (k-NN) graphs and computes their normalized adjs.

    Args:
        features (list): List of tensors/arrays containing features for each view.
        k (int): Number of neighbors for graph construction.
        device: The target device (CPU/CUDA).

    Returns:
        list: List of PyTorch sparse tensors (normalized adjacency matrices).
    """
    adj_list = []

    for f in features:
        # 1. Construct k-NN Graph (Connectivity mode provides a 0-1 binary matrix)
        # include_self=True adds self-loops (A = A + I), allowing nodes to aggregate their own features
        f_np = f.cpu().numpy() if torch.is_tensor(f) else f
        adj = kneighbors_graph(f_np, n_neighbors=k, mode='connectivity', include_self=True)

        # 2. Symmetrization: A = max(A, A^T)
        # Ensures undirected graph structure, improving structural robustness
        adj = adj.maximum(adj.T)

        # 3. Symmetric Laplacian Normalization

        adj_norm = normalize_laplacian(adj)

        # 4. Conversion to Sparse Torch Tensors
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
        adj_list.append(adj_tensor)

    return adj_list


def prepare_graph_data(features, train_idx, k, device):
    """
    End-to-end pipeline: Feature scaling followed by normalized adjacency matrix generation.

    Args:
        features (list): Raw multi-view feature matrices.
        train_idx (tensor/array): Training set indices for fit-transform logic.
        k (int): Number of neighbors for k-NN graph.
        device: Execution device.

    Returns:
        tuple: (processed_features_tensors, normalized_adj_sparse_tensors)
    """
    processed_features = []
    t_idx = train_idx.cpu().numpy() if torch.is_tensor(train_idx) else train_idx

    # 1. Feature Standardization (Preventing Data Leakage)
    for f in features:
        scaler = StandardScaler()
        # Strictly fit on the training set indices only
        scaler.fit(f[t_idx])
        # Apply the learned transformation to the entire dataset
        f_scaled = scaler.transform(f)
        processed_features.append(torch.FloatTensor(f_scaled).to(device))

    # 2. Adjacency Matrix Construction from Scaled Features
    # Graph structures are built using normalized features to improve similarity metrics
    adjs = get_normalized_adj(processed_features, k, device)

    return processed_features, adjs
