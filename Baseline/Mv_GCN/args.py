import argparse

def parameter_parser():
    """
    Comprehensive Argument Parser for Multi-view GCN experiments.
    Organizes parameters into logical groups for better CLI readability.
    """
    parser = argparse.ArgumentParser(description="Multi-view Learning Baseline (GCN Version)")

    # --- Experiment Metadata ---
    exp_group = parser.add_argument_group('Experimental Settings')
    exp_group.add_argument('--dataset', type=str, default="flower17",
                           help='Name of the target dataset')
    exp_group.add_argument('--rep_num', type=int, default=5,
                           help='Number of independent experimental trials')
    exp_group.add_argument('--res_path', type=str, default="./results/",
                           help='Base directory for saving experimental logs')

    # --- Model Hyperparameters ---
    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--num_epochs', type=int, default=200,
                             help='Total number of training epochs')
    model_group.add_argument('--lr', type=float, default=1e-3,
                             help='Learning rate for Adam optimizer')
    model_group.add_argument('--weight_decay', type=float, default=5e-4,
                             help='L2 regularization coefficient (Weight Decay)')
    model_group.add_argument('--dropout_rate', type=float, default=0.2,
                             help='Dropout probability for regularization')
    model_group.add_argument('--hid_dim', type=int, default=256,
                             help='Dimensionality of the shared latent space')
    model_group.add_argument('--fusion_strategy', type=str, default='adaptive',
                             choices=['concat', 'sum', 'mean', 'adaptive'],
                             help='Feature fusion methodology')

    # --- GNN Specific Settings ---
    gnn_group = parser.add_argument_group('GNN Specific Settings')
    gnn_group.add_argument('--k', type=int, default=10,
                           help='Number of neighbors for k-NN graph construction')

    # --- Data Partitioning & Hardware ---
    hw_group = parser.add_argument_group('Data & Hardware Settings')
    hw_group.add_argument('--train_ratio', type=float, default=0.1, help='Ratio of training samples')
    hw_group.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation samples')
    hw_group.add_argument('--test_ratio', type=float, default=0.8, help='Ratio of test samples')
    hw_group.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
    hw_group.add_argument('--device', type=str, default="0", help='Target GPU index (e.g., "0", "1") or "cpu"')
    hw_group.add_argument('--fastmode', action='store_true', default=False,
                          help='Enable fast mode: simplify validation frequency during training')

    return parser.parse_args()
