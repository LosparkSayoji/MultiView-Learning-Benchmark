import argparse

def parameter_parser():
    """
    Standardized Argument Parser for the Multi-view Transformer Framework.
    Features categorized groups for architectural, experimental, and hardware settings.
    """
    parser = argparse.ArgumentParser(description="Multi-view Learning Baseline (Transformer Version)")

    # --- Experimental Foundation ---
    exp_group = parser.add_argument_group('Experimental Settings')
    exp_group.add_argument('--dataset', type=str, default="flower17",
                           help='Target dataset name (e.g., flower17, 100leaves)')
    exp_group.add_argument('--rep_num', type=int, default=5,
                           help='Number of independent experimental repetitions')
    exp_group.add_argument('--res_path', type=str, default="./results/",
                           help='Path to store performance logs and result files')

    # --- Core Optimization Hyperparameters ---
    opt_group = parser.add_argument_group('Optimization Settings')
    opt_group.add_argument('--num_epochs', type=int, default=200,
                           help='Total number of training iterations')
    opt_group.add_argument('--lr', type=float, default=1e-3,
                           help='Initial learning rate')
    opt_group.add_argument('--weight_decay', type=float, default=5e-4,
                           help='L2 regularization (Weight Decay) coefficient')
    opt_group.add_argument('--dropout_rate', type=float, default=0.2,
                           help='Probability of dropout for preventing overfitting')
    opt_group.add_argument('--hid_dim', type=int, default=256,
                           help='d_model: Dimensionality of the latent alignment space')
    opt_group.add_argument('--fusion_strategy', type=str, default='CLS Token',
                           choices=['concat', 'sum', 'mean', 'adaptive', 'CLS Token'],
                           help='Strategy used to integrate cross-view features')

    # --- Transformer Architecture Specifics ---
    tf_group = parser.add_argument_group('Transformer Architecture')
    tf_group.add_argument('--nhead', type=int, default=4,
                          help='Number of heads in the multi-head attention mechanism')
    tf_group.add_argument('--num_layers', type=int, default=2,
                          help='Number of Transformer Encoder layers')
    tf_group.add_argument('--dim_feedforward', type=int, default=512,
                          help='Dimensionality of the feed-forward network (usually 2x-4x hid_dim)')

    # --- Data Splitting & Hardware ---
    hw_group = parser.add_argument_group('Data & Hardware Configuration')
    hw_group.add_argument('--train_ratio', type=float, default=0.1, help='Ratio for training set')
    hw_group.add_argument('--val_ratio', type=float, default=0.1, help='Ratio for validation set')
    hw_group.add_argument('--test_ratio', type=float, default=0.8, help='Ratio for testing set')
    hw_group.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
    hw_group.add_argument('--device', type=str, default="0", help='Target GPU ID or "cpu"')
    hw_group.add_argument('--fastmode', action='store_true', default=False,
                          help='Enable fast mode to reduce validation frequency')

    return parser.parse_args()
