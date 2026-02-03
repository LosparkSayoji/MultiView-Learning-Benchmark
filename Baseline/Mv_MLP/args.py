import argparse


def parameter_parser():
    """
    Standardized Argument Parser for Multi-view Learning Experiments.
    Categorizes parameters into Experiment, Model, Data, and Hardware settings.
    """
    parser = argparse.ArgumentParser(description="Multi-view Learning Baseline Framework")

    # --- Experiment Configurations ---
    group_exp = parser.add_argument_group('Experiment Settings')
    group_exp.add_argument('--dataset', type=str, default="flower17",
                           help='Target dataset name (e.g., 100leaves).')
    group_exp.add_argument('--rep_num', type=int, default=5,
                           help='Number of experimental repetitions for statistical stability.')
    group_exp.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility.')

    # --- Model Hyperparameters ---
    group_model = parser.add_argument_group('Model Hyperparameters')
    group_model.add_argument('--num_epochs', type=int, default=200,
                             help='Maximum number of training epochs.')
    group_model.add_argument('--lr', type=float, default=1e-3,
                             help='Initial learning rate (typical range for MLP: 1e-3 to 1e-4).')
    group_model.add_argument('--weight_decay', type=float, default=5e-4,
                             help='L2 regularization coefficient (Weight Decay).')
    group_model.add_argument('--dropout_rate', type=float, default=0.2,
                             help='Dropout probability to prevent overfitting.')
    group_model.add_argument('--hid_dim', type=int, default=256,
                             help='Dimensionality of the common feature alignment space.')
    group_model.add_argument('--fusion_strategy', type=str, default='adaptive',
                             choices=['concat', 'sum', 'mean', 'adaptive'],
                             help='Strategy to fuse multi-view features.')

    # --- Data Splitting & Preprocessing ---
    group_data = parser.add_argument_group('Data Splitting')
    group_data.add_argument('--train_ratio', type=float, default=0.1,
                            help='Proportion of data used for training (0.0 to 1.0).')
    group_data.add_argument('--val_ratio', type=float, default=0.1,
                            help='Proportion of data used for validation.')
    group_data.add_argument('--test_ratio', type=float, default=0.8,
                            help='Proportion of data used for testing.')

    # --- Hardware & Environment ---
    group_hw = parser.add_argument_group('Hardware Settings')
    group_hw.add_argument('--device', type=str, default="0",
                          help='GPU device ID (e.g., "0", "1") or "cpu".')
    group_hw.add_argument('--res_path', type=str, default="./results/",
                          help='Directory path to save experimental logs and results.')
    group_hw.add_argument('--fastmode', action='store_true', default=False,
                          help='Skip frequent validation to accelerate training.')

    return parser.parse_args()