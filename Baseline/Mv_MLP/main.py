import os
import datetime
import random
import numpy as np
import torch
from warnings import simplefilter

# Environment and Warning Configurations
os.environ["OMP_NUM_THREADS"] = "1"
simplefilter(action='ignore', category=FutureWarning)

# Custom Module Imports
from train import train, test
from DataLoader.Mv_DataLoader import mv_load_data, mv_generate_permutation
from args import parameter_parser


def run_classification(features, num_class, num_views, feature_dims, labels,
                       train_idx, val_idx, test_idx, args, device):
    """
    Executes multiple experimental repetitions and logs statistical results.

    Args:
        features (list/tensor): Input multi-view features.
        num_class (int): Number of target classes.
        num_views (int): Total number of views.
        feature_dims (list): Dimensions of features for each view.
        labels (tensor): Ground truth labels.
        train_idx, val_idx, test_idx (array): Indices for data splitting.
        args: Parsed command-line arguments.
        device: Torch device (CPU or CUDA).
    """
    acc_list, f1_list, best_epoch_list = [], [], []

    print(f"\n>>> Starting Experiment | Dataset: {args.dataset} | Method: {args.fusion_strategy} <<<")

    for rep in range(args.rep_num):
        print(f'--- Repetition {rep + 1}/{args.rep_num} ---')

        # 1. Training Phase (Returns the optimized model and best epoch)
        model, processed_features, labels, best_epoch = train(
            features, num_class, feature_dims, labels,
            train_idx, val_idx, args, device
        )

        # 2. Testing Phase
        acc, f1 = test(model, processed_features, labels, test_idx)

        acc_list.append(acc)
        f1_list.append(f1)
        best_epoch_list.append(best_epoch)

    # --- Statistical Analysis ---
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)

    print("=" * 60)
    print(f"Final Statistics for {args.dataset}:")
    print(f"ACC: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%")
    print(f"F1 : {f1_mean * 100:.2f}% ± {f1_std * 100:.2f}%")
    print("=" * 60)

    # --- Results Persistence ---
    os.makedirs(args.res_path, exist_ok=True)
    result_file = os.path.join(args.res_path, f"{args.dataset}.txt")

    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(f"{'-' * 100}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Reps: {args.rep_num}\n")
        f.write(f"Hyperparams: Strategy={args.fusion_strategy} | LR={args.lr} | WD={args.weight_decay} | "
                f"Hid={args.hid_dim} | Dropout={args.dropout_rate}\n")
        f.write(f"Metrics: ACC={acc_mean:.4f} ± {acc_std:.4f} | F1={f1_mean:.4f} ± {f1_std:.4f} | "
                f"Best_Epochs={best_epoch_list}\n")


# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Parse CLI Arguments
    args = parameter_parser()

    # 2. Device Setup
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{args.device}')

    # 3. Experiment Pipeline
    DATASETS_LIST = ['100leaves']
    # TODO: Modify DATA_ROOT to your local path or use relative paths for portability
    DATA_ROOT = 'D:/Lab/dataset/multiview datasets/'

    for dataset in DATASETS_LIST:
        args.dataset = dataset

        # Load Multi-view Data
        features, labels, feature_dims, num_views, num_class = mv_load_data(args.dataset, DATA_ROOT)

        # Generate Train/Val/Test Splits
        train_idx, val_idx, test_idx = mv_generate_permutation(labels, args)

        # Launch Classification Task
        run_classification(
            features, num_class, num_views, feature_dims, labels,
            train_idx, val_idx, test_idx, args, device
        )
