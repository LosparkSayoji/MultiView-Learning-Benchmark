import os
import datetime
import random
import numpy as np
import torch
from warnings import simplefilter

# Environment and Warning Configurations
os.environ["OMP_NUM_THREADS"] = "1"
simplefilter(action='ignore', category=FutureWarning)

# Internal module imports
from train import train, test
from DataLoader.Mv_DataLoader import mv_load_data, mv_generate_permutation
from args import parameter_parser


def run_classification(features, num_class, num_views, feature_dims, labels,
                       train_idx, val_idx, test_idx, args, device):
    """
    Executes multiple experimental trials and logs the statistical results.
    Specifically tailored for the Multi-view Transformer pipeline.
    """
    acc_list, f1_list, best_epoch_list = [], [], []

    print(f"\n>>> Starting Experiment | Dataset: {args.dataset} | Strategy: {args.fusion_strategy} <<<")

    for rep in range(args.rep_num):
        print(f'--- Trial {rep + 1}/{args.rep_num} ---')

        # 1. Training Phase
        # Returns the optimized model, preprocessed features, and the best-performing epoch
        model, processed_features, labels, best_epoch = train(
            features, num_class, feature_dims, labels,
            train_idx, val_idx, args, device
        )

        # 2. Evaluation Phase
        acc, f1 = test(model, processed_features, labels, test_idx)

        acc_list.append(acc)
        f1_list.append(f1)
        best_epoch_list.append(best_epoch)

    # --- Statistical Aggregation ---
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)

    print("=" * 60)
    print(f"Final Statistics for {args.dataset}:")
    print(f"Average ACC: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%")
    print(f"Average F1 : {f1_mean * 100:.2f}% ± {f1_std * 100:.2f}%")
    print("=" * 60)

    # --- Results Persistence (I/O) ---
    os.makedirs(args.res_path, exist_ok=True)
    result_file = os.path.join(args.res_path, f"{args.dataset}.txt")

    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(f"{'=' * 100}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Repetitions: {args.rep_num}\n")
        f.write(f"Architecture: d_model={args.hid_dim} | nhead={args.nhead} | layers={args.num_layers} | d_ff={args.dim_feedforward}\n")
        f.write(f"Hyperparams: Strategy={args.fusion_strategy} | Epochs={args.num_epochs} | LR={args.lr} | WD={args.weight_decay} | Drop={args.dropout_rate}\n")
        f.write(f"Performance:  ACC={acc_mean:.4f} ± {acc_std:.4f} | F1={f1_mean:.4f} ± {f1_std:.4f} | Best_Epochs={best_epoch_list}\n")


# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Parse CLI Arguments
    args = parameter_parser()

    # 2. Hardware Configuration
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{args.device}')

    # 3. Experiment Pipeline
    # List of datasets to iterate over
    datasets_to_run = ['HW']

    # Path configuration for multi-view datasets
    # Recommendation: Move DATA_ROOT to args.py for better portability
    DATA_ROOT = 'D:/Lab/dataset/multiview datasets/'

    for dataset in datasets_to_run:
        args.dataset = dataset

        # Load raw data from .mat files
        features, labels, feature_dims, num_views, num_class = mv_load_data(
            args.dataset,
            DATA_ROOT
        )

        # Generate stratified index splits (Train/Val/Test)
        train_idx, val_idx, test_idx = mv_generate_permutation(labels, args)

        # Launch the repetitive experimental process
        run_classification(
            features, num_class, num_views, feature_dims, labels,
            train_idx, val_idx, test_idx, args, device
        )
