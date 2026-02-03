import os
import datetime
import random
import numpy as np
import torch
from warnings import simplefilter

# Internal module imports
from utils import prepare_graph_data
from train import train, test
from DataLoader.Mv_DataLoader import mv_load_data, mv_generate_permutation
from args import parameter_parser

# Environment and Warning Configurations
os.environ["OMP_NUM_THREADS"] = "1"
simplefilter(action='ignore', category=FutureWarning)


def run_classification(features, num_class, num_views, feature_dims, labels,
                       train_idx, val_idx, test_idx, args, device):
    """
    Executes repeated experimental trials for Multi-view GCN classification.
    Handles graph construction, model training, and performance logging.
    """

    # 1. Graph Construction (Feature Scaling + Adjacency Matrix Generation)
    # The 'k' parameter typically refers to the k-nearest neighbors used for graph construction
    processed_features, adjs = prepare_graph_data(features, train_idx, k=args.k, device=device)

    acc_list, f1_list, best_epoch_list = [], [], []

    print(f"\n>>> Launching Experiment | Dataset: {args.dataset} | Fusion: {args.fusion_strategy} <<<")

    for rep in range(args.rep_num):
        print(f'--- Trial {rep + 1}/{args.rep_num} ---')

        # 2. Training Phase (Returns optimized model and best-performing epoch)
        model, labels, best_epoch = train(
            processed_features, adjs, num_class, feature_dims,
            labels, train_idx, val_idx, args, device
        )

        # 3. Evaluation Phase
        acc, f1 = test(model, processed_features, adjs, labels, test_idx, args, device)

        acc_list.append(acc)
        f1_list.append(f1)
        best_epoch_list.append(best_epoch)

    # --- Statistical Aggregation ---
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)

    print("=" * 60)
    print(f"Aggregated Statistics for {args.dataset}:")
    print(f"Average ACC: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%")
    print(f"Average F1 : {f1_mean * 100:.2f}% ± {f1_std * 100:.2f}%")
    print("=" * 60)

    # --- Results Persistence ---
    os.makedirs(args.res_path, exist_ok=True)
    result_file = os.path.join(args.res_path, f"{args.dataset}.txt")

    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(f"{'=' * 100}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Trials: {args.rep_num}\n")
        f.write(f"Parameters: Strategy={args.fusion_strategy} | k_neighbors={args.k} | Epochs={args.num_epochs} | "
                f"LR={args.lr} | WD={args.weight_decay} | Hidden={args.hid_dim} | Dropout={args.dropout_rate}\n")
        f.write(f"Performance: ACC={acc_mean:.4f} ± {acc_std:.4f} | F1={f1_mean:.4f} ± {f1_std:.4f} | "
                f"Best_Epochs={best_epoch_list}\n")


# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialize CLI Argument Parser
    args = parameter_parser()

    # Hardware Configuration (Default to CPU if CUDA is unavailable)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{args.device}')

    # Experiment Pipeline for Target Datasets
    DATASETS_LIST = ['100leaves']

    # Path Configuration
    # TODO: Modify DATA_ROOT to your local path or use relative paths for portability
    DATA_ROOT = 'D:/Lab/dataset/multiview datasets/'

    for dataset in DATASETS_LIST:
        args.dataset = dataset

        # Multi-view Data Loading
        features, labels, feature_dims, num_views, num_class = mv_load_data(args.dataset, DATA_ROOT)

        # Dataset Splitting (Train/Val/Test)
        train_idx, val_idx, test_idx = mv_generate_permutation(labels, args)

        # Execute Classification Task
        run_classification(
            features, num_class, num_views, feature_dims, labels,
            train_idx, val_idx, test_idx, args, device
        )
