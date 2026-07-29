<div align="center">

# MultiView Learning Benchmark

A lightweight benchmark for multi-view learning, providing datasets and baseline implementations based on MLP, GCN, and Transformer.


</div>

## Overview

This repository provides a lightweight benchmark for multi-view learning and semi-supervised classification. It is intended to help researchers and beginners reproduce basic experiments, understand common multi-view architectures, and develop new methods.

The repository currently includes:

- MLP-based multi-view baseline
- GCN-based multi-view baseline
- Transformer-based multi-view baseline
- Dataset download utilities
- Standardized preprocessing and experimental pipelines

## Repository Structure

```text
MultiView-Learning-Benchmark/
├── Baseline/
│   ├── Mv_MLP/
│   ├── Mv_GCN/
│   ├── Mv_Transformer/
│   └── requirements.txt
├── Dataset/
│   └── download_multi-view_data.py
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/LosparkSayoji/MultiView-Learning-Benchmark.git
cd MultiView-Learning-Benchmark
```

Install the required packages:

```bash
pip install -r Baseline/requirements.txt
pip install gdown
```

## Dataset

Download the multi-view datasets by running:

```bash
python Dataset/download_multi-view_data.py --output ./dataset
```

The datasets will be downloaded from Google Drive and stored in the specified output directory.

## Running the Baselines

### MLP

```bash
cd Baseline/Mv_MLP
python main.py
```

### GCN

```bash
cd Baseline/Mv_GCN
python main.py
```

### Transformer

```bash
cd Baseline/Mv_Transformer
python main.py
```

Before running an experiment, check the corresponding `args.py` file and configure the dataset path and experimental parameters as needed.

## Requirements

The main dependencies include:

- Python 3
- PyTorch ≥ 2.1
- NumPy
- SciPy
- scikit-learn
- tqdm
- gdown

## License

This project is released under the [MIT License](LICENSE).
