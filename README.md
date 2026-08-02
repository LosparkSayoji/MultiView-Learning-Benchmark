<div align="center">

# MultiView Learning Benchmark

A lightweight PyTorch benchmark for multi-view learning and semi-supervised classification, with MLP-, GCN-, and Transformer-based baselines.

</div>

## Overview

This repository provides a compact and accessible benchmark for multi-view learning. It is designed to help researchers and beginners reproduce baseline experiments, understand common multi-view architectures, and build new methods on top of a consistent experimental pipeline.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/LosparkSayoji/MultiView-Learning-Benchmark.git
cd MultiView-Learning-Benchmark
```

Install the dependencies:

```bash
pip install -r Baseline/requirements.txt
pip install gdown
```

The main requirements are Python 3, PyTorch 2.1 or later, NumPy, SciPy, scikit-learn, tqdm, and gdown.

## Dataset

The multi-view datasets used by this benchmark are available in the following shared Google Drive folder:

> **[Download the multi-view datasets from Google Drive](https://drive.google.com/drive/folders/1TyiQNOuCH7zn0R55EfxM4mUwB05VsoMf?usp=sharing)**

You can also download the datasets automatically from the repository root:

```bash
python Dataset/download_multi-view_data.py --output ./dataset
```

## Configuration

Before running a baseline, open its `main.py` file and update the dataset list and local data path:

```python
DATASETS_LIST = ["100leaves"]
DATA_ROOT = "/path/to/your/multiview-datasets/"
```

Other experimental settings can be configured through the corresponding `args.py` file or command-line arguments.

For example:

```bash
python main.py --device cpu --rep_num 5 --num_epochs 200
```
