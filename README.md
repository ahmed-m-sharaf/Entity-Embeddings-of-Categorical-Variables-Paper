# Entity Embeddings of Categorical Variables

This repository implements the [Entity Embeddings of Categorical Variables paper](https://arxiv.org/abs/1604.06737) using the standard deep learning course project template structure. The task involves predicting Rossmann store sales.

## Structure 

The code strictly follows the `dl_course_project_template`:

- `config.py`: Defines standard `BaselineConfig` and ablation experiments via `@dataclass`.
- `data.py`: Handles data loading, preprocessing, feature engineering, and PyTorch dataloaders mapping via `RossmannDataset`.
- `model.py`: Contains the deep neural network architecture `RossmannEmbeddingModel` capturing entity embeddings.
- `loss.py`: Simple MSE or L1 loss retrieval.
- `metrics.py`: Calculation of MAPE.
- `train.py`: Contains wandb instrumentation, handles training over the ensemble, saves checkpoints and runs evaluations.
- `evaluate.py`: Standalone evaluation script loading from a given checkpoint.
- `utils.py`: Auxiliary tools such as device detection and random seeding.

## Setup

First, make sure all Python requirements are satisfied:

```bash
pip install -r requirements.txt
```

Ensure the datasets are placed inside the `data/` directory:
- `data/train.csv`
- `data/store_states.csv`

## Execution

To train the original baseline model (which incorporates random splitting):

```bash
python train.py --config baseline
```

To run ablation experiments, simply specify the config named defined in `config.py` (e.g., `ablation_time_based`, `ablation_embedding`, `ablation_bn`, `ablation_dec_lr`, `ablation_inc_lr`):

```bash
python train.py --config ablation_bn
```

To evaluate a previously saved model:

```bash
python evaluate.py --config baseline --checkpoint models/baseline/baseline_1.pth
```
