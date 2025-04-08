# AG News Classification with LoRA-Modified RoBERTa

This project implements a low-rank adaptation (LoRA) of the RoBERTa model for the AG News classification task, with a focus on parameter efficiency and training optimization.

## Project Structure

```
dp_sp25_proj2/
├── data/                   # AG News dataset
├── models/                 # Model checkpoints
├── outputs/               # Training outputs and logs
├── scripts/               # Training and evaluation scripts
│   ├── train.py          # Main training script
│   ├── predict.py        # Prediction script
│   └── run_experiments.sh # Experiment runner
└── src/                   # Source code
    ├── data/             # Data loading and preprocessing
    ├── models/           # Model architecture
    └── training/         # Training utilities
```

## Setup

1. Create and activate conda environment:
```bash
conda create -n dp_sp25_proj2 python=3.10
conda activate dp_sp25_proj2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Run a single experiment:
```bash
python scripts/train.py --exp_name "experiment_name" --batch_size 32 --learning_rate 2e-4
```

2. Run multiple experiments:
```bash
./scripts/run_experiments.sh
```

### Prediction

Generate predictions on test set:
```bash
python scripts/predict.py --model_path "path/to/best/model.pth"
```

## Experiment Configuration

The following parameters can be configured in `src/training/config.py`:

- Model parameters (LoRA rank, attention heads, etc.)
- Training hyperparameters (batch size, learning rate, etc.)
- Output and logging settings
- Data settings

## Outputs

- Model checkpoints are saved in `outputs/best_models/`
- Training logs and metrics are saved in `outputs/training_runs/`
- Best performing models are tracked in Git

## Monitoring

- Training progress is displayed in real-time
- GPU memory usage is monitored
- Tensorboard logs are available for visualization
- Training metrics are saved in CSV format