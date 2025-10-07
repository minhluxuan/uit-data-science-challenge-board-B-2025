# VIHALLU

This repository contains code for training and running inference on an ensemble of 35 Qwen3-Embedding-4B models for the VIHALLU task.

## Project Structure

```
.
├── checkpoints/          # Model checkpoints (35 models)
├── data/                 # Dataset files
├── old_results/             # Previous inference results and submissions
├── new_results/             # The new inference results will be store here
├── download_checkpoints.py  # Script to download pre-trained checkpoints
├── train.py             # Training script for all models
├── infer.py             # Inference script for individual models
├── preprocess.py        # Data preprocessing script
├── ensemble.py       # Final ensemble inference script
└── previous_submit.py   # Best results submitted on CodaBench
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can either train models from scratch or use pre-trained checkpoints.

### Option 1: Using previous test logits (Recommend)

1. Ensemble all previously generated test logits:
```bash
python ensemble.py
```
This will ensemble the predictions from all models and create the final `submit.csv`.

### Option 2: Using Pre-trained Checkpoints
Each model requires approximately 2 minutes for inference on an A100 40GB GPU. Therefore, 35 models take a total of 35 × 2 = 70 minutes.

1. Download the pre-trained checkpoints:
```bash
python download_checkpoints.py
```
This will download all model checkpoints to the `checkpoints/` directory.

2. Run inference for all models:
```bash
python infer.py
```
This will generate predictions and logits for all 35 models in the `results/` directory.

3. Generate final submission:
```bash
python ensemble.py
```
This will ensemble the predictions from all models and create the final `submit.csv`.

## Model Architecture

- Base Model: Qwen3-Embedding-4B
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Number of Ensemble Members: 35
- Training Parameters:
  - LoRA rank (r): 32
  - LoRA alpha: 64
  - Learning rate: 1e-4
  - Batch size: 8
  - Epochs: 2
  - Max sequence length: 512

## Inference

Each model produces logits that are saved as PyTorch tensors (`test-logits.pt`) and predictions in CSV format (`submit.csv`). The final ensemble combines these predictions to create the final submission.

## Directory Structure

- `checkpoints/{i}/best-checkpoint/`: Contains the trained model for ensemble member i
- `results/{i}/`: Contains predictions and logits for ensemble member i
- `data/`: Contains the preprocessed datasets
