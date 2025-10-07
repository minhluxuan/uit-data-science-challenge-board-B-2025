# VIHALLU

This repository contains code for training and running inference on an ensemble of 35 Qwen3-Embedding-4B models for the VIHALLU task.

## Project Structure

```
.
├── checkpoints/             # Model checkpoints (35 models)
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 34/
├── data/                    # Dataset files
├── orig_results/            # Previous inference results
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 34/
├── results/                 # New inference results will be stored here
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 34/
├── download_checkpoints.py  # Script to download pre-trained checkpoints
├── train.py                 # Train all models
├── infer.py                 # Run inference for individual models
├── context_reduction.py     # Preprocess data by reducing irrelevant contexts
├── correct_bmd1905.py       # Preprocess data by correcting lexical errors in prompts
├── ensemble.py              # Final ensemble inference script
└── private-test-submit.csv  # Best results submitted on CodaBench
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can either train models from scratch or use pre-trained checkpoints.

### Option 1: Using previous generated test logits (Recommend)

1. Ensemble all previously generated test logits:
```bash
python ensemble.py --orig
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
  - Batch size: 16
  - Epochs: 2
  - Max sequence length: 512

## Inference

Each model produces logits that are saved as PyTorch tensors (`test-logits.pt`) and predictions in CSV format (`submit.csv`). The final ensemble combines these predictions to create the final submission.

## Directory Structure

- `checkpoints/{i}/best-checkpoint/`: Contains the trained model for ensemble member i
- `results/{i}/`: Contains predictions and logits for ensemble member i
- `data/`: Contains the preprocessed datasets
