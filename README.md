# VIHALLU

This repository contains code for training and running inference on an ensemble of 35 Qwen3-Embedding-4B models for the VIHALLU task.

## Project Structure

```
.
├── checkpoints/          # Model checkpoints (35 models)
├── data/                 # Dataset files
├── results/             # Inference results and submissions
├── download_checkpoints.py   # Script to download pre-trained checkpoints
├── train.py             # Training script for all models
├── infer.py             # Inference script for individual models
├── preprocess.py        # Data preprocessing script
└── final_infer.py       # Final ensemble inference script
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can either train models from scratch or use pre-trained checkpoints.

### Option 1: Using Pre-trained Checkpoints

1. Download the pre-trained checkpoints:
```bash
python download_checkpoints.py
```
This will download all model checkpoints to the `checkpoints/` directory.

2. Preprocess the data:
```bash
python preprocess.py
```
This will create clean-reduced versions of the dataset in the `data/` directory:
- `vihallu-train-clean-reduced.csv`
- `vihallu-val-clean-reduced.csv`
- `vihallu-pvtest-clean-reduced.csv`

3. Run inference for all models:
```bash
python infer.py
```
This will generate predictions and logits for all 35 models in the `results/` directory.

4. Generate final submission:
```bash
python final_infer.py
```
This will ensemble the predictions from all models and create the final `submit.csv`.

### Option 2: Training from Scratch

1. Preprocess the data (same as above):
```bash
python preprocess.py
```

2. Train all models:
```bash
python train.py
```
This will train 35 different models with different random seeds. Each model will be saved in the `checkpoints/{i}/best-checkpoint/` directory.

- To train specific models, you can specify model IDs:
```bash
python train.py 0 1 2  # Train only models 0, 1, and 2
```

3. Run inference and generate final submission (same as steps 3-4 above).

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
