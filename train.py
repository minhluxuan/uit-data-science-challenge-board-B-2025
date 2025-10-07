import os
import sys
import time
import argparse

import torch
import random
import evaluate
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import set_seed
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

acc_metric = evaluate.load("accuracy")
prec_metric = evaluate.load("precision")
rec_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    global acc_metric, prec_metric, rec_metric, f1_metric
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    prec = prec_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    rec = rec_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


model_id = "Qwen/Qwen3-Embedding-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, dtype="bfloat16")
base_model.config.pad_token_id = tokenizer.pad_token_id


def set_global_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator()
    gen.manual_seed(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)

    set_seed(seed, deterministic=deterministic)

    return gen


def train_model(model_id):
    global tokenizer, base_model
    
    ensemble_idx = model_id
    seed_map = [i for i in range(20)] + [i for i in range(10)] + [i for i in range(5)]
    SEED = 42 * (seed_map[ensemble_idx] + 1)
    
    path_to_checkpoints = f"./checkpoints/{ensemble_idx}"
    SPLIT = {
        "train": f"./data/vihallu-train-clean-reduced.csv",
        "val": f"./data/vihallu-val-clean-reduced.csv",
        "test": "./data/vihallu-pvtest-clean-reduced.csv",
    }
    
    DET_ALGO = False
    set_global_seed(SEED, DET_ALGO)
    
    label2id = {"no": 0, "intrinsic": 1, "extrinsic": 2}
    id2label = {v: k for k, v in label2id.items()}

    train_df = pd.read_csv(SPLIT["train"])
    val_df = pd.read_csv(SPLIT["val"])
    test_df = pd.read_csv(SPLIT["test"])

    for df in [train_df, val_df]:
        df["label"] = df["label"].map(label2id)
        df["label"] = df["label"].astype(int)

    print(f"""Dataset sizes:
        train={len(train_df)},
        val={len(val_df)},
        test={len(test_df)},
    """
    )

    train_dataset = Dataset.from_pandas(train_df).with_format("torch")
    val_dataset = Dataset.from_pandas(val_df).with_format("torch")
    test_dataset = Dataset.from_pandas(test_df).with_format("torch")

    def q_tokenize_fn(examples):
        contexts, prompts, responses = examples["context"], examples["prompt"], examples["response"]
        texts = [f"Context: {c}\n Question: {q}\nResponse: {r}" for c, q, r in zip(contexts, prompts, responses)]
        return tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

    def r_tokenize_fn(examples):
        contexts, prompts, responses = examples["context"], examples["prompt"], examples["response"]
        texts = [f"Context: {c}\nQuestion: {q}\n Response: {r}" for c, q, r in zip(contexts, prompts, responses)]
        return tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

    rm_cols = ["id", "context", "prompt", "response"]
    train_dataset = train_dataset.map(q_tokenize_fn, batched=True, remove_columns=rm_cols).rename_column("label", "labels")
    val_dataset = val_dataset.map(r_tokenize_fn, batched=True, remove_columns=rm_cols).rename_column("label", "labels")
    test_dataset = test_dataset.map(q_tokenize_fn, batched=True, remove_columns=rm_cols + ["predict_label"])

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=path_to_checkpoints,
        eval_strategy="epoch",
        
        save_strategy="epoch",
        save_total_limit=None,
        
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        

        report_to="none",
        logging_dir="./logs",
        logging_steps=50,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        fp16=False,
        bf16=True,
        seed=SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer_outputs = trainer.train()
    print(trainer_outputs)
    trainer.save_model(f"{path_to_checkpoints}/best-checkpoint/")
    

def main():
    NUM_MODELS = 35

    parser = argparse.ArgumentParser(
        description="Train machine learning models."
    )
    parser.add_argument(
        "model_ids",
        nargs="*",
        type=int,
        help=f"IDs of models to train (0–{NUM_MODELS - 1}). If omitted, all models will be trained."
    )

    args = parser.parse_args()

    if not args.model_ids:
        print(f"Training ALL {NUM_MODELS} models...\n")
        for model_id in range(NUM_MODELS):
            train_model(model_id)
    else:
        for model_id in args.model_ids:
            if 0 <= model_id < NUM_MODELS:
                train_model(model_id)
            else:
                print(f"Invalid model ID: {model_id}. Please choose between 0 and {NUM_MODELS - 1}.")
                sys.exit(1)

if __name__ == "__main__":
    main()
