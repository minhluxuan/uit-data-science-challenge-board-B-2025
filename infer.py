import os
import gc
import sys
import time
import torch
import argparse
import numpy as np

from peft import PeftModel
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# DATA

print("Loading and tokenizing dataset...")
SPLIT = {
    "test": "./data/vihallu-pvtest-clean-reduced.csv",
}

testset = load_dataset("csv", data_files=SPLIT["test"], split="train").with_format("torch")

model_name = "Qwen/Qwen3-Embedding-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

def q_tokenize_fn(batch):
    global tokenizer
    contexts, questions, responses = batch["context"], batch["prompt"], batch["response"]
    texts = [f"Context: {c}\n Question: {q}\nResponse: {r}"for c, q, r in zip(contexts, questions, responses)]
    return tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

rmcols = ["id", "context", "prompt", "response"]
tk_testset = testset.map(q_tokenize_fn, batched=True, remove_columns=rmcols + ["predict_label"])


# MODEL

print("Loading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, dtype="bfloat16")
base_model.config.pad_token_id = tokenizer.pad_token_id


# INFER

def get_logits(idx):
    global base_model
    
    ckpt_path = f"./checkpoints/{idx}/best-checkpoint"
    model = PeftModel.from_pretrained(base_model, ckpt_path, local_files_only=True)

    training_args = TrainingArguments(
        output_dir=f"/{idx}",
        per_device_eval_batch_size=64,
        logging_strategy="no",
        save_strategy="no",
        do_train=False,
        do_eval=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    test_preds = trainer.predict(tk_testset).predictions

    del model
    gc.collect()

    return test_preds


label2id = {"no": 0, "intrinsic": 1, "extrinsic": 2}
id2label = {k: v for v, k in label2id.items()}

def get_labels(logits):
    global id2label
    ids = np.argmax(logits, axis=-1).tolist()
    return [id2label[id] for id in ids]


def get_finals(idx, dataset, labels, output_name):
    dataset = dataset.to_pandas()
    dataset["predict_label"] = labels

    dataset.to_csv(f"./results/{idx}/{output_name}.csv", index=False)
    if output_name == "test_predictions":
        reduced = dataset[["id", "predict_label"]]
        reduced.to_csv(f"./results/{idx}/submit.csv", index=False)
    return dataset


def infer_model(idx):
    global testset
    os.makedirs(f"./results/{idx}", exist_ok=True)

    test_logits = get_logits(idx)
    torch.save(test_logits, f"./results/{idx}/test-logits.pt")
    test_labels = get_labels(test_logits)

    get_finals(idx, testset, test_labels, "test_predictions")


def main():
    NUM_MODELS = 35

    parser = argparse.ArgumentParser(
        description="Run inference for trained models."
    )
    parser.add_argument(
        "model_ids",
        nargs="*",
        type=int,
        help=f"IDs of models to run inference on (0–{NUM_MODELS - 1}). If omitted, all models will be inferred."
    )

    args = parser.parse_args()

    if not args.model_ids:
        print(f"Running inference for all {NUM_MODELS} models...\n")
        for model_id in range(NUM_MODELS):
            infer_model(model_id)
    else:
        for model_id in args.model_ids:
            if 0 <= model_id <= NUM_MODELS - 1:
                infer_model(model_id)
            else:
                print(f"Invalid model ID: {model_id}. Please choose between 0 and {NUM_MODELS - 1}.")
                sys.exit(1)


if __name__ == "__main__":
    main()
