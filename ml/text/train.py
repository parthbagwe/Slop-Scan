# ml/text/train.py
#
# PURPOSE: Load the HuggingFace dataset, fine-tune a RoBERTa model,
# evaluate it, and save it to disk so the detector can load it.
#
# WHY FINE-TUNE INSTEAD OF USING THE BASE MODEL?
# "roberta-base" is a general-purpose language model.
# It doesn't know what "AI-generated text" means.
# Fine-tuning = show it thousands of examples labeled
# human=0 or AI=1, and it learns the distinction.
#
# DATASET: mehddii/ai-text-detector-v2
# Each row has: text (string), label (0=human, 1=AI)
#
# THIS IS YOUR BASIC MLOps:
# 1. Load data
# 2. Train model
# 3. Evaluate (accuracy, F1)
# 4. Save model + metrics to disk
# 5. Detector loads saved model → no training at inference time
#
# Run this ONCE from your terminal:
#   cd ai-slop-detector
#   python ml/text/train.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"           # base model from HuggingFace
SAVE_DIR     = Path("ml/saved_model")   # where to write the trained model
METRICS_FILE = SAVE_DIR / "metrics.json"
MAX_LENGTH   = 512                      # RoBERTa token limit
BATCH_SIZE   = 16                       # reduce to 8 if you get OOM errors
EPOCHS       = 3
LEARNING_RATE = 2e-5
# ────────────────────────────────────────────────────────────────


def load_and_prepare_dataset():
    """
    Load mehddii/ai-text-detector-v2 from HuggingFace Hub.

    WHAT THE DATASET LOOKS LIKE:
    Each row:  { "text": "...", "label": 0 or 1 }
    label 0 = human-written
    label 1 = AI-generated

    We split into train (80%), validation (10%), test (10%).
    WHY THREE SPLITS?
    - train: model learns from this
    - validation: we monitor this during training to detect overfitting
    - test: final honest evaluation (never seen during training)
    """
    logger.info("Loading dataset from HuggingFace Hub...")
    ds = load_dataset("mehddii/ai-text-detector-v2")
    logger.info(f"Dataset loaded: {ds}")

    # If the dataset only has a 'train' split, create val+test ourselves
    if "validation" not in ds and "test" not in ds:
        logger.info("Splitting train into train/val/test (80/10/10)...")
        train_val_test = ds["train"].train_test_split(test_size=0.2, seed=42)
        val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)
        ds = {
            "train":      train_val_test["train"],
            "validation": val_test["train"],
            "test":       val_test["test"],
        }
    elif "validation" not in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds = {"train": split["train"], "validation": split["test"]}

    # Log class distribution so we can spot imbalance
    # If 90% of training data is AI, model will just predict AI for everything
    labels = ds["train"]["label"] if hasattr(ds["train"], "__getitem__") else []
    if labels:
        human_count = sum(1 for l in labels if l == 0)
        ai_count    = sum(1 for l in labels if l == 1)
        logger.info(f"Train labels: human={human_count}, AI={ai_count}")

    return ds


def tokenize_dataset(ds, tokenizer):
    """
    Convert raw text into token IDs that RoBERTa understands.

    WHAT TOKENIZATION DOES:
    "Hello world" → [0, 31414, 232, 2]
    (special start token, word pieces, end token)

    TRUNCATION: texts longer than 512 tokens get cut.
    PADDING: shorter texts get padded to the same length in a batch.
    This is needed because GPU operations require fixed-size tensors.
    """
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    logger.info("Tokenizing dataset...")
    tokenized = {}
    for split_name, split_data in ds.items():
        tokenized[split_name] = split_data.map(
            tokenize,
            batched=True,        # Process in batches (faster)
            batch_size=64,
            remove_columns=["text"],  # We no longer need raw text
        )
        tokenized[split_name].set_format("torch")  # Return PyTorch tensors

    return tokenized


def compute_metrics(eval_pred):
    """
    Called by the Trainer after each validation epoch.

    WHY F1 SCORE, NOT JUST ACCURACY?
    Accuracy is misleading on imbalanced datasets.
    If 90% of data is AI, predicting "AI" always gives 90% accuracy
    but the model learned nothing useful.

    F1 = harmonic mean of precision and recall.
    Precision = of all things we called AI, how many actually were?
    Recall    = of all actual AI texts, how many did we catch?
    F1 balances both.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy":  accuracy_score(labels, predictions),
        "f1":        f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall":    recall_score(labels, predictions, average="weighted"),
    }


def train():
    """Main training function."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ─────────────────────────────────────────
    ds = load_and_prepare_dataset()

    # ── 2. Load tokenizer and model ─────────────────────────────
    logger.info(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # num_labels=2 → binary classification (human vs AI)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "human", 1: "AI"},
        label2id={"human": 0, "AI": 1},
    )

    # ── 3. Tokenize ──────────────────────────────────────────────
    tokenized = tokenize_dataset(ds, tokenizer)

    # ── 4. Define training config ────────────────────────────────
    run_name = f"slop-detector-{datetime.now().strftime('%Y%m%d-%H%M')}"

    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR / "checkpoints"),
        run_name=run_name,

        # Training schedule
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,           # Warm up LR for first 10% of steps
        weight_decay=0.01,          # L2 regularization (prevents overfitting)

        # Batch sizes
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Eval can use more memory

        # Evaluation
        eval_strategy="epoch",      # Evaluate after every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_dir=str(SAVE_DIR / "logs"),
        logging_steps=50,

        # Use GPU if available
        fp16=torch.cuda.is_available(),  # Mixed precision (faster on GPU)
        
        report_to="none",           # Disable wandb/mlflow for basic setup
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        compute_metrics=compute_metrics,
        callbacks=[
            # Stop early if validation F1 doesn't improve for 2 epochs
            EarlyStoppingCallback(early_stopping_patience=2)
        ],
    )

    # ── 5. Train ─────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── 6. Final evaluation on test set ─────────────────────────
    if "test" in tokenized:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized["test"])
        logger.info(f"Test results: {test_results}")

        # Detailed per-class report
        predictions = trainer.predict(tokenized["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        report = classification_report(
            predictions.label_ids, preds,
            target_names=["human", "AI"]
        )
        logger.info(f"\n{report}")

        # Save metrics to disk for reference
        metrics = {
            "model": MODEL_NAME,
            "dataset": "mehddii/ai-text-detector-v2",
            "trained_at": datetime.now().isoformat(),
            "test_accuracy": test_results.get("eval_accuracy"),
            "test_f1":       test_results.get("eval_f1"),
            "test_precision": test_results.get("eval_precision"),
            "test_recall":   test_results.get("eval_recall"),
            "epochs": EPOCHS,
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {METRICS_FILE}")

    # ── 7. Save the final model ──────────────────────────────────
    logger.info(f"Saving model to {SAVE_DIR}...")
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))

    logger.info("Training complete!")
    logger.info(f"Model saved to: {SAVE_DIR.absolute()}")


if __name__ == "__main__":
    train()