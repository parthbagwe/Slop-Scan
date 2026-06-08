# ml/text/retrain.py
#
# PURPOSE: Take new human-labeled examples and fine-tune
# the existing saved model further.
#
# WHY NOT RETRAIN FROM SCRATCH?
# Retraining from scratch takes hours.
# Fine-tuning the existing model on new examples takes minutes.
# This is called "incremental learning" or "continual learning".
#
# We run just 1-2 epochs on the new data so the model
# adapts without "forgetting" what it already learned.
# (Catastrophic forgetting is a real problem in ML!)

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR   = Path("ml/saved_model")
BACKUP_DIR  = Path("ml/saved_model_backup")


def retrain_with_feedback(texts: list, labels: list) -> bool:
    """
    Fine-tune the existing model on new feedback data.
    
    Args:
        texts:  list of text strings
        labels: list of 0 (human) or 1 (AI)
    
    Returns:
        True if retraining succeeded, False otherwise
    """
    if not MODEL_DIR.exists():
        logger.error("No saved model found. Run train.py first.")
        return False

    if len(texts) < 10:
        logger.warning("Too few samples to retrain meaningfully")
        return False

    try:
        import torch
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            AdamW,
            get_linear_schedule_with_warmup,
        )
        from sklearn.metrics import f1_score

        # ── 1. Backup current model ──────────────────────────────
        # Always backup before overwriting — safety net
        import shutil
        if MODEL_DIR.exists():
            if BACKUP_DIR.exists():
                shutil.rmtree(BACKUP_DIR)
            shutil.copytree(MODEL_DIR, BACKUP_DIR)
            logger.info("Backed up current model")

        # ── 2. Load existing model ───────────────────────────────
        logger.info("Loading existing model for fine-tuning...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

        # ── 3. Create dataset ────────────────────────────────────
        class FeedbackDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt",
                )
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids":      self.encodings["input_ids"][idx],
                    "attention_mask": self.encodings["attention_mask"][idx],
                    "labels":         self.labels[idx],
                }

        dataset    = FeedbackDataset(texts, labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # ── 4. Fine-tune ─────────────────────────────────────────
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        # Lower learning rate than initial training (2e-5)
        # We don't want to overwrite what the model already knows

        # Only 2 epochs on feedback data
        # More epochs = overfitting on small dataset
        EPOCHS = 2

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * EPOCHS,
        )

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch   = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch,
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}")

        # ── 5. Quick evaluation ──────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        f1 = f1_score(all_labels, all_preds, average="weighted")
        logger.info(f"Post-retrain F1 on feedback data: {f1:.3f}")

        # ── 6. Save updated model ────────────────────────────────
        model.save_pretrained(str(MODEL_DIR))
        tokenizer.save_pretrained(str(MODEL_DIR))
        logger.info("Updated model saved successfully")

        # ── 7. Reload detector's cached model ───────────────────
        # Force detector.py to reload the model on next request
        import ml.text.detector as det
        det._model        = None
        det._tokenizer    = None
        det._model_loaded = False
        logger.info("Detector cache cleared — will reload on next request")

        return True

    except Exception as e:
        logger.error(f"Retraining failed: {e}")

        # Restore backup if retraining failed
        if BACKUP_DIR.exists():
            import shutil
            shutil.rmtree(MODEL_DIR)
            shutil.copytree(BACKUP_DIR, MODEL_DIR)
            logger.info("Restored backup model after failed retrain")

        return False