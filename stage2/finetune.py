"""
Stage 2 - Fine-tuning facebook/bart-base for Fact vs Opinion classification
==============================================================================
This is the "Fine-Tuning" route from the assignment's list of possible Stage 2
approaches, combined with a knowledge-base context injection: every input gets
a short tag prepended that summarises which linguistic markers (hedges,
evaluative words, attribution phrases, etc.) the knowledge base detected in it.
The classifier then learns directly from the labelled data how those signals
relate to the Fact / Opinion label, instead of us writing rules for it the
way Stage 1 does.

Models like Flan-T5 or a chat-tuned GPT can be asked to "classify this as
Fact or Opinion" directly because they were trained on instruction-following
data. Plain BART was not, so prompting it the same way does not work well -
fine-tuning a classification head on top of it does.

Typical CPU runtime for a few epochs on a few hundred short sentences is a
couple of minutes; a GPU speeds this up substantially.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BartForSequenceClassification,
    BartTokenizerFast,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score

from knowledge_base.knowledge_base import marker_feature_string

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/bart-base"
LABEL2ID = {"Fact": 0, "Opinion": 1}
ID2LABEL = {0: "Fact", 1: "Opinion"}


def _augment(text: str) -> str:
    """Prepend the knowledge-base marker tag to a raw input text."""
    return f"{marker_feature_string(text)} {text}"


class FactOpinionDataset(Dataset):
    """Wraps tokenized (text, label) pairs for the DataLoader."""

    def __init__(self, texts: List[str], labels: Optional[List[str]],
                 tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            [_augment(t) for t in texts],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = [LABEL2ID[l] for l in labels] if labels is not None else None

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


def train_bart_classifier(
    train_texts: List[str],
    train_labels: List[str],
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[str]] = None,
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 256,
    model_id: str = MODEL_ID,
    device: Optional[str] = None,
    seed: int = 11,
) -> Tuple[BartForSequenceClassification, BartTokenizerFast, Dict[str, list]]:
    """
    Fine-tune facebook/bart-base as a 2-way Fact/Opinion classifier.

    Returns the trained model, its tokenizer, and a history dict with the
    per-epoch train loss and validation accuracy/F1 (used for the learning
    curve plot in the pipeline).
    """
    # Seed torch / numpy / CUDA so a re-run reproduces the reported numbers.
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Fine-tuning %s on device: %s", model_id, device)

    tokenizer = BartTokenizerFast.from_pretrained(model_id)
    model = BartForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    train_loader = DataLoader(
        FactOpinionDataset(train_texts, train_labels, tokenizer, max_length),
        batch_size=batch_size, shuffle=True,
    )

    val_loader = None
    if val_texts is not None and val_labels is not None:
        val_loader = DataLoader(
            FactOpinionDataset(val_texts, val_labels, tokenizer, max_length),
            batch_size=batch_size,
        )

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    history = {"train_loss": [], "val_accuracy": [], "val_f1_macro": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += outputs.loss.item()

        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        logger.info("Epoch %d/%d - train loss: %.4f", epoch, epochs, avg_loss)

        if val_loader is not None:
            acc, f1 = _evaluate_loader(model, val_loader, device)
            history["val_accuracy"].append(acc)
            history["val_f1_macro"].append(f1)
            logger.info("Epoch %d/%d - val accuracy: %.4f, val F1-macro: %.4f",
                        epoch, epochs, acc, f1)

    return model, tokenizer, history


@torch.no_grad()
def _evaluate_loader(model, loader, device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


@torch.no_grad()
def predict(model, tokenizer, texts: List[str], device=None,
            batch_size: int = 16, max_length: int = 256) -> List[str]:
    """Run the fine-tuned model over raw texts, returning Fact/Opinion labels."""
    device = device or next(model.parameters()).device
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        encoded = tokenizer(
            [_augment(t) for t in chunk],
            truncation=True, padding=True, max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        batch_preds = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(ID2LABEL[p] for p in batch_preds)
    return preds


def save_model(model, tokenizer, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Saved fine-tuned model to %s", save_dir)


def load_model(save_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizerFast.from_pretrained(save_dir)
    model = BartForSequenceClassification.from_pretrained(save_dir).to(device)
    return model, tokenizer
