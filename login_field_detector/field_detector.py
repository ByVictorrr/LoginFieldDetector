import os
import json
from collections import Counter
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    Trainer,
    TrainingArguments,
    BertForTokenClassification,
    BertTokenizerFast,
    EarlyStoppingCallback, BertForSequenceClassification,
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from .html_feature_extractor import HTMLFeatureExtractor, LABELS


# Utility Functions
def compute_metrics(pred):
    """Calculate evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    valid_mask = labels != -100
    labels_flat = labels[valid_mask]
    preds_flat = preds[valid_mask]
    return {"accuracy": accuracy_score(labels_flat, preds_flat)}


class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss for imbalanced classes."""

    def __init__(self, class_weights, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fn = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.device = device

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(self.device)
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class LoginFieldDetector:
    """Model for login field detection using BERT."""

    def __init__(self, model_dir=f"fine_tuned_model", labels=LABELS, device=None):
        if not labels:
            raise ValueError("Labels must be provided to initialize the model.")

        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        except Exception as e:
            print(f"Warning: {model_dir} not found or invalid. Using 'bert-base-uncased' as default. Error: {e}")
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )

        self.model = self.model.to(self.device)
        self.feature_extractor = HTMLFeatureExtractor(self.label2id)

    def create_dataset(self, inputs, labels):
        """Align 1D labels with tokenized inputs."""
        encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return Dataset.from_dict(data)

    def process_urls(self, html_list, o_label_ratio=0.001):
        """Preprocess and balance data."""
        inputs, labels = [], []
        for html_text in html_list:
            try:
                tokens, token_labels, _ = self.feature_extractor.get_features(html_text)
                assert len(tokens) == len(token_labels)
                inputs.extend(tokens)
                labels.extend(token_labels)
            except Exception as e:
                print(f"Error processing HTML: {e}")
        return self._filter_o_labels(inputs, labels, o_label_ratio)

    def _filter_o_labels(self, inputs, labels, ratio):
        """Reduce the number of 'O' labels for balance."""
        o_inputs, o_labels, non_o_inputs, non_o_labels = [], [], [], []
        for inp, lbl in zip(inputs, labels):
            if lbl == self.label2id["O"]:
                o_inputs.append(inp)
                o_labels.append(lbl)
            else:
                non_o_inputs.append(inp)
                non_o_labels.append(lbl)
        max_o_count = int(ratio * len(labels))
        filtered_inputs = o_inputs[:max_o_count] + non_o_inputs
        filtered_labels = o_labels[:max_o_count] + non_o_labels
        return filtered_inputs, filtered_labels

    def train(self, html_content_list, output_dir="fine_tuned_model", epochs=10, batch_size=4):
        """Train the model."""
        inputs, labels = self.process_urls(html_content_list)
        dataset = self.create_dataset(inputs, labels)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

        class_weights = self._compute_class_weights(labels)
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir="logs",
            fp16=torch.cuda.is_available(),
        )
        trainer = WeightedTrainer(
            model=self.model,
            device=self.device,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=class_weights,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)

    def _compute_class_weights(self, labels):
        """Compute class weights for imbalanced data."""
        unique_classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels,
        )
        full_weights = np.ones(len(self.labels), dtype=np.float32)
        for cls, weight in zip(unique_classes, weights):
            full_weights[cls] = weight
        return torch.tensor(full_weights).to(self.device)

    def predict(self, html_text):
        """Make predictions on new HTML content."""
        tokens, _, xpaths = self.feature_extractor.get_features(html_text)
        # Tokenize the features
        encodings = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            max_length=512,  # Model's maximum input length
            return_tensors="pt",
        )

        # Move inputs to the model's device
        inputs = {key: value.to(self.device) for key, value in encodings.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        # Convert logits to predicted labels
        predicted_ids = torch.argmax(logits, dim=-1).tolist()

        # Generate predictions with corresponding xpaths
        return [
            {"token": token, "predicted_label": self.id2label[pred], "xpath": xpath}
            for token, pred, xpath in zip(tokens, predicted_ids, xpaths)
        ]


if __name__ == "__main__":
    detector = LoginFieldDetector()
