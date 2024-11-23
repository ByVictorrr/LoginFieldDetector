import os
import json
from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    LayoutLMForTokenClassification,
    LayoutLMTokenizerFast,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from .html_feature_extractor import HTMLFeatureExtractor, LABELS
from .cached_url import fetch_html

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["USE_GPU"] = "true"


def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Flatten arrays
    labels_flat = labels[labels != -100]
    preds_flat = preds[labels != -100]

    # Print classification report
    print(classification_report(labels_flat, preds_flat, target_names=LABELS))

    accuracy = accuracy_score(labels_flat, preds_flat)

    return {"accuracy": accuracy}


class WeightedTrainer(Trainer):
    """Custom Trainer with support for weighted loss."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and handle logits/labels alignment."""
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_length, num_classes]
        labels = inputs["labels"]  # Shape: [batch_size, seq_length]

        # Flatten logits and labels
        logits = logits.view(-1, logits.size(-1))  # Shape: [batch_size * seq_length, num_classes]
        labels = labels.view(-1)  # Shape: [batch_size * seq_length]

        # Check alignment
        if logits.size(0) != labels.size(0):
            raise ValueError(f"Logits and labels shape mismatch: {logits.size(0)} != {labels.size(0)}")

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class LoginFieldDetector:
    """Main class for training and evaluating the LayoutLM model for login field detection."""

    def __init__(self, model_dir="fine_tuned_model", labels=None, device=None):
        self.labels = labels or LABELS
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained model and tokenizer
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.model = LayoutLMForTokenClassification.from_pretrained(
            "microsoft/layoutlmv2-base-uncased",
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(self.device)
        self.model.gradient_checkpointing_enable()

        self.feature_extractor = HTMLFeatureExtractor(self.label2id)

    def process_urls(self, urls, o_label_ratio=0.001):
        """Fetch HTML, extract features, and preprocess."""
        inputs, labels = [], []
        for url in tqdm(urls, desc="Processing URLs"):
            try:
                html = fetch_html(url)
                if html:
                    tokens, token_labels, _ = self.feature_extractor.get_features(html)
                    assert len(tokens) == len(
                        token_labels), f"Tokens and labels mismatch: {len(tokens)} != {len(token_labels)}"
                    inputs.extend(tokens)
                    labels.extend(token_labels)
            except Exception as e:
                print(f"Error processing {url}: {e}")

        return self._filter_o_labels(inputs, labels, o_label_ratio)

    def _filter_o_labels(self, inputs, labels, ratio):
        """Filter 'O' labels to balance dataset."""
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

    def create_dataset(self, inputs, labels):
        encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        adjusted_labels = []
        for label, input_ids in zip(labels, encodings["input_ids"]):
            if isinstance(label, int):
                adjusted_labels.append([label] * len(input_ids))  # Sequence-level labels
            elif isinstance(label, list):
                adjusted_labels.append(label + [-100] * (len(input_ids) - len(label)))  # Token-level labels

        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(adjusted_labels, dtype=torch.long),
        })

    def train(self, urls, output_dir="fine_tuned_model", epochs=10, batch_size=1):
        """Train the model."""
        inputs, labels = self.process_urls(urls)

        # Check for sufficient data
        if len(inputs) < 2:
            raise ValueError("Insufficient data for training.")

        dataset = self.create_dataset(inputs, labels)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

        # Compute class weights
        class_weights = self._compute_class_weights(labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_dir="logs",
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=class_weights,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(output_dir)

    def _compute_class_weights(self, labels):
        """Compute weights for imbalanced classes."""
        weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def predict(self, html_text):
        """Run predictions on given HTML."""
        if not html_text or not html_text.strip():
            raise ValueError("HTML input is empty or invalid.")

        tokens, _, xpaths = self.feature_extractor.get_features(html_text)

        # Debugging tokens
        print(f"Tokens: {tokens}")
        if not tokens:
            return [{"token": None, "label": "O", "xpath": None}]

        encodings = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in encodings.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Flatten logits and get predictions
        logits = logits.view(-1, logits.size(-1))
        preds = torch.argmax(logits, dim=-1).tolist()
        return [{"token": t, "label": self.id2label[p], "xpath": x} for t, p, x in zip(tokens, preds, xpaths)]


if __name__ == "__main__":
    detector = LoginFieldDetector()
