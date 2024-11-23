import os
import json
from collections import Counter

from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
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

    # Calculate metrics
    accuracy = accuracy_score(labels_flat, preds_flat)
    class_dist = dict(Counter(preds_flat))  # Get class distribution

    # Print metrics for debugging
    print(f"Accuracy: {accuracy}")
    print(f"Class Distribution: {class_dist}")

    return {"accuracy": accuracy}


class WeightedTrainer(Trainer):
    """Custom Trainer with support for weighted loss and logging."""

    def __init__(self, class_weights, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fn = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.device = device

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to use weighted CrossEntropyLoss.
        Log predictions and loss values for debugging.
        """
        # Extract labels and move to the correct device
        labels = inputs.pop("labels").to(self.device)  # Shape: [batch_size, seq_length]

        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_length, num_classes]

        # Debugging logits and labels before reshaping
        print(f"Logits shape (before): {logits.shape}")  # [batch_size, seq_length, num_classes]
        print(f"Labels shape (before): {labels.shape}")  # [batch_size, seq_length]

        # Reshape logits and labels for loss computation
        logits = logits.view(-1, logits.size(-1))  # Flatten to [total_tokens, num_classes]
        labels = labels.view(-1)  # Flatten to [total_tokens]
        # Debugging logits and labels after reshaping
        print(f"Logits shape (after): {logits.shape}")  # [total_tokens, num_classes]
        print(f"Labels shape (after): {labels.shape}")  # [total_tokens]

        # Compute loss using CrossEntropyLoss
        loss = self.loss_fn(logits, labels)

        # Log debugging information
        preds = torch.argmax(logits, dim=-1)  # Predictions
        print(f"Logits (first 5 tokens): {logits[:5]}")  # Log the first few logits
        print(f"Predictions (first 5 tokens): {preds[:5]}")  # Log the first few predictions
        print(f"True Labels (first 5 tokens): {labels[:5]}")  # Log the first few true labels
        print(f"Loss: {loss.item()}")  # Log the computed loss value

        # Return loss (and optionally model outputs for advanced use cases)
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
            "microsoft/layoutlm-base-uncased",
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )
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
        """
        Create a dataset compatible with BERT.
        """
        # Tokenize inputs
        encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
            is_split_into_words=True,  # Required for token-level alignment
        )

        expanded_labels = []
        for i, input_ids in enumerate(encodings["input_ids"]):
            word_ids = encodings.word_ids(batch_index=i)  # Get token-to-word mapping
            sequence_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    sequence_labels.append(-100)  # Ignore special tokens
                elif word_id != previous_word_id:
                    sequence_labels.append(labels[i][word_id])  # Use word-level labels
                else:
                    sequence_labels.append(-100)  # Ignore subword tokens
                previous_word_id = word_id

            expanded_labels.append(sequence_labels)

        # Convert expanded labels to tensor
        encodings["labels"] = torch.tensor(expanded_labels, dtype=torch.long)

        # Create dataset
        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": expanded_labels,  # Token-level aligned labels
        }
        return Dataset.from_dict(data)

    def train(self, urls, output_dir=f"fine_tuned_model", epochs=10, batch_size=4):
        """Train the model with cached HTML."""
        inputs, labels = self.process_urls(urls)

        if len(inputs) < 2:
            raise ValueError("Dataset is too small. Provide more URLs or check the HTML parser.")

        dataset = self.create_dataset(inputs, labels)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # Compute class weights
        num_classes = len(self.labels)
        unique_classes = np.unique(labels)
        partial_class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels,
        )

        # Fill missing class weights
        complete_class_weights = np.ones(num_classes, dtype=np.float32)  # Default weight 1.0
        for cls, weight in zip(unique_classes, partial_class_weights):
            complete_class_weights[cls] = weight

        class_weights = torch.tensor(complete_class_weights).to(self.device)
        print(f"Class weights: {class_weights}")

        # **Visualization**
        class_counts = Counter(labels)
        classes = [self.id2label[i] for i in range(len(self.labels))]
        counts = [class_counts.get(i, 0) for i in range(len(self.labels))]

        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts)
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=f"logs",
            logging_strategy="steps",
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model(output_dir)

    def predict(self, html_text):
        """Predict the labels for a given HTML text with better word alignment."""
        # Extract features
        tokens, _, xpaths = self.feature_extractor.get_features(html_text)

        # Tokenize inputs
        encodings = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
            is_split_into_words=True,
        )

        inputs = {key: value.to(self.device) for key, value in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).tolist()

        # Combine results
        predictions = [
            {"token": token, "predicted_label": self.id2label[pred], "xpath": xpath}
            for token, pred, xpath in zip(tokens, preds, xpaths)
        ]

        return predictions


if __name__ == "__main__":
    detector = LoginFieldDetector()
