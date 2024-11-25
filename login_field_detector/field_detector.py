import os
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
    LayoutLMTokenizerFast,
    LayoutLMForSequenceClassification,
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_dir)
            self.model = LayoutLMForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        except Exception as e:
            print(f"Warning: {model_dir} not found or invalid. Using 'microsoft/layoutlmv2-base-uncased' as default. "
                  f"Error: {e}")
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
            self.model = LayoutLMForSequenceClassification.from_pretrained(
                "microsoft/layoutlmv2-base-uncased",
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )

        self.model = self.model.to(self.device)
        self.feature_extractor = HTMLFeatureExtractor(self.label2id)

    def create_dataset(self, inputs, labels, bboxes):
        """Align 1D labels with tokenized inputs."""
        encodings = self.tokenizer(
            inputs,
            bboxes=bboxes,
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

    def process_urls(self, file_list, o_label_ratio=0.001):
        """Preprocess, balance data, and include bounding boxes."""
        inputs, labels, bboxes = [], [], []
        for file_path in file_list:
            try:
                # Extract features including bounding boxes
                tokens, token_labels, _, token_bboxes = self.feature_extractor.get_features(file_path=file_path)
                assert len(tokens) == len(token_labels) == len(token_bboxes)

                # Extend the inputs, labels, and bboxes
                inputs.extend(tokens)
                labels.extend(token_labels)
                bboxes.extend(token_bboxes)
            except Exception as e:
                print(f"Error processing HTML: {e}")

        # Balance and filter inputs, labels, and bounding boxes
        return self._filter_unlabeled_labels(inputs, labels, bboxes, o_label_ratio)

    def _filter_unlabeled_labels(self, inputs, labels, bboxes, ratio):
        """Reduce the number of 'UNLABELED' labels for balance, including bboxes."""
        o_inputs, o_labels, o_bboxes = [], [], []
        non_o_inputs, non_o_labels, non_o_bboxes = [], [], []

        # Separate 'UNLABELED' and labeled data
        for inp, lbl, bbox in zip(inputs, labels, bboxes):
            if lbl == self.label2id["UNLABELED"]:
                o_inputs.append(inp)
                o_labels.append(lbl)
                o_bboxes.append(bbox)
            else:
                non_o_inputs.append(inp)
                non_o_labels.append(lbl)
                non_o_bboxes.append(bbox)

        # Limit the 'UNLABELED' data based on the ratio
        max_o_count = int(ratio * len(labels))
        filtered_inputs = o_inputs[:max_o_count] + non_o_inputs
        filtered_labels = o_labels[:max_o_count] + non_o_labels
        filtered_bboxes = o_bboxes[:max_o_count] + non_o_bboxes

        return filtered_inputs, filtered_labels, filtered_bboxes

    def train(self, file_list, output_dir="fine_tuned_model", epochs=10, batch_size=4):
        """Train the model."""
        inputs, labels, bboxes = self.process_urls(file_list)
        dataset = self.create_dataset(inputs, labels, bboxes)
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
        print("Starting training...")
        trainer.train()

        # Save model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.evaluate(val_dataset)

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

    def predict(self, url=None, file_path=None):
        """Make predictions on new HTML content."""
        tokens, _, xpaths, bboxes = self.feature_extractor.get_features(url=url, file_path=file_path)
        # Tokenize the features
        encodings = self.tokenizer(
            tokens,
            bboxes=bboxes,
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

    def visualize_class_distribution(self, labels):
        """Plot class distribution."""
        counts = Counter(labels)
        class_names = [self.id2label[i] for i in range(len(self.labels))]
        class_counts = [counts.get(i, 0) for i in range(len(self.labels))]

        plt.figure(figsize=(12, 6))
        plt.bar(class_names, class_counts, color="skyblue")
        plt.xlabel("Class Labels")
        plt.ylabel("Frequency")
        plt.title("Class Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, true_labels, pred_labels):
        """Plot confusion matrix with improved x-axis label spacing."""
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(self.labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)

        # Set figure size to provide more space
        fig, ax = plt.subplots(figsize=(12, 12))  # Adjust the size as needed
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)

        # Increase padding for x-axis labels
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")  # Rotate and align x-axis labels
        plt.subplots_adjust(bottom=0.25)  # Add padding to the bottom to avoid label cutoff

        # Set title and show the plot
        plt.title("Confusion Matrix")
        plt.show()
    def evaluate(self, dataset):
        """Evaluate the model on a dataset and plot confusion matrix."""
        predictions, true_labels = [], []
        for example in dataset:
            # Prepare inputs and labels
            inputs = {key: torch.tensor(value).unsqueeze(0).to(self.device) for key, value in example.items() if
                      key != "labels"}
            labels = torch.tensor(example["labels"]).unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits

            # Get predictions and true labels, filtering out ignored tokens (-100)
            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            true = labels.squeeze().tolist()

            filtered_preds = [p for p, t in zip(preds, true) if t != -100]
            filtered_true = [t for t in true if t != -100]

            predictions.extend(filtered_preds)
            true_labels.extend(filtered_true)

        # Define all possible labels (use the full id2label mapping)
        all_labels = sorted(self.id2label.keys())  # Ensure all labels are included
        target_labels = [self.id2label[label] for label in all_labels]

        self.visualize_class_distribution(true_labels)
        self.plot_confusion_matrix(true_labels, predictions)
        # Generate the classification report
        print(classification_report(
            true_labels,
            predictions,
            target_names=target_labels,
            labels=all_labels  # Ensure classification_report includes all classes
        ))


if __name__ == "__main__":
    detector = LoginFieldDetector()
