import logging
import json
from collections import Counter, defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from torch import softmax
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    BertTokenizerFast,
    BertForSequenceClassification,
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from .html_feature_extractor import HTMLFeatureExtractor, LABELS
from .html_fetcher import HTMLFetcher

log = logging.getLogger(__name__)

# Utility Functions
def compute_metrics(pred):
    """Calculate evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    valid_mask = labels != -100
    labels_flat = labels[valid_mask]
    preds_flat = preds[valid_mask]
    return {"accuracy": accuracy_score(labels_flat, preds_flat)}




class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for handling inputs and labels.
    """

    def __init__(self, inputs, labels, tokenizer, max_length=512):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize input and prepare tensor format
        encoding = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dim
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def create_dataset(self, inputs, labels, batch_size, shuffle=True):
    """Create a PyTorch DataLoader from inputs and labels."""
    dataset = CustomDataset(inputs, labels, self.tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        log.info(f"Using device: {self.device}")
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        except Exception as e:
            log.warning(f"Warning: {model_dir} not found or invalid. Using 'bert-base-uncased' as default. Error: {e}")
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        self.model = self.model.to(self.device)
        self.feature_extractor = HTMLFeatureExtractor(self.label2id)
        self.url_loader = HTMLFetcher()

    def create_dataset(self, inputs, labels, batch_size):
        """Align 1D labels with tokenized inputs."""
        encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        })
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def process_urls(self, urls, o_label_ratio=0.001):
        """Preprocess, balance data, and include bounding boxes."""
        inputs, labels = [], []
        for url, text in self.url_loader.fetch_all(urls).items():
            try:
                # Extract features including bounding boxes
                tokens, token_labels, _ = self.feature_extractor.get_features(text)
                assert len(tokens) == len(token_labels)

                # Extend the inputs, labels, and bboxes
                inputs.extend(tokens)
                labels.extend(token_labels)
            except Exception as e:
                log.warning(f"Error processing {url}: {e}")

        # Balance and filter inputs, labels, and bounding boxes
        return self._filter_unlabeled_labels(inputs, labels, o_label_ratio)

    def _filter_unlabeled_labels(self, inputs, labels, ratio):
        """Reduce the number of 'UNLABELED' labels for balance, including bboxes."""
        o_inputs, o_labels = [], []
        non_o_inputs, non_o_labels = [], []
        # Separate 'UNLABELED' and labeled data
        for inp, lbl in zip(inputs, labels):
            if lbl == self.label2id["UNLABELED"]:
                o_inputs.append(inp)
                o_labels.append(lbl)
            else:
                non_o_inputs.append(inp)
                non_o_labels.append(lbl)

        # Limit the 'UNLABELED' data based on the ratio
        max_o_count = int(ratio * len(labels))
        filtered_inputs = o_inputs[:max_o_count] + non_o_inputs
        filtered_labels = o_labels[:max_o_count] + non_o_labels

        return filtered_inputs, filtered_labels

    def train(self, urls, output_dir="fine_tuned_model", epochs=10, batch_size=8):
        """
        Train the model using PyTorch DataLoader with class weights.
        """
        # Fetch and preprocess data
        inputs, labels = self.process_urls(urls)

        # Create PyTorch Dataset
        dataset = CustomDataset(inputs, labels, self.tokenizer)

        # Split into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Compute class weights
        class_weights = self._compute_class_weights(labels)

        # Move class weights to the correct device
        class_weights = class_weights.to(self.device)

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Training Loss: {avg_loss:.4f}")

            # Validation loop
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)

                    val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Save model and tokenizer
        torch.save(self.model.state_dict(), f"{output_dir}/model.pt")
        self.tokenizer.save_pretrained(output_dir)

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

    from collections import defaultdict

    def predict(self, url, probability_threshold=0.5):
        """
        Make predictions on new HTML content, allowing multiple entries per label
        above a specified probability threshold and sorted by probability.
        """
        html_content = self.url_loader.fetch_html(url)
        tokens, _, xpaths = self.feature_extractor.get_features(html_content)

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

        # Apply softmax to logits to get probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Convert logits to predicted labels
        predicted_ids = torch.argmax(logits, dim=-1).tolist()

        # Group predictions by labels, filtering by probability threshold
        label_predictions = defaultdict(list)  # Use defaultdict for easier grouping
        for token, pred_id, prob_vector, xpath in zip(tokens, predicted_ids, probabilities, xpaths):
            label = self.id2label[pred_id]
            prob = prob_vector[pred_id].item()

            # Only include predictions above the probability threshold
            if prob >= probability_threshold:
                label_predictions[label].append({
                    "token": token,
                    "xpath": xpath,
                    "probability": prob,
                })

        # Normalize probabilities for each label group and sort by probability (highest first)
        for label, predictions in label_predictions.items():
            # Sort by probability (highest probability first)
            predictions.sort(key=lambda pred: pred["probability"], reverse=True)

            # Normalize probabilities
            total_prob = sum(pred["probability"] for pred in predictions)
            for pred in predictions:
                pred["normalized_probability"] = pred["probability"] / total_prob

        # Return predictions sorted by probability for each label
        return label_predictions

    def visualize_class_distribution(self, labels):
        """Plot class distribution."""
        counts = Counter(labels)
        class_names = [self.id2label[i] for i in range(len(self.labels))]
        class_counts = [counts.get(i, 0) for i in range(len(self.labels))]
        plt.figure(figsize=(14, 10))
        plt.bar(class_names, class_counts, color="skyblue")
        plt.xlabel("Class Labels")
        plt.title("Class Distribution")
        plt.ylabel("Frequency")
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

            predictions.append(preds)
            true_labels.append(true)

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
