import json
import logging
import os.path
import shutil
import time
from collections import Counter, defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    Trainer,
    TrainingArguments,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast, AutoModelForSequenceClassification, AutoTokenizer, DistilBertConfig,
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import hf_hub_download
from .html_feature_extractor import HTMLFeatureExtractor, LABELS
from .html_fetcher import HTMLFetcher

log = logging.getLogger(__name__)

LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


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


def download_model_files(root_dir):
    """Downloads the necessary model files from Hugging Face Hub and ensures they are in the model directory root."""
    model_dir = os.path.join(root_dir, "download_model")
    repo_id = "byvictorrr/html-login-field-detector"
    os.makedirs(model_dir, exist_ok=True)

    for filename in ["model.safetensors", "config.json", "tokenizer.json"]:
        try:
            # Download the file
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=model_dir,
                force_filename=filename
            )
            # Move file to the root of model_dir if necessary
            if os.path.basename(file_path) not in os.listdir(model_dir):
                destination_path = os.path.join(model_dir, filename)
                shutil.copy(file_path, destination_path)
                log.info(f"Copied {filename} to {destination_path}")
            else:
                log.info(f"{filename} is already in the correct location: {file_path}")
        except Exception as e:
            log.error(f"Error downloading {filename}: {e}")

    # Verify all files exist
    for filename in ["model.safetensors", "config.json", "tokenizer.json"]:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            log.error(f"{filename} is missing from {model_dir}!")
        else:
            log.info(f"{filename} exists at {file_path}")

    return model_dir


class LoginFieldDetector:
    """Model for login field detection using BERT."""

    def __init__(self, model_dir=None, labels=LABELS, device=None):
        if not labels:
            raise ValueError("Labels must be provided to initialize the model.")
        self.labels = labels
        self.label2id = LABEL2ID
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.urls_path = os.path.join(root_dir, "dataset", "training_urls.json")
        log.info(f"Using device: {self.device}")
        # Download model files if not provided
        if model_dir is None:
            log.info("Downloading model files from Hugging Face...")
            model_dir = download_model_files(root_dir)
            log.info(f"{model_dir} includes: {os.listdir(model_dir)}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
            log.info("Not need to train this model because it has been fetched.")
        except Exception as e:
            log.warning(f"Failed to load model from {model_dir}: {e}. Falling back to 'distilbert-base-uncased'.")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            # Create a config with model_type
            config = DistilBertConfig(
                num_labels=len(labels),
                id2label=self.id2label,
                label2id=self.label2id,
                model_type="distilbert"  # Explicitly set the model_type
            )
            self.model = DistilBertForSequenceClassification(config)

        self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model)
        self.writer = SummaryWriter(log_dir="logs")
        self.feature_extractor = HTMLFeatureExtractor(self.label2id)
        self.url_loader = HTMLFetcher()
        self.model_dir = model_dir

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

    def process_urls(self, urls, force=False, o_label_ratio=0.5, screenshots=False):
        """Preprocess the urls for training the model and balance the data

        :param urls: List of URLs to process.
        :param force: Force re-fetching all URLs.
        :param screenshots: Whether you want to screenshot the urls being processed
        :param o_label_ratio: Ratio of 'UNLABELED' labels to retain.
        :return: Filtered inputs and labels.
        """
        inputs, labels = [], []
        for url, text in self.url_loader.fetch_all(urls, force=force, screenshot=screenshots).items():
            try:
                tokens, token_labels, _ = self.feature_extractor.get_features(text)
                assert len(tokens) == len(token_labels)
                inputs.extend(tokens)
                labels.extend(token_labels)
            except Exception as e:
                log.warning(f"Error processing {url}: {e}")

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

    def train(self, urls=None, epochs=10, batch_size=16, force=False, screenshots=False):
        """Train the model.

        :param urls: List of URLs to fetch and process for training.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param screenshots: Whether you want to screenshot the urls being processed
        :param force: Force re-fetching and reprocessing all URLs.
        """
        start_time = time.time()
        if not urls:
            with open(self.urls_path, "r") as flp:
                urls = json.load(flp)

        log.info("Collecting data...")
        inputs, labels = self.process_urls(urls, force=force, screenshots=screenshots)

        log.info("Preparing datasets...")
        dataset = self.create_dataset(inputs, labels)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()
        class_weights = self._compute_class_weights(labels)

        log.info("Starting training...")
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir="logs",
            gradient_accumulation_steps=4,  # Accumulate gradient over 4 steps
            remove_unused_columns=False,
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
        trainer.train(resume_from_checkpoint=False if force else True)
        # Log the time taken to TensorBoard
        elapsed_time = time.time() - start_time
        self.writer.add_scalar("Training/Time_Seconds", elapsed_time)
        self.writer.close()
        log.info(f"Training completed in {elapsed_time:.2f} seconds.")

        # Save model and tokenizer
        log.info("Saving model and tokenizer...")
        self.model.save_pretrained(self.model_dir)  # Save model weights
        self.tokenizer.save_pretrained(self.model_dir)  # Save tokenizer
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

    def predict(self, url=None, html_content=None, probability_threshold=0.9):
        """Make predictions on new HTML content.

        Allowing multiple entries per label above a specified probability threshold and sorted by probability.
        """
        if not url and not html_content:
            raise ValueError(f"Need to pass {html_content=} or {url=}.")
        if not (html_content := self.url_loader.fetch_html(url) if url else html_content):
            return []

        tokens, _, xpaths = self.feature_extractor.get_features(html_content)
        # Tokenize the features
        if not tokens:
            return []
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
        log.info(classification_report(
            true_labels,
            predictions,
            target_names=target_labels,
            labels=all_labels  # Ensure classification_report includes all classes
        ))


if __name__ == "__main__":
    # Get logging level from environment variable, default to WARNING
    log_level = os.getenv("PYTHONLOGGING", "INFO")
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    detector = LoginFieldDetector(model_dir=output_dir)
    detector.train(force=True, epochs=50)
