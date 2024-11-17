import functools
import json
import os.path
import re
from collections import defaultdict
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from .html_serializer import parse_html, LABELS, fetch_html


@functools.cache
def get_training_urls():
    """Load training URLs from a JSON file."""
    dir_name = os.path.dirname(__file__)
    with open(os.path.join(dir_name, "training_urls.json"), "r") as flp:
        data = json.load(flp)

    # Extract all URLs under the "urls" keys in the JSON
    urls = []
    for category in data:
        urls.extend(category["urls"])

    return urls


TRAINING_URLS = get_training_urls()


def clean_token(token):
    """Normalize and clean tokens to reduce noise."""
    token = token.lower()
    token = re.sub(r"[^a-z0-9@._-]", "", token)  # Remove unwanted characters
    return token


def split_dataset(urls, test_size=0.2, random_state=42):
    """Split URLs into training and validation sets."""
    return train_test_split(urls, test_size=test_size, random_state=random_state)


class LoginFieldDetector:
    def __init__(self, model_name="bert-base-uncased", labels=None):
        self.labels = labels or LABELS
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def tokenize_and_align_labels(self, tokens, labels):
        """Tokenize tokens and align labels."""
        inputs = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        word_ids = inputs.word_ids(batch_index=0)  # Map back to original words
        aligned_labels = []

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(labels[word_id])

        inputs["labels"] = torch.tensor([aligned_labels])
        return inputs

    def process_urls(self, url_list):
        """Process multiple URLs and generate tokens/labels."""
        tokens, labels = [], []
        for url in url_list:
            print(f"Processing {url}")
            html = fetch_html(url)  # Fetch HTML using requests
            if html:
                t, l = parse_html(html, self.label2id)
                tokens.extend([clean_token(token) for token in t])
                labels.extend(l)
        return tokens, labels

    def create_dataset(self, url_list):
        """Create a dataset from a list of URLs."""
        tokens, labels = self.process_urls(url_list)
        inputs = self.tokenize_and_align_labels(tokens, labels)

        # Consolidate features into a dataset
        data = defaultdict(list)
        for key, value in inputs.items():
            data[key].append(value.squeeze(0))  # Remove batch dimension

        return Dataset.from_dict(data)

    def train(self, urls=TRAINING_URLS, output_dir="./results", epochs=5, batch_size=4, learning_rate=5e-5):
        """Train the model."""
        # Create a single dataset
        train_dataset = self.create_dataset(urls)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        self.evaluate_model(train_dataset)

    def evaluate_model(self, dataset):
        """Evaluate the model on a given dataset."""
        predictions, true_labels = [], []

        for example in dataset:
            # Convert inputs to PyTorch tensors
            inputs = {key: torch.tensor(value).unsqueeze(0) for key, value in example.items() if key != "labels"}
            labels = torch.tensor(example["labels"]).unsqueeze(0)  # Labels as tensor

            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Predictions and true labels
            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            true = labels.squeeze().tolist()

            # Filter out special labels (-100)
            filtered_preds = [p for p, t in zip(preds, true) if t != -100]
            filtered_true = [t for t in true if t != -100]

            predictions.extend(filtered_preds)
            true_labels.extend(filtered_true)

        # Check label alignment
        unique_labels = sorted(set(true_labels + predictions))
        target_labels = [self.id2label[label] for label in unique_labels]

        # Generate classification report
        print(classification_report(true_labels, predictions, target_names=target_labels))

    def predict(self, html):
        tokens, _ = parse_html(html, self.label2id)

        if not tokens:
            print("No tokens found in the HTML.")
            return []

        inputs = self.tokenize_and_align_labels(tokens, [0] * len(tokens))
        outputs = self.model(**inputs)

        probabilities = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

        results = []
        for token, pred, prob in zip(tokens, predictions, probabilities):
            if pred != -100:  # Ignore special tokens
                label = self.id2label[pred]
                confidence = prob[pred]
                if label != "O":  # Ignore irrelevant tokens
                    results.append({"token": token, "label": label, "confidence": confidence})

        return results


if __name__ == "__main__":
    detector = LoginFieldDetector()
    detector.train()
