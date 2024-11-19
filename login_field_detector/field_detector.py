import os
import functools
import json
from collections import defaultdict

from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import torch
from datasets import Dataset
from sklearn.metrics import classification_report

from .html_feature_extractor import HTMLFeatureExtractor, LABELS
from .cached_url import fetch_html

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["USE_GPU"] = "true"


@functools.cache
def get_training_urls():
    """Load training URLs from a JSON file."""
    dir_name = os.path.dirname(__file__)
    with open(os.path.join(dir_name, "training_urls.json"), "r") as flp:
        urls = json.load(flp)
    return urls


TRAINING_URLS = get_training_urls()

GIT_DIR = os.path.dirname(os.path.dirname(__file__))


class LoginFieldDetector:
    def __init__(self, model_dir=f"{GIT_DIR}/fine_tuned_model", labels=None, device=None):
        self.labels = labels or LABELS
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Set the device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if os.path.exists(model_dir):
            # Load fine-tuned model and tokenizer
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = BertForTokenClassification.from_pretrained(
                model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        else:
            # Load base model for first-time training
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.model = BertForTokenClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )

        # Move model to the correct device
        self.model = self.model.to(self.device)
        self.model_dir = model_dir
        self.feature_extractor = HTMLFeatureExtractor(self.label2id)

    def tokenize_and_align_labels(self, tokens, labels):
        """Tokenize and align input tokens with their labels."""
        inputs = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        word_ids = inputs.word_ids(batch_index=0)

        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:  # Special tokens ([CLS], [SEP])
                aligned_labels.append(-100)
            else:
                aligned_labels.append(labels[word_id])  # Align label to token

        inputs["labels"] = torch.tensor([aligned_labels])

        # Move tensors to the correct device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def process_urls(self, url_list):
        """Fetch HTML, parse it, and generate token-label pairs."""
        all_tokens, all_labels, all_xpaths = [], [], []
        for url in url_list:
            print(f"Processing {url}")
            html = fetch_html(url)
            if html:
                tokens, labels, xpaths = self.feature_extractor.get_features(html)
                if tokens and labels:
                    all_tokens.append(tokens)
                    all_labels.append(labels)
                    all_xpaths.append(xpaths)
        return all_tokens, all_labels, all_xpaths

    def create_dataset(self, tokens_list, labels_list):
        """Create a dataset from tokens and labels."""
        data = defaultdict(list)
        for tokens, labels in zip(tokens_list, labels_list):
            inputs = self.tokenize_and_align_labels(tokens, labels)
            for key, value in inputs.items():
                data[key].append(value.squeeze(0))  # Add one example per URL

        return Dataset.from_dict(data)

    def train(self, urls=TRAINING_URLS, output_dir=f"{GIT_DIR}/fine_tuned_model", epochs=5, batch_size=4,
              learning_rate=5e-5):
        """Train the model with cached HTML."""
        # Step 1: Fetch and cache HTML for all URLs
        cached_html_data = []
        for url in urls:
            html = fetch_html(url)
            if html:
                cached_html_data.append((url, html))

        if len(cached_html_data) < 2:
            raise ValueError("Dataset is too small. Provide more URLs or check the HTML parser.")

        # Step 2: Process the HTML data to create a dataset
        tokens_list, labels_list, _ = [], [], []
        for url, html in cached_html_data:
            tokens, labels, _ = self.feature_extractor.get_features(html)
            tokens_list.append(tokens)
            labels_list.append(labels)

        # Create dataset
        dataset = self.create_dataset(tokens_list, labels_list)

        # Split dataset
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # Step 3: Training configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            max_grad_norm=1.0,  # Gradient clipping
            logging_dir=f"{GIT_DIR}/logs",
            logging_strategy="steps",
            fp16=True,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        # Save model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Evaluate the model
        self.evaluate_model(val_dataset)

    def evaluate_model(self, dataset):
        """Evaluate the model on a given dataset."""
        predictions, true_labels = [], []

        for example in dataset:
            inputs = {key: torch.tensor(value).unsqueeze(0).to(self.device) for key, value in example.items() if
                      key != "labels"}
            labels = torch.tensor(example["labels"]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            true = labels.squeeze().tolist()

            filtered_preds = [p for p, t in zip(preds, true) if t != -100]
            filtered_true = [t for t in true if t != -100]

            predictions.extend(filtered_preds)
            true_labels.extend(filtered_true)

        target_labels = [self.id2label[label] for label in sorted(set(true_labels + predictions))]
        print(classification_report(true_labels, predictions, target_names=target_labels))

    def predict(self, html_text):
        tokens, labels, xpaths = self.feature_extractor.get_features(html_text)
        features = self.tokenize_and_align_labels(tokens, labels)
        outputs = self.model(**features)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()

        predictions = []
        for token, pred_id, xpath in zip(tokens, predicted_ids, xpaths):
            label = self.id2label[pred_id]
            predictions.append({"token": token, "predicted_label": label, "xpath": xpath})

        return predictions


if __name__ == "__main__":
    detector = LoginFieldDetector()
    detector.train()
