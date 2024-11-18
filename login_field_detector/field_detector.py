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

from .html_serializer import parse_html, LABELS, fetch_html

os.environ["TOKENIZERS_PARALLELISM"] = "true"


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
        return inputs

    def process_urls(self, url_list):
        """Fetch HTML, parse it, and generate token-label pairs."""
        all_tokens, all_labels = [], []
        for url in url_list:
            print(f"Processing {url}")
            html = fetch_html(url)
            if html:
                tokens, labels = parse_html(html, self.label2id)
                if tokens and labels:
                    all_tokens.append(tokens)
                    all_labels.append(labels)
        return all_tokens, all_labels

    def create_dataset(self, url_list):
        """Create a HuggingFace dataset from tokenized input and labels."""
        tokens_list, labels_list = self.process_urls(url_list)

        data = defaultdict(list)
        for tokens, labels in zip(tokens_list, labels_list):
            inputs = self.tokenize_and_align_labels(tokens, labels)
            for key, value in inputs.items():
                data[key].append(value.squeeze(0))  # Add one example per URL

        return Dataset.from_dict(data)

    def train(self, urls=TRAINING_URLS, output_dir=f"{GIT_DIR}/results", epochs=5, batch_size=4, learning_rate=5e-5):
        """Train the model."""
        dataset = self.create_dataset(urls)

        if len(dataset) < 2:
            raise ValueError("Dataset is too small. Provide more URLs or check the HTML parser.")

        # Split the dataset
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            max_grad_norm=1.0,  # Gradient clipping
            logging_dir=f"{GIT_DIR}/logs",
            logging_steps=10,
            save_total_limit=2,  # Limit checkpoint saves
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
        self.evaluate_model(val_dataset)

    def evaluate_model(self, dataset):
        """Evaluate the model on a given dataset."""
        predictions, true_labels = [], []

        for example in dataset:
            inputs = {key: torch.tensor(value).unsqueeze(0) for key, value in example.items() if key != "labels"}
            labels = torch.tensor(example["labels"]).unsqueeze(0)

            with torch.no_grad():
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

        target_labels = [self.id2label[label] for label in sorted(set(true_labels + predictions))]
        print(classification_report(true_labels, predictions, target_names=target_labels))

    def predict(self, html_text):
        """Generate predictions for raw HTML."""
        tokens, _ = parse_html(html_text, self.label2id)
        inputs = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        predicted_labels = [self.id2label[p] for p in predictions if p != -100]
        return list(zip(tokens, predicted_labels))


if __name__ == "__main__":
    detector = LoginFieldDetector()
    detector.train()
