import os
import functools
import json
from transformers import (
    Trainer,
    TrainingArguments,
    BertTokenizerFast,
    BertForSequenceClassification,
)
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(model_dir):
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )

        self.model = self.model.to(self.device)
        self.feature_extractor = HTMLFeatureExtractor(self.label2id)

    def process_urls(self, url_list, o_label_ratio_threshold=0.1):
        """
        Fetch HTML, parse it, and generate feature-label pairs with a threshold on the ratio of "O" labels.

        :param url_list: List of URLs to process.
        :param o_label_ratio_threshold: Maximum allowable ratio of "O" labels in the data.
        :return: Filtered inputs and labels.
        """
        all_inputs, all_labels = [], []

        for url in tqdm(url_list, desc="Processing URLs"):
            print(f"Processing {url}")
            try:
                html = fetch_html(url)
                if html:
                    inputs, labels, _ = self.feature_extractor.get_features(html)
                    all_inputs.extend(inputs)
                    all_labels.extend(labels)
            except Exception as e:
                print(f"Error processing {url}: {e}")

        return all_inputs, all_labels

    def create_dataset(self, inputs, labels):
        """Create a dataset compatible with BERT."""
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

    def evaluate_model(self, dataset):
        """Evaluate the model on a given dataset."""
        predictions, true_labels = [], []

        for example in dataset:
            # Convert inputs to tensors
            input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
            labels = torch.tensor(example["labels"], dtype=torch.long).unsqueeze(0).to(self.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits

            # Get predictions
            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            true = labels.squeeze().tolist()

            # Append results
            predictions.extend(preds if isinstance(preds, list) else [preds])
            true_labels.extend(true if isinstance(true, list) else [true])

        # Generate evaluation metrics
        print("Evaluation Results:")
        print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        target_names = [self.id2label[i] for i in sorted(set(true_labels + predictions))]
        print(classification_report(true_labels, predictions, target_names=target_names))

    def train(self, urls=TRAINING_URLS, output_dir=f"{GIT_DIR}/fine_tuned_model", epochs=1, batch_size=4):
        """Train the model with cached HTML."""
        inputs, labels = self.process_urls(urls)

        if len(inputs) < 2:
            raise ValueError("Dataset is too small. Provide more URLs or check the HTML parser.")

        dataset = self.create_dataset(inputs, labels)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=f"{GIT_DIR}/logs",
            logging_strategy="steps",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.evaluate_model(val_dataset)

    def predict(self, html_text):
        """Predict the labels for a given HTML text."""
        tokens, _, xpaths = self.feature_extractor.get_features(html_text)

        encodings = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {key: value.to(self.device) for key, value in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1).tolist()

        predictions = [{"token": token, "predicted_label": self.id2label[pred], "xpath": xpath}
                       for token, pred, xpath in zip(tokens, predicted_ids, xpaths)]
        return predictions


if __name__ == "__main__":
    detector = LoginFieldDetector()
    detector.train()
