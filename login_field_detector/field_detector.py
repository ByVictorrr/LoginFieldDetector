import os
import functools
import json
from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    BertTokenizerFast,
    BertForSequenceClassification,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


class WeightedTrainer(Trainer):
    """Custom Trainer to apply class weights."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels
        labels = inputs["labels"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate weighted loss
        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


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

    def process_urls(self, url_list, o_label_ratio=0.001):
        all_inputs, all_labels = [], []

        for url in tqdm(url_list, desc="Processing URLs"):
            print(f"Processing {url}")
            try:
                html = fetch_html(url)
                if html:
                    inputs, labels, _ = self.feature_extractor.get_features(html)
                    if inputs and labels:
                        all_inputs.extend(inputs)
                        all_labels.extend(labels)
            except Exception as e:
                print(f"Error processing {url}: {e}")

        # Separate "O" labels and others
        o_inputs, o_labels = [], []
        non_o_inputs, non_o_labels = [], []

        for input_data, label in zip(all_inputs, all_labels):
            if label == self.label2id["O"]:
                o_inputs.append(input_data)
                o_labels.append(label)
            else:
                non_o_inputs.append(input_data)
                non_o_labels.append(label)

        # Limit the number of "O" labels based on the ratio
        max_o_count = int(o_label_ratio * len(all_labels))
        limited_o_inputs = o_inputs[:max_o_count]
        limited_o_labels = o_labels[:max_o_count]

        # Combine filtered "O" labels with the others
        filtered_inputs = limited_o_inputs + non_o_inputs
        filtered_labels = limited_o_labels + non_o_labels

        return filtered_inputs, filtered_labels

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
            input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0).to(self.device)
            labels = torch.tensor(example["labels"], dtype=torch.long).unsqueeze(0).to(self.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            true = labels.squeeze().tolist()

            predictions.extend(preds if isinstance(preds, list) else [preds])
            true_labels.extend(true if isinstance(true, list) else [true])

        print("Evaluation Results:")
        print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        target_names = [self.id2label[i] for i in sorted(set(true_labels + predictions))]
        print(classification_report(true_labels, predictions, target_names=target_names))

    def train(self, urls=TRAINING_URLS, output_dir=f"{GIT_DIR}/fine_tuned_model", epochs=10, batch_size=4):
        """Train the model with cached HTML."""
        inputs, labels = self.process_urls(urls)

        if len(inputs) < 2:
            raise ValueError("Dataset is too small. Provide more URLs or check the HTML parser.")

        dataset = self.create_dataset(inputs, labels)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # Ensure class weights are computed for all possible classes
        num_classes = len(self.labels)  # Total number of classes
        unique_classes = np.unique(labels)  # Classes present in the data

        # Compute weights only for present classes
        partial_class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels,
        )

        # Fill missing class weights with a default value (e.g., 1.0)
        complete_class_weights = np.ones(num_classes, dtype=np.float32)  # Default weight 1.0
        for cls, weight in zip(unique_classes, partial_class_weights):
            complete_class_weights[cls] = weight

        # Convert to torch tensor
        class_weights = torch.tensor(complete_class_weights).to(self.device)
        print(f"Class weights: {class_weights}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",  # Evaluate every `eval_steps`
            eval_steps=500,  # Customize the evaluation frequency
            save_strategy="steps",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=f"{GIT_DIR}/logs",
            logging_strategy="steps",
            metric_for_best_model="accuracy",  # Metric to monitor
            load_best_model_at_end=True,  # Load the best model when training stops
            fp16=torch.cuda.is_available(),
        )

        # Use the custom WeightedTrainer
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=class_weights,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
