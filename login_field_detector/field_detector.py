import os
import functools
import json
from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForSequenceClassification,
    Trainer,
    TrainingArguments,
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
        return json.load(flp)


TRAINING_URLS = get_training_urls()
GIT_DIR = os.path.dirname(os.path.dirname(__file__))


class LoginFieldDetector:
    def __init__(self, model_dir=f"{GIT_DIR}/fine_tuned_model", labels=None, device=None):
        self.labels = labels or LABELS
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use LayoutLM-specific tokenizer and model
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

        if os.path.exists(model_dir):
            try:
                self.model = LayoutLMForSequenceClassification.from_pretrained(
                    "microsoft/layoutlm-base-uncased",
                    num_labels=len(self.labels),
                    id2label=self.id2label,
                    label2id=self.label2id,
                )
            except Exception as e:
                print(f"Error loading model from {model_dir}: {e}")
                print("Initializing a new LayoutLM model...")
                self.model = LayoutLMForSequenceClassification.from_pretrained(
                    "microsoft/layoutlm-base-uncased",
                    num_labels=len(self.labels),
                    id2label=self.id2label,
                    label2id=self.label2id,
                )
        else:
            print("No checkpoint found. Initializing a new LayoutLM model...")
            self.model = LayoutLMForSequenceClassification.from_pretrained(
                "microsoft/layoutlm-base-uncased",
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
        :return: All tokens, labels, and bounding boxes.
        """
        all_tokens, all_labels, all_bboxes = [], [], []

        for url in tqdm(url_list, desc="Processing URLs"):
            try:
                html = fetch_html(url)
                if html:
                    tokens, labels, _, bboxes = self.feature_extractor.get_features(html)

                    if tokens and labels:
                        # Calculate the ratio of "O" labels
                        o_label_count = sum(1 for label in labels if label == self.feature_extractor.label2id["O"])
                        total_labels = len(labels)
                        o_label_ratio = o_label_count / total_labels

                        # Apply threshold
                        if o_label_ratio <= o_label_ratio_threshold:
                            # Accept the full dataset if ratio is within threshold
                            all_tokens.extend(tokens)
                            all_labels.extend(labels)
                            all_bboxes.extend(bboxes)
                        else:
                            # Downsample "O" labels to meet the threshold
                            filtered_tokens, filtered_labels, filtered_bboxes = [], [], []
                            for token, label, bbox in zip(tokens, labels, bboxes):
                                if label == self.feature_extractor.label2id["O"]:
                                    if o_label_count > o_label_ratio_threshold * total_labels:
                                        o_label_count -= 1
                                        continue  # Skip some "O" labels
                                filtered_tokens.append(token)
                                filtered_labels.append(label)
                                filtered_bboxes.append(bbox)

                            all_tokens.extend(filtered_tokens)
                            all_labels.extend(filtered_labels)
                            all_bboxes.extend(filtered_bboxes)
            except Exception as e:
                print(f"Error processing {url}: {e}")

        return all_tokens, all_labels, all_bboxes

    def create_dataset(self, tokens, labels, bboxes):
        """Create a dataset compatible with LayoutLM."""
        # Ensure token and bbox lengths match
        assert len(tokens) == len(bboxes), "Mismatch between tokens and bounding boxes"

        # Filter out invalid tokens
        valid_tokens = [token for token in tokens if token.strip()]
        valid_bboxes = [bbox for token, bbox in zip(tokens, bboxes) if token.strip()]
        assert len(valid_tokens) == len(valid_bboxes), "Filtered mismatch between tokens and bounding boxes"

        # Tokenize with bounding boxes
        encodings = self.tokenizer(
            valid_tokens,
            boxes=valid_bboxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # Create dataset with encodings and labels
        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "bbox": encodings["bbox"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return Dataset.from_dict(data)

    def evaluate_model(self, dataset):
        """Evaluate the model on a given dataset."""
        predictions, true_labels = [], []

        for example in dataset:
            inputs = {
                "input_ids": example["input_ids"].unsqueeze(0).to(self.device),
                "attention_mask": example["attention_mask"].unsqueeze(0).to(self.device),
                "bbox": example["bbox"].unsqueeze(0).to(self.device),
            }
            labels = torch.tensor(example["labels"], dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            true = labels.squeeze().tolist()

            predictions.extend([preds] if isinstance(preds, int) else preds)
            true_labels.extend([true] if isinstance(true, int) else true)

        print("Evaluation Results:")
        print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        target_names = [self.id2label[i] for i in sorted(set(true_labels + predictions))]
        print(classification_report(true_labels, predictions, target_names=target_names))

    def train(self, urls=TRAINING_URLS, output_dir=f"{GIT_DIR}/fine_tuned_model", epochs=1, batch_size=4):
        """Train the model with cached HTML."""
        tokens, labels, bboxes = self.process_urls(urls)

        if len(tokens) < 2:
            raise ValueError("Dataset is too small. Provide more URLs or check the HTML parser.")

        # Fit the vectorizer on meta features
        self.feature_extractor.fit_meta_features(tokens)

        dataset = self.create_dataset(tokens, labels, bboxes)

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
        tokens, _, xpaths, bboxes = self.feature_extractor.get_features(html_text)
        encodings = self.tokenizer(
            tokens,
            boxes=bboxes,
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

        return [{"token": token, "predicted_label": self.id2label[pred], "xpath": xpath}
                for token, pred, xpath in zip(tokens, predicted_ids, xpaths)]


if __name__ == "__main__":
    detector = LoginFieldDetector()
    detector.train()
