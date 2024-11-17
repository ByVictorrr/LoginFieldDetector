import functools
import json
import os.path

from collections import defaultdict
from transformers import BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset
import torch
import torch.nn.functional as F

from .html_serializer import parse_html, fetch_html, LABELS


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


class LoginFieldDetector:
    def __init__(self, model_name="bert-base-uncased", labels=None):
        self.labels = labels or LABELS
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id
        )

    def tokenize_and_align_labels(self, tokens, labels):
        """Tokenize tokens and align labels."""
        inputs = self.tokenizer(tokens, truncation=True, padding=True, is_split_into_words=True, return_tensors="pt")
        word_ids = inputs.word_ids(batch_index=0)  # Map back to original words
        aligned_labels = []

        current_word_idx = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != current_word_idx:
                aligned_labels.append(labels[word_id])
                current_word_idx = word_id
            else:
                aligned_labels.append(-100)

        inputs["labels"] = torch.tensor([aligned_labels])
        return inputs

    def process_urls(self, url_list):
        """Process multiple URLs and generate tokens/labels."""
        tokens, labels = [], []
        for url in url_list:
            print(f"Processing {url}")
            html = fetch_html(url)
            if html:
                t, l = parse_html(html, self.label2id)
                tokens.extend(t)
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

    def train(self, urls=TRAINING_URLS, output_dir="./results", epochs=20, batch_size=4, learning_rate=5e-5):
        """Train the model."""
        from transformers import Trainer, TrainingArguments
        dataset = self.create_dataset(urls)
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # Use the same dataset for simplicity
            data_collator=data_collator,
        )
        trainer.train()

    def predict(self, html):
        tokens, _ = parse_html(html, self.label2id)
        inputs = self.tokenize_and_align_labels(tokens, [0] * len(tokens))
        outputs = self.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1).tolist()  # Get probabilities
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()[0]

        for token, pred, prob in zip(tokens, predictions, probabilities):
            print(f"Token: {token}, Predicted Label: {self.id2label[pred]}, Probabilities: {prob}")

        result = [{"token": token, "label": self.id2label[pred]} for token, pred in zip(tokens, predictions) if
                  pred != -100]
        return result


if __name__ == "__main__":
    pass
