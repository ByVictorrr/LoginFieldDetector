"""

"""
import re
import logging
import functools
import json
import os.path
from sklearn.feature_extraction import DictVectorizer
from bs4 import BeautifulSoup
from transformers import BertTokenizerFast


@functools.cache
def load_oauth_names():
    with open(os.path.join(os.path.dirname(__file__), "oauth_providers.json")) as flp:
        data = json.load(flp)
    return data


def get_xpath(element):
    """Generate XPath for a given BeautifulSoup element."""
    parts = []
    while element:
        siblings = element.find_previous_siblings(element.name)
        position = len(siblings) + 1  # XPath is 1-indexed
        parts.insert(0, f"{element.name}[{position}]")
        element = element.parent
    return "/" + "/".join(parts)


LABELS = ["O", "USERNAME", "PASSWORD", "2FA", "NEXT", "LOGIN", "OAUTH"]

OAUTH_PROVIDERS = load_oauth_names()

PATTERNS = {
    "USERNAME": re.compile(r"(email|phone|user|username|account|login)", re.IGNORECASE),
    "PASSWORD": re.compile(r"(pass|password|pwd|secret)", re.IGNORECASE),
    "2FA": re.compile(r"(2fa|auth|code|otp|verification|token)", re.IGNORECASE),
    "LOGIN": re.compile(r"(login|log in|sign in|continue)", re.IGNORECASE),
}


class HTMLFeatureExtractor:
    def __init__(self, label2id, oauth_providers=None):
        """
        Initialize the extractor with label mappings and optional OAuth providers.
        :param label2id: Mapping of labels to IDs.
        :param oauth_providers: List of OAuth provider names (e.g., ["google", "facebook"]).
        """
        self.label2id = label2id
        self.oauth_providers = oauth_providers or OAUTH_PROVIDERS
        self.vectorizer = DictVectorizer(sparse=False)  # Initialize DictVectorizer

    def _extract_features(self, tag):
        """Extract features for any HTML tag."""
        text = tag.get_text(strip=True).lower()
        attributes = " ".join(
            "".join(attr) if isinstance(attr, list) else attr for attr in tag.attrs.values()
        ).lower()
        features = {
            "tag": tag.name,
            "text": text,
            "parent_tag": tag.find_parent().name if tag.find_parent() else "root",
            **tag.attrs,
        }

        label = "O"
        input_type = tag.get("type", "text").lower()
        if tag.name == "input":
            if PATTERNS["USERNAME"].search(attributes):
                label = "USERNAME"
            if input_type == "password" or PATTERNS["PASSWORD"].search(attributes):
                label = "PASSWORD"
            if PATTERNS["2FA"].search(attributes):
                label = "2FA"
        else:
            if PATTERNS["LOGIN"].search(text + attributes):
                label = "LOGIN"
            if any(provider in text for provider in self.oauth_providers):
                label = "OAUTH"
            if "next" in text or "continue" in text:
                label = "NEXT"

        return label, features

    def fit(self, token_features_list):
        """Fit DictVectorizer on the entire dataset."""
        self.vectorizer.fit(token_features_list)

    def transform(self, token_features_list):
        """Transform token features into numerical vectors."""
        return self.vectorizer.transform(token_features_list)

    def fit_transform(self, token_features_list):
        """Fit and transform in a single step."""
        return self.vectorizer.fit_transform(token_features_list)

    def get_features(self, html):
        """Extract tokens, labels, and xpaths from HTML."""
        soup = BeautifulSoup(html, "lxml")
        tokens, labels, xpaths = [], [], []

        for tag in soup.find_all(["input", "button", "a", "iframe"]):
            if any([tag.attrs.get("type") == "hidden",
                    "hidden" in tag.attrs.get("class", []),
                    "display:none" in tag.attrs.get("style", ""),
                    ]):
                continue
            label, token = self._extract_features(tag)
            xpath = get_xpath(tag)

            tokens.append(token)
            labels.append(self.label2id[label])
            xpaths.append(xpath)

            logging.debug(f"Processed Tag: {tag}, Token: {token}, XPath: {xpath}")

        return tokens, labels, xpaths
