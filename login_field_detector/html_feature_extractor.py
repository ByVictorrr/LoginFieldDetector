import re
import logging
import functools
import json
import os
from sklearn.feature_extraction import DictVectorizer
from bs4 import BeautifulSoup


@functools.cache
def load_oauth_names():
    """Load OAuth provider names from a JSON file."""
    with open(os.path.join(os.path.dirname(__file__), "oauth_providers.json")) as flp:
        return json.load(flp)


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
        self.vectorizer = DictVectorizer(sparse=False)  # Initialize DictVectorizer for tag.attrs

    @staticmethod
    def _get_bounding_box(tag):
        """
        Placeholder for bounding box extraction.
        This can be enhanced with a layout or DOM rendering mechanism.
        """
        return [0, 0, 100, 100]

    @staticmethod
    def _get_parent_tag(tag):
        """Get the parent tag name. Defaults to 'root' if no parent exists."""
        parent = tag.find_parent()
        return parent.name if parent else "root"

    def _extract_features(self, tag):
        """
        Extract features for any HTML tag.
        Includes text, tag, attributes, bounding box, and parent information.
        """
        text = tag.get_text(strip=True).lower()
        attributes = tag.attrs  # Use tag.attrs directly as meta-data

        label = "O"
        input_type = tag.get("type", "text").lower()

        if tag.name == "input":
            if PATTERNS["USERNAME"].search(str(attributes)):
                label = "USERNAME"
            elif input_type == "password" or PATTERNS["PASSWORD"].search(str(attributes)):
                label = "PASSWORD"
            elif PATTERNS["2FA"].search(str(attributes)):
                label = "2FA"
        else:
            if PATTERNS["LOGIN"].search(text + str(attributes)):
                label = "LOGIN"
            elif any(provider in text for provider in self.oauth_providers):
                label = "OAUTH"
            elif "next" in text or "continue" in text:
                label = "NEXT"

        features = {
            "tag": tag.name,
            "text": text,
            "parent_tag": self._get_parent_tag(tag),
            "attributes": attributes,
        }
        return label, features

    def fit_meta_features(self, token_features_list):
        """Fit the DictVectorizer to meta features (tag.attrs)."""
        meta_features_list = [token["attributes"] for token in token_features_list]
        self.vectorizer.fit(meta_features_list)

    def transform_meta_features(self, token_features_list):
        """Transform meta features into numerical vectors."""
        meta_features_list = [token["attributes"] for token in token_features_list]
        return self.vectorizer.transform(meta_features_list)

    def fit_transform_meta_features(self, token_features_list):
        """Fit and transform in a single step."""
        meta_features_list = [token["attributes"] for token in token_features_list]
        return self.vectorizer.fit_transform(meta_features_list)

    def get_features(self, html):
        """Extract tokens, labels, xpaths, and bounding boxes from HTML."""
        soup = BeautifulSoup(html, "lxml")
        tokens, labels, xpaths, bboxes = [], [], [], []

        for tag in soup.find_all(["input", "button", "a", "iframe"]):
            label, token = self._extract_features(tag)
            xpath = get_xpath(tag)

            bbox = self._get_bounding_box(tag)

            tokens.append(token)
            labels.append(self.label2id[label])
            xpaths.append(xpath)
            bboxes.append(bbox)

        return tokens, labels, xpaths, bboxes
