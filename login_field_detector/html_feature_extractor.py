import re
import logging
import functools
import json
import os.path
from sklearn.feature_extraction import DictVectorizer
from bs4 import BeautifulSoup


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

    @staticmethod
    def _extract_input_features(tag):
        """Extract features for input elements."""
        input_type = tag.get("type", "text").lower()
        attributes = " ".join("".join(attr) if isinstance(attr, list) else attr for attr in tag.attrs.values()).lower()
        if PATTERNS["USERNAME"].search(attributes):
            return "USERNAME"
        if input_type == "password" or PATTERNS["PASSWORD"].search(attributes):
            return "PASSWORD"
        if PATTERNS["2FA"].search(attributes):
            return "2FA"
        return "O"

    def _extract_other_features(self, tag):
        """Extract features for other elements like button, a, iframe."""
        text = tag.get_text(strip=True).lower()
        href = tag.get("href", "").lower()
        attributes = " ".join("".join(v) if isinstance(v, list) else v for v in tag.attrs.values()).lower()

        if PATTERNS["LOGIN"].search(text + attributes):
            return "LOGIN"
        if any(provider in text + href for provider in self.oauth_providers):
            return "OAUTH"
        if "next" in text or "continue" in text:
            return "NEXT"
        return "O"

    @staticmethod
    def _generate_token(tag):
        """Create a token representation for the given tag."""
        parent = tag.find_parent()
        parent_tag = parent.name if parent else "root"
        return {
            "tag": tag.name,
            "text": tag.get_text(strip=True).lower(),
            "parent_tag": parent_tag,
            **tag.attrs.items(),
        }

    def fit(self, token_features_list):
        """Fit DictVectorizer on the entire dataset."""
        self.vectorizer.fit(token_features_list)

    def transform(self, token_features_list):
        """Transform token features into numerical vectors."""
        return self.vectorizer.transform(token_features_list)

    def get_features(self, html):
        """Extract tokens, labels, and xpaths from HTML."""
        soup = BeautifulSoup(html, "lxml")
        tokens, labels, xpaths = [], [], []

        for tag in soup.find_all(["input", "button", "a", "iframe"]):
            if tag.name == "input" and tag.attrs.get("type") == "hidden":
                continue  # Skip hidden inputs

            if tag.name == "input":
                label = self._extract_input_features(tag)
            else:
                label = self._extract_other_features(tag)

            # Generate token and XPath
            token = self._generate_token(tag)
            xpath = get_xpath(tag)

            # Append results
            tokens.append(token)
            labels.append(self.label2id[label])
            xpaths.append(xpath)

            # Debug logging
            logging.debug(f"Processed Tag: {tag}, Token: {token}, Label: {label}, XPath: {xpath}")

        # Fit the vectorizer once on the entire dataset
        if not self.vectorizer.feature_names_:  # Fit only if not fitted
            self.fit(tokens)
        # Transform features to vectors
        return self.transform(tokens), labels, xpaths
