import re
import json
import os
import logging
from bs4 import BeautifulSoup


def load_oauth_names():
    """Load OAuth provider names from a JSON file."""
    oauth_file = os.path.join(os.path.dirname(__file__), "oauth_providers.json")
    with open(oauth_file, "r") as flp:
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


# Label definitions
LABELS = ["O", "USERNAME", "PASSWORD", "2FA", "NEXT", "LOGIN", "OAUTH"]
PATTERNS = {
    "USERNAME": re.compile(r"(email|phone|user|username|account|login)", re.IGNORECASE),
    "PASSWORD": re.compile(r"(pass|password|pwd|secret)", re.IGNORECASE),
    "2FA": re.compile(r"(2fa|auth|code|otp|verification|token)", re.IGNORECASE),
    "LOGIN": re.compile(r"(login|log in|sign in|continue)", re.IGNORECASE),
}
OAUTH_PROVIDERS = load_oauth_names()


class HTMLFeatureExtractor:
    def __init__(self, label2id, oauth_providers=None):
        """
        Initialize the extractor with label mappings and optional OAuth providers.
        """
        self.label2id = label2id
        self.oauth_providers = oauth_providers or OAUTH_PROVIDERS

    def preprocess_field(self, tag):
        """
        Preprocess an HTML token to combine text and sorted metadata.
        """
        # Extract text and metadata
        text = tag.get_text(strip=True).lower()
        # Sort metadata by keys and values (if they are lists)
        sorted_metadata = {
            k: " ".join(sorted(v)) if isinstance(v, list) else v
            for k, v in sorted(tag.attrs.items())
        }

        # Serialize metadata into a key=value format
        metadata_str = " | ".join(f"{k}={v}" for k, v in sorted_metadata.items())

        # Combine text and metadata
        combined_input = f"{text} | {metadata_str}"
        return combined_input

    def get_features(self, html):
        """
        Extract tokens, labels, and xpaths from HTML.
        """
        soup = BeautifulSoup(html, "lxml")
        tokens, labels, xpaths = [], [], []

        for tag in soup.find_all(["input", "button", "a", "iframe"]):
            # Skip irrelevant tags
            if any(
                    [
                        tag.attrs.get("type") == "hidden",
                        "hidden" in tag.attrs.get("class", []),
                        "display:none" in tag.attrs.get("style", ""),
                    ]
            ):
                continue

            # Determine the label
            label = self._determine_label(tag)

            # Generate XPath (if needed)
            xpath = get_xpath(tag)

            # Preprocess token
            preprocessed_token = self.preprocess_field(tag)

            # Append results
            tokens.append(preprocessed_token)
            labels.append(self.label2id[label])
            xpaths.append(xpath)

        return tokens, labels, xpaths

    def _determine_label(self, tag):
        """Determine the label of an HTML tag based on patterns."""
        text = tag.get_text(strip=True).lower()
        # Sort metadata by keys and values (if they are lists)
        input_type = tag.attrs.get("type", "text").lower()
        if tag.name == "input":
            if PATTERNS["USERNAME"].search(text):
                return "USERNAME"
            if input_type == "password" or PATTERNS["PASSWORD"].search(text):
                return "PASSWORD"
            if PATTERNS["2FA"].search(text):
                return "2FA"
        else:
            if PATTERNS["LOGIN"].search(text):
                return "LOGIN"
            if any(provider in text for provider in self.oauth_providers):
                return "OAUTH"
            if "next" in text or "continue" in text:
                return "NEXT"
        return "O"
