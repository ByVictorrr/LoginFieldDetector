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
    "USERNAME": re.compile(r"(email|e-mail|phone|user|username|account|id|identifier)", re.IGNORECASE),
    "PASSWORD": re.compile(r"(pass|password|pwd|secret|key|pin|phrase)", re.IGNORECASE),
    "2FA": re.compile(r"(2fa|auth|code|otp|verification|token|one-time)", re.IGNORECASE),
    "LOGIN": re.compile(r"(login|log in|sign in|access|proceed|continue|submit|sign-on)", re.IGNORECASE),
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
        Preprocess an HTML token to combine text, parent, sibling, and metadata.
        """
        text = tag.get_text(strip=True).lower()

        # Sort and process metadata attributes
        sorted_metadata = {
            k: " ".join(sorted(v)) if isinstance(v, list) else str(v)
            for k, v in sorted(tag.attrs.items())
        }
        metadata_str = " ".join(f"[{k.upper()}:{v}]" for k, v in sorted_metadata.items())

        # Extract parent tag and sibling information
        parent_tag = f"[PARENT:{tag.parent.name}]" if tag.parent else "[PARENT:NONE]"
        previous_sibling = f"[PREV_SIBLING:{tag.find_previous_sibling().name}]" if tag.find_previous_sibling() else "[PREV_SIBLING:NONE]"
        next_sibling = f"[NEXT_SIBLING:{tag.find_next_sibling().name}]" if tag.find_next_sibling() else "[NEXT_SIBLING:NONE]"

        # Combine all into a single string
        combined_input = f"[TAG:{tag.name}] {f'[TEXT:{text}]' if text else ''} {parent_tag} {previous_sibling} {next_sibling} {metadata_str}"
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
                        "display: none" in tag.attrs.get("style", ""),
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
        text = tag.get_text(strip=True).lower()  # Extract the visible text inside the tag
        input_type = tag.attrs.get("type", "text").lower()  # Default to "text" if no type

        # Normalize attributes: lowercase keys, convert lists to space-separated strings
        attributes = {
            k.lower(): (v if isinstance(v, str) else " ".join(v) if isinstance(v, list) else "")
            for k, v in tag.attrs.items()
        }

        # If the tag is an input element
        if tag.name == "input":
            # Check for USERNAME
            if PATTERNS["USERNAME"].search(text) or any(PATTERNS["USERNAME"].search(v) for v in attributes.values()):
                return "USERNAME"

            # Check for PASSWORD
            if (
                    input_type == "password"
                    or PATTERNS["PASSWORD"].search(text)
                    or any(PATTERNS["PASSWORD"].search(v) for v in attributes.values())
                    or tag.attrs.get("type", "").lower() == "password"
            ):
                return "PASSWORD"

                # Check for 2FA
            if PATTERNS["2FA"].search(text) or any(PATTERNS["2FA"].search(v) for v in attributes.values()):
                return "2FA"

        # Process 'button' tags
        if tag.name == "button":
            if PATTERNS["LOGIN"].search(text) or attributes.get("type") == "submit":
                return "LOGIN"

        # Process 'a' tags
        if tag.name == "a":
            if any(provider in text for provider in self.oauth_providers):
                return "OAUTH"
            if "next" in text or "continue" in text:
                return "NEXT"

        # Process 'iframe' tags
        if tag.name == "iframe":
            if any(provider in text for provider in self.oauth_providers):
                return "OAUTH"

        # Default label
        return "O"

