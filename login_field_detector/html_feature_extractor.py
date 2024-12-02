import logging
import os
import re
import json
from bs4 import BeautifulSoup
from babel import Locale

log = logging.getLogger(__file__)


def get_xpath(element):
    """Generate XPath for a given BeautifulSoup element."""
    parts = []
    while element:
        siblings = element.find_previous_siblings(element.name)
        position = len(siblings) + 1  # XPath is 1-indexed
        parts.insert(0, f"{element.name}[{position}]")
        element = element.parent
    return "/" + "/".join(parts)


with open(os.path.join(os.path.dirname(__file__), "keywords.json"), "r") as key_fp:
    keywords = json.load(key_fp)


# Function to generate language regex dynamically
def generate_language_switch():
    langs = []
    for code in Locale("en").languages.keys():
        try:
            locale = Locale.parse(code)
            name = locale.get_display_name("en").lower()
            native = locale.get_display_name(code).lower()
            langs.append(name)
            if name != native:
                langs.append(native)
        except Exception as e:
            log.debug(f"Error processing language code '{code}': {e}")
    return "|".join(map(re.escape, set(langs)))  # Escape for regex safety


LABELS = keywords["labels"]
PATTERNS = {key: re.compile(value, re.IGNORECASE) for key, value in keywords["label_regexes"].items()}
PATTERNS["LANGUAGE_SWITCH"] = re.compile(fr"(\b({generate_language_switch()})\b)", re.IGNORECASE)


def preprocess_field(tag):
    """Preprocess an HTML token to include text, parent, sibling, and metadata."""
    text = tag.get_text(strip=True).lower()
    parent_text = tag.parent.get_text(strip=True).lower() if tag.parent else ""
    prev_sibling_text = tag.find_previous_sibling().get_text(strip=True).lower() if tag.find_previous_sibling() else ""
    next_sibling_text = tag.find_next_sibling().get_text(strip=True).lower() if tag.find_next_sibling() else ""

    # Collect metadata
    sorted_metadata = {k: " ".join(sorted(v)) if isinstance(v, list) else str(v) for k, v in tag.attrs.items()}
    metadata_str = " ".join(f"[{k.upper()}:{v}]" for k, v in sorted_metadata.items())

    # Combine fields
    return f"[TAG:{tag.name}] [TEXT:{text}] [PARENT:{parent_text}] [PREV_SIBLING:{prev_sibling_text}] " \
           f"[NEXT_SIBLING:{next_sibling_text}] {metadata_str}"


def determine_label(tag):
    """Determine the label of an HTML tag based on patterns."""
    text = tag.get_text(strip=True).lower()  # Extract the visible text inside the tag

    # Normalize attributes: lowercase keys, convert lists to space-separated strings
    attributes = {
        k.lower(): (v if isinstance(v, str) else " ".join(v) if isinstance(v, list) else "")
        for k, v in tag.attrs.items()
    }

    # Check patterns for label78:2B:64:CE:21:A7s
    for label, pattern in PATTERNS.items():
        if pattern.search(text) or any(pattern.search(v) for v in attributes.values()):
            if label in keywords["input_labels"] and tag.name == "input":
                return label
            elif label not in keywords["input_labels"] and tag.name != "input":
                return label
            else:
                continue
    # Default label
    return LABELS[0]


def is_item_visible(tag):
    return not any([tag.attrs.get("type") == "hidden",
                    "hidden" in tag.attrs.get("class", []),
                    "display: none" in tag.attrs.get("style", ""),
                    ])


class HTMLFeatureExtractor:
    def __init__(self, label2id, oauth_providers=None):
        """Initialize the extractor with label mappings and optional OAuth providers."""
        self.label2id = label2id
        if not oauth_providers:
            oauth_file = os.path.join(os.path.dirname(__file__), "keywords.json")
            with open(oauth_file, "r") as flp:
                oauth_providers = json.load(flp)
        self.oauth_providers = oauth_providers

    def get_features(self, html_text):
        """Extract tokens, labels, xpaths, and bounding boxes from an HTML file."""
        # Read and parse the HTML
        soup = BeautifulSoup(html_text, "lxml")

        tokens, labels, xpaths = [], [], []

        # Process relevant tags
        for tag in soup.find_all(["input", "button", "a", "iframe"]):
            # Skip irrelevant tags
            if not is_item_visible(tag):
                continue

            # Determine the label
            label = determine_label(tag)  # Replace with your actual logic

            # Generate XPath
            xpath = get_xpath(tag)  # Replace with your XPath generation logic
            # Preprocess token
            preprocessed_token = preprocess_field(tag)  # Replace with your preprocessing logic

            # Append results
            tokens.append(preprocessed_token)
            labels.append(self.label2id[label])
            xpaths.append(xpath)

        return tokens, labels, xpaths
