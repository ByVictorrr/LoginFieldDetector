import functools
import json
import os.path

import requests
from bs4 import BeautifulSoup


@functools.cache
def load_oauth_names():
    with open(os.path.join(os.path.dirname(__file__), "oauth_providers.json")) as flp:
        data = json.load(flp)
    return data


def fetch_html(url):
    """Fetch HTML content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


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


def parse_html(html, label2id):
    """Parse HTML, extract elements, and include XPath as the token."""
    soup = BeautifulSoup(html, "lxml")
    tokens = []
    labels = []

    for tag in soup.find_all(["input", "button", "a", "iframe"]):
        # xpath = get_xpath(tag)
        label = "O"  # Default label

        if tag.name == "input":
            input_type = tag.get("type", "text").lower()  # Default to "text"
            name = tag.get("name", "").lower()
            placeholder = tag.get("placeholder", "").lower()

            if input_type in ["email", "text"] and any(k in name for k in ["email", "phone", "user"]):
                label = "USERNAME"
            elif input_type == "password":
                label = "PASSWORD"
            elif input_type == "text" and any(k in placeholder for k in ["2fa", "auth", "code"]):
                label = "2FA"

        elif tag.name in ["button", "a", "iframe"]:
            text = tag.get_text(strip=True).lower()
            href = tag.get("href", "").lower()

            # Check for specific keywords in text or href
            if any(k in text for k in ["login", "log in", "sign in"]):
                label = "LOGIN"
            elif any(k in text for k in ["next", "continue"]):
                label = "NEXT"
            elif any(provider in text for provider in OAUTH_PROVIDERS) or any(provider in href for provider in OAUTH_PROVIDERS):
                label = "OAUTH"

        # Debug unmatched tags
        if label == "O":
            print(f"Unmatched tag: {tag}")

        # Append token and label
        # tokens.append(xpath)
        tokens.append(str(tag))
        labels.append(label2id[label])

    return tokens, labels

