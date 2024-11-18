import re
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
        response = requests.get(url, timeout=10, allow_redirects=True)
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


class HTMLFeatureExtractor:
    def __init__(self, label2id, oauth_providers=None):
        """
        Initialize the extractor with label mappings and optional OAuth providers.
        :param label2id: Mapping of labels to IDs.
        :param oauth_providers: List of OAuth provider names (e.g., ["google", "facebook"]).
        """
        self.label2id = label2id
        self.oauth_providers = oauth_providers or OAUTH_PROVIDERS

    @staticmethod
    def extract_input_features(tag):
        """Extract features for input elements."""
        input_type = tag.get("type", "text").lower()
        name = tag.get("name", "").lower()
        placeholder = tag.get("placeholder", "").lower()
        patterns = {
            "USERNAME": re.compile(r"(email|phone|user)", re.IGNORECASE),
            "2FA": re.compile(r"(2fa|auth|code|otp)", re.IGNORECASE),
        }

        def matches_any(strings, candidates):
            return any(s in candidates for s in strings)

        if input_type in ["email", "user", "username"] or patterns["USERNAME"].search(name + placeholder) or matches_any(["email", "user", "username"], tag.attrs.values()):
            label = "USERNAME"
        elif input_type == "password" or matches_any(["pass", "password"], tag.attrs.values()):
            label = "PASSWORD"
        elif input_type == "text" and patterns["2FA"].search(placeholder):
            label = "2FA"
        else:
            label = "O"

        return label

    def extract_other_features(self, tag):
        """Extract features for other elements like button, a, iframe."""
        text = tag.get_text(strip=True).lower()
        href = tag.get("href", "").lower()
        attrs = " ".join(f"{k} {v}" for k, v in tag.attrs.items()).lower()

        def matches_any(strings, candidates):
            return any(s in candidates for s in strings)

        if matches_any(["login", "log in", "sign in"], text) or matches_any(["login"], attrs):
            label = "LOGIN"
        elif matches_any(["next", "continue"], text):
            label = "NEXT"
        elif matches_any(self.oauth_providers, text + href):
            label = "OAUTH"
        else:
            label = "O"
        return label

    def get_features(self, html):
        """Extract tokens, labels, and xpaths from HTML."""
        soup = BeautifulSoup(html, "lxml")
        tokens = []
        labels = []
        xpaths = []

        for tag in soup.find_all(["input", "button", "a", "iframe"]):
            if tag.name == "input" and tag.attrs.get("type") == "hidden":
                continue  # Skip hidden inputs

            if tag.name == "input":
                label = self.extract_input_features(tag)
            else:
                label = self.extract_other_features(tag)

            # Combine attributes to create the token

            # Add token and label
            token = {
                "tag": tag.name,
                "text": tag.get_text(strip=True).lower(),
                "attrs": " ".join(f"{k}={v}" for k, v in tag.attrs.items() if v),
            }
            tokens.append(" ".join(filter(None, token.values())))
            labels.append(self.label2id[label])
            xpaths.append(get_xpath(tag))

        return tokens, labels, xpaths
