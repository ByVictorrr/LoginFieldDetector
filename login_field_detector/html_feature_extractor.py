import re
import json
import os
import logging
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from login_field_detector import DataLoader


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
LABELS = [
    "UNLABELED",
    "USERNAME",
    "PHONE_NUMBER",
    "PASSWORD",
    "LOGIN_BUTTON",
    "TWO_FACTOR_AUTH",
    "SOCIAL_LOGIN_BUTTONS",
    "CAPTCHA",
    # below does not count
    "FORGOT_PASSWORD",
    "SIGN_UP",
    "REMEMBER_ME",
    "HELP_LINK",
    "PRIVACY_POLICY",
    "TERMS_OF_SERVICE",
    "NAVIGATION_LINK",
    "BANNER",
    "ADVERTISEMENTS",
    "COOKIE_POLICY",
    "IMPRINT"
]

PATTERNS = {
    "USERNAME": re.compile(r"(email|e-mail|phone|user|username|account|id|identifier)", re.IGNORECASE),
    "PHONE_NUMBER": re.compile(r"(phone|mobile|contact number|cell)", re.IGNORECASE),
    "PASSWORD": re.compile(r"(pass|password|pwd|secret|key|pin|phrase)", re.IGNORECASE),
    "LOGIN_BUTTON": re.compile(r"(login|log in|sign in|access|proceed|continue|submit|sign-on)", re.IGNORECASE),
    "FORGOT_PASSWORD": re.compile(r"(forgot password|reset password|can't access|retrieve)", re.IGNORECASE),
    "CAPTCHA": re.compile(r"(captcha|i'm not a robot|security check|verify)", re.IGNORECASE),
    "TWO_FACTOR_AUTH": re.compile(r"(2fa|authenticator|verification code|token|one-time code)", re.IGNORECASE),
    "SOCIAL_LOGIN_BUTTONS": re.compile(r"(login with|sign in with|connect with|continue with)", re.IGNORECASE),
    "SIGN_UP": re.compile(r"(sign up|register|create account|join now|get started)", re.IGNORECASE),
    "REMEMBER_ME": re.compile(r"(remember me|stay signed in|keep me logged in)", re.IGNORECASE),
    "HELP_LINK": re.compile(r"(help|support|faq|contact us|need assistance)", re.IGNORECASE),
    "PRIVACY_POLICY": re.compile(r"(privacy policy|data protection|terms of privacy|gdpr)", re.IGNORECASE),
    "TERMS_OF_SERVICE": re.compile(r"(terms of service|terms and conditions|user agreement)", re.IGNORECASE),
    "NAVIGATION_LINK": re.compile(r"(home|back|next|previous|menu|navigate|main page)", re.IGNORECASE),
    "BANNER": re.compile(r"(banner|announcement|alert|header|promotion)", re.IGNORECASE),
    "ADVERTISEMENTS": re.compile(r"(ad|advertisement|promo|sponsored|ads by)", re.IGNORECASE),
    "COOKIE_POLICY": re.compile(r"(cookie policy|cookies|tracking policy|data usage)", re.IGNORECASE),
    "IMPRINT": re.compile(r"(imprint|legal notice|about us|company details|contact info)", re.IGNORECASE),
}


def preprocess_field(tag):
    """Preprocess an HTML token to combine text, parent, sibling, and metadata."""
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
    return f"[TAG:{tag.name}] {f'[TEXT:{text}]' if text else ''} {parent_tag} {previous_sibling} {next_sibling} {metadata_str}"


def determine_label(tag):
    """Determine the label of an HTML tag based on patterns."""
    text = tag.get_text(strip=True).lower()  # Extract the visible text inside the tag

    # Normalize attributes: lowercase keys, convert lists to space-separated strings
    attributes = {
        k.lower(): (v if isinstance(v, str) else " ".join(v) if isinstance(v, list) else "")
        for k, v in tag.attrs.items()
    }

    # Check patterns for labels
    for label, pattern in PATTERNS.items():
        if pattern.search(text) or any(pattern.search(v) for v in attributes.values()):
            return label

    # Default label
    return LABELS[0]


class HTMLFeatureExtractor:
    def __init__(self, label2id, oauth_providers=None):
        """Initialize the extractor with label mappings and optional OAuth providers."""
        self.label2id = label2id
        if not oauth_providers:
            oauth_file = os.path.join(os.path.dirname(__file__), "oauth_providers.json")
            with open(oauth_file, "r") as flp:
                oauth_providers = json.load(flp)
        self.oauth_providers = oauth_providers

    def get_features(self, url=None, file_path=None):
        """Extract tokens, labels, xpaths, and bounding boxes from an HTML file."""
        # Read and parse the HTML
        if not url and not file_path:
            raise ValueError(f"{file_path=} and {url=} can not be None. One has to be used.")
        if url:
            with DataLoader() as dataloader:
                file_path = dataloader.fetch_html(url)

        with open(file_path, "r", encoding="utf-8") as html_fp:
            html_text = html_fp.read()
        soup = BeautifulSoup(html_text, "lxml")

        tokens, labels, xpaths, bboxes = [], [], [], []

        # Start Playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()

            # Load the local HTML file
            page.goto(f"file:///{os.path.abspath(file_path)}")

            # Process relevant tags
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
                label = determine_label(tag)  # Replace with your actual logic

                # Generate XPath
                xpath = get_xpath(tag)  # Replace with your XPath generation logic

                # Locate the element in Playwright
                try:
                    element = page.locator(f"xpath={xpath}")
                    bbox = element.bounding_box()
                except Exception as e:
                    print(f"Error locating element for XPath {xpath}: {e}")
                    bbox = None

                # Preprocess token
                preprocessed_token = preprocess_field(tag)  # Replace with your preprocessing logic

                # Append results
                tokens.append(preprocessed_token)
                labels.append(self.label2id[label])
                xpaths.append(xpath)
                bboxes.append(bbox)

            browser.close()

        return tokens, labels, xpaths, bboxes
