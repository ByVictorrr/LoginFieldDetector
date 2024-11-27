import logging
import os
import re
import json
from bs4 import BeautifulSoup
from babel import Locale

log = logging.getLogger(__file__)


def get_langs():
    """Retrieves a list of all languages with their English and native names using Babel."""
    languages = []
    for code in Locale("en").languages.keys():
        try:
            locale = Locale.parse(code)
            name = locale.get_display_name("en").lower()
            native = locale.get_display_name(code).lower()
            if name == native:
                languages.append(name)
            else:
                languages.extend([name, native])
        except Exception as e:
            log.error(f"Error processing language code '{code}': {e}")
    return languages


langs = get_langs()


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

LABELS = keywords["labels"]
PATTERNS = {
    "FORGOT_PASSWORD": re.compile(
        r"(forgot (?:password|account)|reset password|can't access|retrieve|trouble signing in|recover your account)",
        re.IGNORECASE
    ),
    "ADVERTISEMENTS": re.compile(
        r"(ad|advertisement|promo|sponsored|ads by|learn more|check out|special offer|deal)",
        re.IGNORECASE
    ),
    "NAVIGATION_LINK": re.compile(
        r"\b(home|back|next|previous|main\s*menu|navigation|navigate|main\s*page|show more|view details|dashboard|explore site)\b",
        re.IGNORECASE
    ),

    "HELP_LINK": re.compile(
        r"(help|support|faq|contact us|need assistance|get help|troubleshoot|customer service)",
        re.IGNORECASE
    ),
    "LANGUAGE_SWITCH": re.compile(
        fr"(\b({'|'.join([re.escape(lang) for lang in langs])})\b)", re.IGNORECASE
    ),

    "SIGN_UP": re.compile(
        r"(sign up|register|create account|join now|get started|new here|enroll|begin)",
        re.IGNORECASE
    ),
    "REMEMBER_ME": re.compile(
        r"(remember me|stay signed in|keep me logged in|remember login|save session)",
        re.IGNORECASE
    ),
    "PRIVACY_POLICY": re.compile(
        r"(privacy policy|data protection|terms of privacy|gdpr|your privacy|privacy settings)",
        re.IGNORECASE
    ),
    "TERMS_OF_SERVICE": re.compile(
        r"(terms of service|terms and conditions|user agreement|tos|terms of use)",
        re.IGNORECASE
    ),
    "BANNER": re.compile(
        r"(banner|announcement|alert|header|promotion|notification|pop-up|headline)",
        re.IGNORECASE
    ),
    "COOKIE_POLICY": re.compile(
        r"(cookie policy|cookies|tracking policy|data usage|we use cookies|accept cookies)",
        re.IGNORECASE
    ),
    "IMPRINT": re.compile(
        r"(imprint|legal notice|about us|company details|contact info|disclaimer|company profile)",
        re.IGNORECASE
    ),
    # Important
    "LOGIN_BUTTON": re.compile(
        r"(log\s*in|sign\s*in|sign\s*on|access account|proceed to login|continue to login|submit credentials|enter account|login now)",
        re.IGNORECASE
    ),
    "USERNAME": re.compile(
        r"(e-?mail|phone|user(?:name)?|login(?: name)?|account(?: name)?|(?:account\s)?identifier|profile name)",
        re.IGNORECASE
    ),
    "PHONE_NUMBER": re.compile(
        r"(phone|mobile|contact number|cell|telephone|call us)",
        re.IGNORECASE
    ),
    "PASSWORD": re.compile(
        r"(pass|password|pwd|secret|key|pin|phrase|access code|security word)",
        re.IGNORECASE
    ),

    "CAPTCHA": re.compile(
        r"(captcha|i'm not a robot|security check|verify|prove you're human|challenge|reCAPTCHA)",
        re.IGNORECASE
    ),
    "SOCIAL_LOGIN_BUTTONS": re.compile(
        fr"(login with|sign in with|connect with|continue with|authenticate with)\s+({'|'.join(keywords['oauth_providers'])})",
        re.IGNORECASE
    ),
    "TWO_FACTOR_AUTH": re.compile(
        r"(2fa|authenticator|verification code|token|one-time code|security key|two-step verification|otp)",
        re.IGNORECASE
    ),
}


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
            return label

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
