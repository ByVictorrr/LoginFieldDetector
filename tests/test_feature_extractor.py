import json
import os
import logging
import pytest
from bs4 import BeautifulSoup
from login_field_detector import determine_label, HTMLFetcher, HTMLFeatureExtractor, LABELS

log = logging.getLogger(__file__)

with open(os.path.join(os.path.dirname(__file__), "test_training_urls.json"), "r") as fp:
    TEST_URLS = json.load(fp=fp)


@pytest.fixture(scope="module")
def fetcher():
    """Fixture for HTMLFetcher."""
    return HTMLFetcher()


@pytest.fixture(scope="module")
def extractor():
    """Fixture for HTMLFeatureExtractor."""
    label2id = {label: idx for idx, label in enumerate(LABELS)}
    return HTMLFeatureExtractor(label2id)


@pytest.mark.parametrize("url", TEST_URLS)
def test_html_extraction(fetcher, extractor, url):
    """Test feature extraction from real URLs."""
    html_content = fetcher.fetch_html(url)
    tokens, labels, xpaths = extractor.get_features(html_content)
    assert len(tokens) == len(labels), f"Mismatch in tokens and labels for {url}"


@pytest.mark.parametrize(
    "html_snippet, expected_label",
    [
        ('<input type="text" name="username">', "USERNAME"),
        ('<input type="password" name="password">', "PASSWORD"),
        ('<button type="submit">Login</button>', "LOGIN_BUTTON"),
        ('<div>Non-related content</div>', "UNLABELED"),
        ('<input type="hidden" name="csrf_token">', "UNLABELED"),
    ]
)
def test_determine_label(html_snippet, expected_label):
    """Test determine_label function."""
    soup = BeautifulSoup(html_snippet, "lxml")
    tag = soup.find()
    assert determine_label(tag) == expected_label
