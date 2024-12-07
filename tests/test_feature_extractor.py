import os
import logging
import pytest
from bs4 import BeautifulSoup
from login_field_detector import determine_label, HTMLFeatureExtractor, LABEL2ID

log = logging.getLogger(__file__)


@pytest.fixture(scope="module")
def extractor():
    """Fixture for HTMLFeatureExtractor."""
    return HTMLFeatureExtractor(LABEL2ID)


@pytest.mark.parametrize("html_file", [
    "crunchyroll.html",
    "dailymotion.html",
    "etsy.html",
])
def test_html_extraction(extractor, html_file):
    """Test feature extraction from real URLs."""
    file_path = os.path.join(os.path.dirname(__file__), "feature_extraction", "valid", html_file)
    assert os.path.exists(file_path), f"{html_file} does not exist!"
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    tokens, labels, xpaths = extractor.get_features(html_text=html_content)
    assert len(tokens) == len(labels), f"Tokens and labels mismatch for {html_file}"


@pytest.mark.parametrize(
    "html_snippet, expected_label",
    [
        ('<input type="text" name="username">', "USERNAME"),
        ('<input type="password" name="password">', "PASSWORD"),
        ('<button type="submit">Login</button>', "LOGIN_BUTTON"),
    ]
)
def test_determine_label(html_snippet, expected_label):
    """Test determine_label function."""
    soup = BeautifulSoup(html_snippet, "lxml")
    tag = soup.find()
    assert determine_label(tag) == expected_label


@pytest.mark.parametrize("html_file", [
    "crunchyroll.html",
    "dailymotion.html",
    "etsy.html",
    "eventbrite.html",
    "gitlab.html",
    "patreon.html",
    "saksfifthavenue.html",
    "skillshare.html",
    "stack_exchange.html",
    "stack_overflow.html",
    "starz.html",
    "swappa.html",
    "toptal.html",
])
def test_valid_html_extraction(html_file):
    """Test feature extraction with valid cached HTML."""
    file_path = os.path.join(os.path.dirname(__file__), "valid", html_file)
    assert os.path.exists(file_path), f"{html_file} does not exist!"

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    extractor = HTMLFeatureExtractor(LABEL2ID)
    tokens, labels, xpaths = extractor.get_features(html_text=html_content)
    assert len(tokens) == len(labels), f"Tokens and labels mismatch for {html_file}"


@pytest.mark.parametrize("html_file", [
    "js_rendered_form.html",
    "large_page.html",
    "malformed.html",
    "minimal.html",
    "missing_names.html",
    "mixed_content.html",
    "non_login.html",
    "overloaded_attributes.html",
    "uncommon_attributes.html",
    "unlabeled_inputs.html",
])
def test_invalid_html_extraction(html_file):
    """Test feature extraction with invalid cached HTML."""
    file_path = os.path.join(os.path.dirname(__file__), "invalid", html_file)
    assert os.path.exists(file_path), f"{html_file} does not exist!"

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    extractor = HTMLFeatureExtractor(LABEL2ID)
    try:
        tokens, labels, _ = extractor.get_features(html_text=html_content)
        assert len(tokens) == len(labels), f"Tokens and labels mismatch for {html_file}"
    except Exception as e:
        pytest.fail(f"Extraction failed for {html_file}: {e}")
