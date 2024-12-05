import os
import pytest
from login_field_detector import HTMLFeatureExtractor, LABEL2ID


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
