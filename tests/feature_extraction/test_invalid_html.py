import os
import pytest
from login_field_detector import HTMLFeatureExtractor, LABEL2ID


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
